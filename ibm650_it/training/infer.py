from __future__ import annotations

import gc
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ibm650_it import REPO_ROOT
from ibm650_it.dataset.io import load_jsonl, relativize_record_paths, resolve_record_base, resolve_record_path, write_jsonl
from ibm650_it.eval.exact_match import compare_pit_files
from ibm650_it.eval.failure_taxonomy import classify_failure
from ibm650_it.eval.reevaluate import reevaluate_prediction_records
from ibm650_it.simh.deckio import write_deck_cards
from ibm650_it.training.prompt_templates import (
    build_chat_messages,
    build_few_shot_chat_messages,
    build_few_shot_prompt,
    build_prompt,
    wrap_pit_completion,
)
from ibm650_it.training.smoke_model import (
    load_sft_examples,
    load_smoke_model,
    predict_few_shot,
    predict_fine_tuned,
    predict_zero_shot,
)


def write_inference_request(prompt: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")
    return output_path


def normalize_completion_text(completion: str) -> str:
    completion = completion.strip()
    match = re.search(r"<PIT>\n?(.*?)\n?</PIT>", completion, flags=re.DOTALL)
    if match:
        completion = match.group(1)
    elif completion.startswith("<PIT>"):
        # Generation was truncated before </PIT>. Strip the opening tag and
        # the single newline that follows it, but preserve leading whitespace
        # on the first PIT card — dictionary cards depend on column alignment
        # and the reference deck starts with ~42 spaces of indent.
        completion = completion.removeprefix("<PIT>").lstrip("\n")
    if "</PIT>" in completion:
        completion = completion.split("</PIT>", 1)[0]
    return completion.strip("\n")


def extract_thinking_trace(completion: str) -> str:
    stripped = completion.strip()
    if not stripped:
        return ""
    if "<PIT>" in stripped:
        return stripped.split("<PIT>", 1)[0].rstrip()
    return ""


@dataclass
class PreflightTokenBudgetReport:
    """Summary of how many reference PIT decks exceed the configured max_new_tokens.

    ``over_budget`` lists the ids whose reference needs more tokens than the cap; any
    fine-tuned run on those cases will hit the cap and produce a truncated deck. The
    caller can surface this report to the user before burning GPU time on a run that
    is guaranteed to fail a band it should have passed (see the 20260406 stage_2k
    regression, where all of B2 and B3 were over-budget at max_new_tokens=1024).
    """

    max_new_tokens: int
    sample_size: int
    over_budget: list[tuple[str, int]]
    largest_ref_tokens: int

    @property
    def over_budget_count(self) -> int:
        return len(self.over_budget)

    @property
    def ok(self) -> bool:
        return self.over_budget_count == 0


def preflight_token_budget(
    *,
    reference_tokens: Iterable[tuple[str, int]],
    max_new_tokens: int,
) -> PreflightTokenBudgetReport:
    """Build a PreflightTokenBudgetReport from (id, ref_token_count) pairs.

    Pure function so the logic is trivially unit-testable without loading a real
    tokenizer or any PIT deck files.
    """
    over: list[tuple[str, int]] = []
    largest = 0
    sample = 0
    for ref_id, ref_tokens in reference_tokens:
        sample += 1
        if ref_tokens > largest:
            largest = ref_tokens
        if ref_tokens > max_new_tokens:
            over.append((str(ref_id), int(ref_tokens)))
    return PreflightTokenBudgetReport(
        max_new_tokens=max_new_tokens,
        sample_size=sample,
        over_budget=over,
        largest_ref_tokens=largest,
    )


def _measure_reference_tokens(
    *,
    reference_records: list[dict[str, Any]],
    reference_base: Path,
    tokenizer: Any,
    repo_root: Path,
    limit: int | None = None,
) -> list[tuple[str, int]]:
    """Encode each reference PIT deck with ``tokenizer`` and yield (id, token_count)."""
    pairs: list[tuple[str, int]] = []
    records = reference_records if limit is None else reference_records[:limit]
    for record in records:
        ref_rel = record.get("reference", {}).get("translate", {}).get("pit_raw_canonical")
        if not ref_rel:
            continue
        ref_path = resolve_record_path(str(ref_rel), reference_base)
        if not ref_path.exists():
            continue
        text = ref_path.read_text(encoding="utf-8")
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            continue
        pairs.append((str(record["id"]), len(encoded)))
    return pairs


def _log_preflight_report(report: PreflightTokenBudgetReport) -> None:
    """Print a human-readable preflight warning to stderr. No-op when ok."""
    if report.ok:
        print(
            f"[preflight] max_new_tokens={report.max_new_tokens} ≥ largest reference "
            f"({report.largest_ref_tokens} tokens over {report.sample_size} samples) — ok",
            file=sys.stderr,
            flush=True,
        )
        return
    preview = ", ".join(f"{ref_id}:{toks}" for ref_id, toks in report.over_budget[:5])
    print(
        f"[preflight][WARN] {report.over_budget_count}/{report.sample_size} reference PIT "
        f"decks exceed max_new_tokens={report.max_new_tokens}. "
        f"Largest reference needs {report.largest_ref_tokens} tokens. "
        f"First offenders: {preview}"
        + (f" (and {report.over_budget_count - 5} more)" if report.over_budget_count > 5 else "")
        + ". Every over-budget case will be decoded up to the cap and then truncated, "
        "which almost certainly drops a PIT dictionary tail and breaks exact match. "
        "Raise --max-new-tokens before starting the run.",
        file=sys.stderr,
        flush=True,
    )


def _load_model_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "model.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"model manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _hf_inference_runtime(torch_module: Any) -> dict[str, Any]:
    use_cuda = torch_module.cuda.is_available()
    return {
        "device": torch_module.device("cuda:0" if use_cuda else "cpu"),
        "torch_dtype": torch_module.bfloat16 if use_cuda else torch_module.float32,
        # Nemotron's Mamba path expects tensors to stay on CUDA during generation.
        # `device_map=\"auto\"` can offload layers to CPU and crash generation.
        "device_map": {"": 0} if use_cuda else None,
    }


@dataclass
class HfGenerationSession:
    tokenizer: Any
    model: Any
    device: Any
    stop_token_sequences: list[list[int]]


class StopOnTokenSequence:
    """Stop when every sequence in the batch has a </PIT>-style stop pattern at its tail.

    Batch-safe: returns True only when ALL rows in ``input_ids`` end with one of the
    configured stop sequences. A batch=1 input just reduces to the obvious case; a
    batch>1 input keeps the model running until the slowest row is done, which is
    what we want — per-sequence early stopping is handled by transformers' built-in
    ``eos_token_id`` path. This criterion is only a belt-and-braces backup for the
    case where the model emits </PIT> but skips EOS.
    """

    def __init__(self, stop_token_sequences: list[list[int]]) -> None:
        self.stop_token_sequences = [sequence for sequence in stop_token_sequences if sequence]

    @staticmethod
    def _row_length(row: Any) -> int:
        shape = getattr(row, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
        return len(row)

    def _row_matches(self, row: Any) -> bool:
        row_length = self._row_length(row)
        for sequence in self.stop_token_sequences:
            sequence_length = len(sequence)
            if row_length < sequence_length:
                continue
            tail_slice = row[-sequence_length:]
            tail = tail_slice.tolist() if hasattr(tail_slice, "tolist") else list(tail_slice)
            if tail == sequence:
                return True
        return False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        del scores, kwargs
        if not self.stop_token_sequences:
            return False
        batch_size = int(input_ids.shape[0])
        for row_idx in range(batch_size):
            if not self._row_matches(input_ids[row_idx]):
                return False
        return True


def _build_stop_token_sequences(tokenizer: Any) -> list[list[int]]:
    variants = [
        "</PIT>",
        "\n</PIT>",
        " </PIT>",
        "</PIT>\n",
        "\n</PIT>\n",
        " </PIT>\n",
    ]
    sequences: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if not encoded:
            continue
        key = tuple(encoded)
        if key in seen:
            continue
        sequences.append(encoded)
        seen.add(key)
    return sequences


def _load_hf_generation_session(
    *,
    model_dir: Path,
    mode: str,
) -> HfGenerationSession:
    manifest = _load_model_manifest(model_dir)
    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "model-backed inference requires torch, transformers, and peft"
        ) from exc

    runtime = _hf_inference_runtime(torch)

    if manifest["backend"] == "transformers_qlora" and mode == "fine_tuned":
        tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer_dir"], trust_remote_code=True)
        model = AutoPeftModelForCausalLM.from_pretrained(
            manifest["adapter_dir"],
            device_map=runtime["device_map"],
            torch_dtype=runtime["torch_dtype"],
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(manifest["model_name"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            manifest["model_name"],
            device_map=runtime["device_map"],
            torch_dtype=runtime["torch_dtype"],
            trust_remote_code=True,
        )
    if runtime["device_map"] is None:
        model = model.to(runtime["device"])
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding is required for batched generation on decoder-only models:
    # with right-padding, attention over the pad tokens corrupts the KV cache for
    # the "real" tokens and the generated sequence is nonsense. See transformers
    # PR #25921. For batch_size=1 this setting is a no-op.
    tokenizer.padding_side = "left"
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    stop_token_sequences = _build_stop_token_sequences(tokenizer)
    return HfGenerationSession(
        tokenizer=tokenizer,
        model=model,
        device=runtime["device"],
        stop_token_sequences=stop_token_sequences,
    )


def _close_hf_generation_session(session: HfGenerationSession | None) -> None:
    if session is None:
        return
    try:
        model = session.model
        tokenizer = session.tokenizer
        del model
        del tokenizer
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _generate_with_hf_model(
    *,
    prompt: str | None,
    session: HfGenerationSession,
    max_new_tokens: int = 1536,
    prompt_input_ids: Any | None = None,
) -> str:
    stopping_criteria = None
    try:
        from transformers import StoppingCriteriaList
        stopping_criteria = StoppingCriteriaList([StopOnTokenSequence(session.stop_token_sequences)])
    except ImportError:
        stopping_criteria = None
    if prompt_input_ids is None:
        if prompt is None:
            raise ValueError("either prompt or prompt_input_ids is required")
        inputs = session.tokenizer(prompt, return_tensors="pt").to(session.device)
    else:
        input_ids = prompt_input_ids
        if getattr(input_ids, "ndim", None) == 1:
            input_ids = input_ids.unsqueeze(0)
        inputs = {"input_ids": input_ids.to(session.device)}
    try:
        import torch

        context = torch.inference_mode()
    except ImportError:
        context = None
    if context is None:
        generated = session.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=session.tokenizer.pad_token_id,
            eos_token_id=session.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=True,
        )
    else:
        with context:
            generated = session.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=session.tokenizer.pad_token_id,
                eos_token_id=session.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                use_cache=True,
            )
    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return session.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _generate_with_hf_model_batch(
    *,
    prompts: list[str] | None = None,
    prompt_input_ids_list: list[Any] | None = None,
    session: HfGenerationSession,
    max_new_tokens: int = 1536,
) -> list[str]:
    """Batched greedy generation.

    Accepts either ``prompts`` (list of strings, will be tokenized here) or
    ``prompt_input_ids_list`` (list of 1-D token tensors from a chat template
    already applied). Returns one decoded completion per input, in order.

    Uses left-padding on the tokenizer so attention over the variable-length
    inputs is correct for decoder-only models. Respects the configured
    ``session.stop_token_sequences`` via :class:`StopOnTokenSequence`, which
    now waits until every row in the batch has hit a stop sequence rather
    than stopping on just the first one.
    """
    if (prompts is None) == (prompt_input_ids_list is None):
        raise ValueError("exactly one of prompts or prompt_input_ids_list is required")

    try:
        from transformers import StoppingCriteriaList
        stopping_criteria = StoppingCriteriaList([StopOnTokenSequence(session.stop_token_sequences)])
    except ImportError:
        stopping_criteria = None

    try:
        import torch
    except ImportError as exc:  # pragma: no cover — runtime requirement
        raise RuntimeError("batched generation requires torch") from exc

    if prompts is not None:
        encoded = session.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"].to(session.device)
        attention_mask = encoded["attention_mask"].to(session.device)
        # For decoded-completion slicing, count real (non-pad) input tokens per row.
        real_input_lengths = attention_mask.sum(dim=1).tolist()
    else:
        # prompt_input_ids_list: list of 1-D tensors of possibly different length.
        # Normalise each to a 1-D python list of ints, then pad on the left.
        as_lists: list[list[int]] = []
        for ids in prompt_input_ids_list:
            if hasattr(ids, "tolist"):
                row = ids.tolist()
                if isinstance(row, list) and row and isinstance(row[0], list):
                    # shape [1, seq_len] — unwrap
                    row = row[0]
            else:
                row = list(ids)
            as_lists.append(row)
        batch = session.tokenizer.pad(
            {"input_ids": as_lists},
            return_tensors="pt",
            padding=True,
        )
        input_ids = batch["input_ids"].to(session.device)
        attention_mask = batch["attention_mask"].to(session.device)
        real_input_lengths = attention_mask.sum(dim=1).tolist()

    with torch.inference_mode():
        generated = session.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=session.tokenizer.pad_token_id,
            eos_token_id=session.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=True,
        )

    padded_input_length = int(input_ids.shape[1])
    completions: list[str] = []
    for row_idx in range(generated.shape[0]):
        row_tokens = generated[row_idx]
        # generated[row] starts with the left-padded input block of length
        # padded_input_length. Everything after that is newly generated.
        new_tokens = row_tokens[padded_input_length:]
        completions.append(
            session.tokenizer.decode(new_tokens, skip_special_tokens=True)
        )
        # real_input_lengths is kept in case we want per-row prompt accounting later.
        _ = real_input_lengths[row_idx]
    return completions


def _build_hf_prompt(
    *,
    mode: str,
    source_text: str,
    few_shot_k: int,
    prompt_style: str,
    enable_thinking: bool | None,
    hf_session: HfGenerationSession,
    support_examples: list[Any] | None,
) -> tuple[str, Any | None]:
    """Build a prompt for hf_session-backed inference, separate from the generate call.

    Returns ``(prompt_text, prompt_input_ids_or_None)``. Pulled out of
    ``_predict_completion`` so the batched inference path can prepare a list of
    prompts up front, then hand them all to :func:`_generate_with_hf_model_batch`
    in a single model.generate() call.
    """
    prompt_input_ids = None
    if prompt_style == "chat":
        if not hasattr(hf_session.tokenizer, "apply_chat_template"):
            raise RuntimeError("chat prompt style requires tokenizer.apply_chat_template support")
        if mode == "few_shot":
            if support_examples is None:
                raise ValueError("few_shot inference requires support_sft")
            prompt_examples = [
                {
                    "source_text": example.source_text,
                    "completion": example.completion,
                }
                for example in support_examples[:few_shot_k]
            ]
            messages = build_few_shot_chat_messages(source_text, prompt_examples)
        else:
            messages = build_chat_messages(source_text)
        template_kwargs: dict[str, Any] = {"add_generation_prompt": True}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        try:
            prompt = hf_session.tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
            prompt_input_ids = hf_session.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                **template_kwargs,
            )
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            prompt = hf_session.tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
            prompt_input_ids = hf_session.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                **template_kwargs,
            )
        return prompt, prompt_input_ids

    if mode == "few_shot":
        if support_examples is None:
            raise ValueError("few_shot inference requires support_sft")
        prompt_examples = [
            {
                "source_text": example.source_text,
                "completion": example.completion,
            }
            for example in support_examples[:few_shot_k]
        ]
        prompt = build_few_shot_prompt(source_text, prompt_examples)
    else:
        prompt = build_prompt(source_text)
    return prompt, None


def _predict_completion(
    *,
    mode: str,
    source_text: str,
    model_dir: Path | None,
    support_sft: Path | None,
    few_shot_k: int,
    max_new_tokens: int,
    prompt_style: str = "plain",
    enable_thinking: bool | None = None,
    hf_session: HfGenerationSession | None = None,
    support_examples: list[Any] | None = None,
) -> tuple[str, str, dict[str, Any], str]:
    manifest = _load_model_manifest(model_dir) if model_dir is not None and (model_dir / "model.json").exists() else None
    if manifest is not None and manifest.get("backend") == "transformers_qlora":
        if hf_session is None:
            raise ValueError("transformers_qlora inference requires a loaded hf_session")
        prompt, prompt_input_ids = _build_hf_prompt(
            mode=mode,
            source_text=source_text,
            few_shot_k=few_shot_k,
            prompt_style=prompt_style,
            enable_thinking=enable_thinking,
            hf_session=hf_session,
            support_examples=support_examples,
        )
        raw_completion = _generate_with_hf_model(
            prompt=prompt,
            session=hf_session,
            max_new_tokens=max_new_tokens,
            prompt_input_ids=prompt_input_ids,
        )
        return (
            raw_completion,
            normalize_completion_text(raw_completion),
            {
                "backend": "transformers_qlora",
                "prompt_style": prompt_style,
                "enable_thinking": enable_thinking,
            },
            prompt,
        )
    if mode == "zero_shot":
        prediction = predict_zero_shot(source_text=source_text)
        prompt = build_prompt(source_text)
    elif mode == "few_shot":
        if support_examples is None:
            raise ValueError("few_shot inference requires support_sft")
        prediction = predict_few_shot(
            source_text=source_text,
            support_examples=support_examples,
            few_shot_k=few_shot_k,
        )
        prompt_examples = [
            {
                "source_text": example.source_text,
                "completion": example.completion,
            }
            for example in support_examples[:few_shot_k]
        ]
        prompt = build_few_shot_prompt(source_text, prompt_examples)
    elif mode == "fine_tuned":
        if model_dir is None:
            raise ValueError("fine_tuned inference requires model_dir")
        prediction = predict_fine_tuned(
            source_text=source_text,
            model_examples=load_smoke_model(model_dir),
        )
        prompt = build_prompt(source_text)
    else:
        raise KeyError(f"unknown inference mode: {mode}")
    metadata = {
        "matched_example_id": prediction.matched_example_id,
        "matched_score": prediction.matched_score,
        "support_example_ids": prediction.support_example_ids,
        "prompt_style": prompt_style,
        "enable_thinking": enable_thinking,
    }
    raw_completion = prediction.completion
    return raw_completion, normalize_completion_text(raw_completion), metadata, prompt


def run_inference(
    *,
    reference_index: Path,
    output_dir: Path,
    mode: str,
    repo_root: Path = REPO_ROOT,
    model_dir: Path | None = None,
    support_sft: Path | None = None,
    few_shot_k: int = 4,
    limit: int | None = None,
    max_new_tokens: int = 1536,
    prompt_style: str = "plain",
    enable_thinking: bool | None = None,
    preserve_raw_completion: bool = False,
    step_budget: str = "50M",
    timeout_seconds: int = 30,
    eval_mode: str = "inline",
    inference_batch_size: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_records = load_jsonl(reference_index)
    if limit is not None:
        reference_records = reference_records[:limit]
    reference_base = resolve_record_base(reference_index)
    prediction_index = output_dir / "predictions.jsonl"
    prediction_records = [
        record
        for record in load_jsonl(prediction_index)
        if resolve_record_path(str(record.get("pit_raw_canonical", "")), output_dir).exists()
    ]
    if prediction_records:
        write_jsonl(prediction_index, prediction_records)
    completed_ids = {str(record["id"]) for record in prediction_records}
    support_examples = load_sft_examples(support_sft) if mode == "few_shot" and support_sft is not None else None
    hf_session = None
    if model_dir is not None and (model_dir / "model.json").exists():
        manifest = _load_model_manifest(model_dir)
        if manifest.get("backend") == "transformers_qlora":
            hf_session = _load_hf_generation_session(model_dir=model_dir, mode=mode)

    if hf_session is not None:
        # Measure every reference PIT deck with the real model tokenizer and warn
        # up front if any exceed max_new_tokens. This catches the 20260406 failure
        # mode — where every B2/B3 reference was > 1024 tokens but max_new_tokens
        # defaulted to 1024 — before burning an hour of GPU time.
        measured = _measure_reference_tokens(
            reference_records=reference_records,
            reference_base=reference_base,
            tokenizer=hf_session.tokenizer,
            repo_root=repo_root,
        )
        report = preflight_token_budget(
            reference_tokens=measured,
            max_new_tokens=max_new_tokens,
        )
        _log_preflight_report(report)

    def _finalize_prediction(
        *,
        reference: dict[str, Any],
        prompt: str,
        raw_completion_text: str,
        completion_text: str,
        metadata: dict[str, Any],
        generation_seconds: float,
        example_started: float,
    ) -> None:
        prediction_dir = output_dir / str(reference["id"])
        prediction_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = write_inference_request(prompt, prediction_dir / "prompt.txt")
        raw_completion_path = None
        thinking_trace_path = None
        if preserve_raw_completion:
            raw_completion_path = write_inference_request(raw_completion_text, prediction_dir / "raw_completion.txt")
            thinking_trace = extract_thinking_trace(raw_completion_text)
            if thinking_trace:
                thinking_trace_path = write_inference_request(thinking_trace, prediction_dir / "thinking_trace.txt")
        candidate_cards = completion_text.splitlines()
        candidate_path = write_deck_cards(prediction_dir / "pit_raw_canonical.dck", candidate_cards)
        reference_pit = resolve_record_path(
            str(reference["reference"]["translate"]["pit_raw_canonical"]),
            reference_base,
        )
        exact_metrics = compare_pit_files(reference_pit, candidate_path)
        failure_type = classify_failure(
            candidate_cards=candidate_cards,
            exact_match=bool(exact_metrics["exact_match"]),
            assemblable=False,
            functional=False,
            assemble_status="not_evaluated" if eval_mode == "skip" else None,
        )
        prediction_record = {
            "id": reference["id"],
            "mode": mode,
            "band": reference["band"],
            "pit_raw_canonical": str(candidate_path),
            "prompt_path": str(prompt_path),
            "retrieval": metadata,
            "raw_completion_path": str(raw_completion_path) if raw_completion_path is not None else None,
            "thinking_trace_path": str(thinking_trace_path) if thinking_trace_path is not None else None,
            "metrics": exact_metrics,
            "assemble": {
                "status": "not_evaluated",
            },
            "run": {
                "status": "not_run",
            },
            "assemblable": False,
            "functional": False,
            "failure_type": failure_type,
            "timings": {
                "generation_seconds": generation_seconds,
                "evaluation_seconds": 0.0,
                "total_seconds": time.perf_counter() - example_started,
            },
        }
        prediction_records.append(relativize_record_paths(prediction_record, output_dir))
        completed_ids.add(str(reference["id"]))
        write_jsonl(prediction_index, prediction_records)

    use_batched_hf = (
        hf_session is not None
        and inference_batch_size > 1
    )

    try:
        if use_batched_hf:
            # Collect references that still need generation, then process them in
            # chunks of `inference_batch_size`. Each chunk: build prompts, do one
            # batched model.generate() call, then post-process each result.
            pending = [
                reference
                for reference in reference_records
                if str(reference["id"]) not in completed_ids
            ]
            for chunk_start in range(0, len(pending), inference_batch_size):
                chunk = pending[chunk_start : chunk_start + inference_batch_size]
                chunk_started = time.perf_counter()
                chunk_prompts: list[str] = []
                chunk_prompt_input_ids: list[Any] = []
                chunk_source_texts: list[str] = []
                any_chat_style = False
                for reference in chunk:
                    source_text = resolve_record_path(
                        str(reference["source"]["it_text_v1"]),
                        reference_base,
                    ).read_text(encoding="utf-8")
                    chunk_source_texts.append(source_text)
                    prompt, prompt_input_ids = _build_hf_prompt(
                        mode=mode,
                        source_text=source_text,
                        few_shot_k=few_shot_k,
                        prompt_style=prompt_style,
                        enable_thinking=enable_thinking,
                        hf_session=hf_session,
                        support_examples=support_examples,
                    )
                    chunk_prompts.append(prompt)
                    if prompt_input_ids is not None:
                        chunk_prompt_input_ids.append(prompt_input_ids)
                        any_chat_style = True
                generation_started = time.perf_counter()
                if any_chat_style:
                    # chat-templated inputs: use the pre-tokenised ids. All chunk entries
                    # must go through the same path — mixing chat/plain inside a chunk
                    # would desync lengths. We assume prompt_style is uniform per run.
                    if len(chunk_prompt_input_ids) != len(chunk):
                        raise RuntimeError(
                            "chat prompt_style did not produce prompt_input_ids for every "
                            "chunk entry; cannot batch"
                        )
                    raw_completions = _generate_with_hf_model_batch(
                        prompt_input_ids_list=chunk_prompt_input_ids,
                        session=hf_session,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    raw_completions = _generate_with_hf_model_batch(
                        prompts=chunk_prompts,
                        session=hf_session,
                        max_new_tokens=max_new_tokens,
                    )
                batch_generation_seconds = time.perf_counter() - generation_started
                # Amortise the batch generate cost evenly across chunk entries so
                # the per-example timings field stays meaningful.
                per_example_generation_seconds = batch_generation_seconds / len(chunk)
                for reference, prompt, raw_completion in zip(chunk, chunk_prompts, raw_completions):
                    metadata = {
                        "backend": "transformers_qlora",
                        "prompt_style": prompt_style,
                        "enable_thinking": enable_thinking,
                        "batch_size": len(chunk),
                    }
                    _finalize_prediction(
                        reference=reference,
                        prompt=prompt,
                        raw_completion_text=raw_completion,
                        completion_text=normalize_completion_text(raw_completion),
                        metadata=metadata,
                        generation_seconds=per_example_generation_seconds,
                        example_started=chunk_started,
                    )
        else:
            for reference in reference_records:
                if str(reference["id"]) in completed_ids:
                    continue
                example_started = time.perf_counter()
                source_text = resolve_record_path(
                    str(reference["source"]["it_text_v1"]),
                    reference_base,
                ).read_text(encoding="utf-8")
                generation_started = time.perf_counter()
                raw_completion_text, completion_text, metadata, prompt = _predict_completion(
                    mode=mode,
                    source_text=source_text,
                    model_dir=model_dir,
                    support_sft=support_sft,
                    few_shot_k=few_shot_k,
                    max_new_tokens=max_new_tokens,
                    prompt_style=prompt_style,
                    enable_thinking=enable_thinking,
                    hf_session=hf_session,
                    support_examples=support_examples,
                )
                generation_seconds = time.perf_counter() - generation_started
                _finalize_prediction(
                    reference=reference,
                    prompt=prompt,
                    raw_completion_text=raw_completion_text,
                    completion_text=completion_text,
                    metadata=metadata,
                    generation_seconds=generation_seconds,
                    example_started=example_started,
                )
    finally:
        _close_hf_generation_session(hf_session)

    prediction_index = write_jsonl(prediction_index, prediction_records)
    if eval_mode == "inline":
        reevaluate_summary = reevaluate_prediction_records(
            reference_index=reference_index,
            prediction_index=prediction_index,
            output_dir=output_dir,
            repo_root=repo_root,
            step_budget=step_budget,
            timeout_seconds=timeout_seconds,
        )
        prediction_index = Path(reevaluate_summary["prediction_index"])
    summary = {
        "mode": mode,
        "count": len(prediction_records),
        "prediction_index": str(prediction_index),
        "eval_mode": eval_mode,
        "prompt_style": prompt_style,
        "enable_thinking": enable_thinking,
        "preserve_raw_completion": preserve_raw_completion,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
