from __future__ import annotations

import gc
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ibm650_it import REPO_ROOT
from ibm650_it.dataset.io import load_jsonl, relativize_record_paths, resolve_record_base, resolve_record_path, write_jsonl
from ibm650_it.eval.exact_match import compare_pit_files
from ibm650_it.eval.failure_taxonomy import classify_failure
from ibm650_it.eval.reevaluate import reevaluate_prediction_records
from ibm650_it.simh.deckio import write_deck_cards
from ibm650_it.training.prompt_templates import build_few_shot_prompt, build_prompt, wrap_pit_completion
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
        completion = completion.removeprefix("<PIT>").strip()
    if "</PIT>" in completion:
        completion = completion.split("</PIT>", 1)[0]
    return completion.strip("\n")


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
    def __init__(self, stop_token_sequences: list[list[int]]) -> None:
        self.stop_token_sequences = [sequence for sequence in stop_token_sequences if sequence]

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        del scores, kwargs
        if not self.stop_token_sequences:
            return False
        for sequence in self.stop_token_sequences:
            sequence_length = len(sequence)
            if input_ids.shape[1] < sequence_length:
                continue
            tail_slice = input_ids[0][-sequence_length:]
            tail = tail_slice.tolist() if hasattr(tail_slice, "tolist") else list(tail_slice)
            if tail == sequence:
                return True
        return False


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
    prompt: str,
    session: HfGenerationSession,
    max_new_tokens: int = 1024,
) -> str:
    stopping_criteria = None
    try:
        from transformers import StoppingCriteriaList
        stopping_criteria = StoppingCriteriaList([StopOnTokenSequence(session.stop_token_sequences)])
    except ImportError:
        stopping_criteria = None
    inputs = session.tokenizer(prompt, return_tensors="pt").to(session.device)
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


def _predict_completion(
    *,
    mode: str,
    source_text: str,
    model_dir: Path | None,
    support_sft: Path | None,
    few_shot_k: int,
    max_new_tokens: int,
    hf_session: HfGenerationSession | None = None,
    support_examples: list[Any] | None = None,
) -> tuple[str, dict[str, Any], str]:
    manifest = _load_model_manifest(model_dir) if model_dir is not None and (model_dir / "model.json").exists() else None
    if manifest is not None and manifest.get("backend") == "transformers_qlora":
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
        if hf_session is None:
            raise ValueError("transformers_qlora inference requires a loaded hf_session")
        completion = _generate_with_hf_model(
            prompt=prompt,
            session=hf_session,
            max_new_tokens=max_new_tokens,
        )
        return normalize_completion_text(completion), {"backend": "transformers_qlora"}, prompt
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
    }
    return normalize_completion_text(prediction.completion), metadata, prompt


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
    max_new_tokens: int = 1024,
    step_budget: str = "50M",
    timeout_seconds: int = 30,
    eval_mode: str = "inline",
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

    try:
        for reference in reference_records:
            if str(reference["id"]) in completed_ids:
                continue
            example_started = time.perf_counter()
            prediction_dir = output_dir / str(reference["id"])
            prediction_dir.mkdir(parents=True, exist_ok=True)
            source_text = resolve_record_path(str(reference["source"]["it_text_v1"]), reference_base).read_text(encoding="utf-8")
            generation_started = time.perf_counter()
            completion_text, metadata, prompt = _predict_completion(
                mode=mode,
                source_text=source_text,
                model_dir=model_dir,
                support_sft=support_sft,
                few_shot_k=few_shot_k,
                max_new_tokens=max_new_tokens,
                hf_session=hf_session,
                support_examples=support_examples,
            )
            generation_seconds = time.perf_counter() - generation_started
            prompt_path = write_inference_request(prompt, prediction_dir / "prompt.txt")
            candidate_cards = completion_text.splitlines()
            candidate_path = write_deck_cards(prediction_dir / "pit_raw_canonical.dck", candidate_cards)
            reference_pit = resolve_record_path(str(reference["reference"]["translate"]["pit_raw_canonical"]), reference_base)
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
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
