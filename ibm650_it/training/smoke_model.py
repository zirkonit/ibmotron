from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ibm650_it.dataset.build_records import alpha_normalize_source


@dataclass(slots=True)
class SmokeExample:
    id: str
    source_text: str
    source_alpha: str
    completion: str
    band: str | None = None


@dataclass(slots=True)
class SmokePrediction:
    completion: str
    matched_example_id: str | None
    matched_score: float | None
    support_example_ids: list[str]


def _extract_source_text(prompt: str) -> str:
    match = re.search(r"<IT>\n(.*?)\n</IT>", prompt, flags=re.DOTALL)
    if not match:
        raise ValueError("prompt does not contain an <IT>...</IT> block")
    return match.group(1) + "\n"


def load_sft_examples(sft_path: Path) -> list[SmokeExample]:
    records = [json.loads(line) for line in sft_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples: list[SmokeExample] = []
    for record in records:
        source_text = record.get("source_text") or _extract_source_text(record["prompt"])
        source_alpha = record.get("source_alpha") or alpha_normalize_source(source_text)
        examples.append(
            SmokeExample(
                id=str(record["id"]),
                source_text=str(source_text),
                source_alpha=str(source_alpha),
                completion=str(record["completion"]),
                band=str(record["band"]) if record.get("band") is not None else None,
            )
        )
    return examples


def train_smoke_model(
    *,
    sft_path: Path,
    output_dir: Path,
    resume_from: Path | None = None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_sft_examples(sft_path)
    if max_examples is not None:
        examples = examples[:max_examples]

    merged: dict[str, SmokeExample] = {}
    if resume_from is not None:
        for example in load_smoke_model(resume_from):
            merged[example.id] = example
    for example in examples:
        merged[example.id] = example

    payload = {
        "backend": "smoke",
        "format": "smoke_v1",
        "sft_path": str(sft_path),
        "resume_from": str(resume_from) if resume_from is not None else None,
        "example_count": len(merged),
        "examples": [asdict(example) for example in sorted(merged.values(), key=lambda item: item.id)],
    }
    model_path = output_dir / "model.json"
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {
        "backend": "smoke",
        "example_count": len(merged),
        "model_path": str(model_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_smoke_model(model_dir: Path) -> list[SmokeExample]:
    payload = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    return [SmokeExample(**example) for example in payload["examples"]]


def _similarity(source_alpha: str, example: SmokeExample) -> float:
    return SequenceMatcher(a=source_alpha, b=example.source_alpha).ratio()


def predict_zero_shot(*, source_text: str) -> SmokePrediction:
    return SmokePrediction(
        completion="",
        matched_example_id=None,
        matched_score=None,
        support_example_ids=[],
    )


def predict_few_shot(
    *,
    source_text: str,
    support_examples: list[SmokeExample],
    few_shot_k: int,
) -> SmokePrediction:
    if not support_examples or few_shot_k <= 0:
        return predict_zero_shot(source_text=source_text)
    source_alpha = alpha_normalize_source(source_text)
    chosen = support_examples[:few_shot_k]
    best = max(chosen, key=lambda example: (_similarity(source_alpha, example), example.id))
    return SmokePrediction(
        completion=best.completion,
        matched_example_id=best.id,
        matched_score=_similarity(source_alpha, best),
        support_example_ids=[example.id for example in chosen],
    )


def predict_fine_tuned(
    *,
    source_text: str,
    model_examples: list[SmokeExample],
) -> SmokePrediction:
    if not model_examples:
        return predict_zero_shot(source_text=source_text)
    source_alpha = alpha_normalize_source(source_text)
    exact = next((example for example in model_examples if example.source_text == source_text), None)
    if exact is not None:
        return SmokePrediction(
            completion=exact.completion,
            matched_example_id=exact.id,
            matched_score=1.0,
            support_example_ids=[exact.id],
        )
    best = max(model_examples, key=lambda example: (_similarity(source_alpha, example), example.id))
    return SmokePrediction(
        completion=best.completion,
        matched_example_id=best.id,
        matched_score=_similarity(source_alpha, best),
        support_example_ids=[best.id],
    )
