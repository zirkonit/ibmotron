from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ibm650_it import REPO_ROOT
from ibm650_it.eval.finalize import reevaluate_and_report_mode
from ibm650_it.eval.locking import finalize_session
from ibm650_it.training.infer import run_inference


def _report_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_match": report["exact_match"],
        "assemblability": report["assemblability"],
        "functional_equivalence": report["functional_equivalence"],
        "per_card_exact": report["per_card_exact"],
        "normalized_edit_distance": report["normalized_edit_distance"],
        "failure_taxonomy": report["failure_taxonomy"],
    }


def _metric_delta(thinking_on: dict[str, Any], thinking_off: dict[str, Any]) -> dict[str, float]:
    keys = [
        "exact_match",
        "assemblability",
        "functional_equivalence",
        "per_card_exact",
        "normalized_edit_distance",
    ]
    return {key: thinking_on[key] - thinking_off[key] for key in keys}


def _build_summary(
    *,
    model_dir: Path,
    reference_index: Path,
    eval_mode: str,
    conditions: dict[str, Any],
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model_dir": str(model_dir),
        "reference_index": str(reference_index),
        "eval_mode": eval_mode,
        "conditions": conditions,
    }
    if reports:
        summary["comparison"] = {
            "thinking_on": _report_metrics(reports["thinking_on"]),
            "thinking_off": _report_metrics(reports["thinking_off"]),
            "delta_thinking_on_minus_off": _metric_delta(reports["thinking_on"], reports["thinking_off"]),
        }
    return summary


def run_thinking_ablation(
    *,
    reference_index: Path,
    output_root: Path,
    model_dir: Path,
    repo_root: Path = REPO_ROOT,
    limit: int | None = None,
    max_new_tokens: int = 1024,
    failure_archive_limit: int = 25,
    step_budget: str = "50M",
    timeout_seconds: int = 30,
    eval_mode: str = "inline",
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    conditions: dict[str, Any] = {}
    reports: dict[str, dict[str, Any]] = {}

    for name, enable_thinking in [("thinking_on", True), ("thinking_off", False)]:
        condition_output = output_root / "predictions" / name
        inference_summary = run_inference(
            reference_index=reference_index,
            output_dir=condition_output,
            mode="fine_tuned",
            repo_root=repo_root,
            model_dir=model_dir,
            limit=limit,
            max_new_tokens=max_new_tokens,
            prompt_style="chat",
            enable_thinking=enable_thinking,
            preserve_raw_completion=True,
            step_budget=step_budget,
            timeout_seconds=timeout_seconds,
            eval_mode="skip",
        )
        conditions[name] = {
            "enable_thinking": enable_thinking,
            "prediction_index": inference_summary["prediction_index"],
            "output_dir": str(condition_output),
        }
        if eval_mode == "inline":
            mode_result = reevaluate_and_report_mode(
                reference_index=reference_index,
                prediction_index=Path(inference_summary["prediction_index"]),
                prediction_output_dir=condition_output,
                report_path=output_root / "reports" / f"{name}.json",
                failure_output_dir=output_root / "failures" / name,
                failure_archive_limit=failure_archive_limit,
                repo_root=repo_root,
                step_budget=step_budget,
                timeout_seconds=timeout_seconds,
            )
            conditions[name] = {
                **conditions[name],
                **mode_result,
            }
            reports[name] = mode_result["report"]

    summary = _build_summary(
        model_dir=model_dir,
        reference_index=reference_index,
        eval_mode="local_cpu_reevaluate" if eval_mode == "inline" else "skip",
        conditions=conditions,
        reports=reports,
    )

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def finalize_thinking_ablation_output(
    *,
    reference_index: Path,
    output_root: Path,
    model_dir: Path,
    failure_archive_limit: int = 25,
    repo_root: Path = REPO_ROOT,
    step_budget: str = "50M",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    with finalize_session(output_root, scope="thinking_ablation") as session:
        existing_summary_path = output_root / "summary.json"
        existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8")) if existing_summary_path.exists() else {}
        conditions: dict[str, Any] = existing_summary.get("conditions", {})
        reports: dict[str, dict[str, Any]] = {}

        for name in ["thinking_on", "thinking_off"]:
            session.write_state(status="running", current_mode=name)
            mode_output = output_root / "predictions" / name
            mode_result = reevaluate_and_report_mode(
                reference_index=reference_index,
                prediction_index=mode_output / "predictions.jsonl",
                prediction_output_dir=mode_output,
                report_path=output_root / "reports" / f"{name}.json",
                failure_output_dir=output_root / "failures" / name,
                failure_archive_limit=failure_archive_limit,
                repo_root=repo_root,
                step_budget=step_budget,
                timeout_seconds=timeout_seconds,
            )
            conditions[name] = {
                **conditions.get(name, {}),
                **mode_result,
            }
            reports[name] = mode_result["report"]

        summary = _build_summary(
            model_dir=model_dir,
            reference_index=reference_index,
            eval_mode="local_cpu_reevaluate",
            conditions=conditions,
            reports=reports,
        )
        existing_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
