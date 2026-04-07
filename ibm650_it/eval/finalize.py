from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ibm650_it import REPO_ROOT
from ibm650_it.eval.archive import archive_failures
from ibm650_it.eval.locking import finalize_session
from ibm650_it.eval.reevaluate import reevaluate_prediction_records
from ibm650_it.eval.report import build_evaluation_report, compare_mode_reports


def reevaluate_and_report_mode(
    *,
    reference_index: Path,
    prediction_index: Path,
    prediction_output_dir: Path,
    report_path: Path,
    failure_output_dir: Path,
    failure_archive_limit: int | None = None,
    repo_root: Path = REPO_ROOT,
    step_budget: str = "50M",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    reevaluate_summary = reevaluate_prediction_records(
        reference_index=reference_index,
        prediction_index=prediction_index,
        output_dir=prediction_output_dir,
        repo_root=repo_root,
        step_budget=step_budget,
        timeout_seconds=timeout_seconds,
    )
    resolved_prediction_index = Path(reevaluate_summary["prediction_index"])
    report = build_evaluation_report(
        reference_index=reference_index,
        prediction_index=resolved_prediction_index,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failure_manifest = archive_failures(
        reference_index=reference_index,
        prediction_index=resolved_prediction_index,
        output_dir=failure_output_dir,
        limit=failure_archive_limit,
    )
    return {
        "prediction_index": str(resolved_prediction_index),
        "report_path": str(report_path),
        "report": report,
        "failure_archive": failure_manifest,
    }


def finalize_train_eval_output(
    *,
    dataset_root: Path,
    output_root: Path,
    eval_split: str = "synthetic_dev.jsonl",
    failure_archive_limit: int = 25,
    repo_root: Path = REPO_ROOT,
    step_budget: str = "50M",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    with finalize_session(output_root, scope="train_eval") as session:
        eval_index = dataset_root / "splits" / eval_split
        existing_summary_path = output_root / "summary.json"
        existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8")) if existing_summary_path.exists() else {}

        evaluations: dict[str, Any] = {}
        reports: dict[str, dict[str, Any]] = {}
        for mode in ["zero_shot", "few_shot", "fine_tuned"]:
            session.write_state(status="running", current_mode=mode)
            mode_result = reevaluate_and_report_mode(
                reference_index=eval_index,
                prediction_index=output_root / "predictions" / mode / "predictions.jsonl",
                prediction_output_dir=output_root / "predictions" / mode,
                report_path=output_root / "reports" / f"{mode}.json",
                failure_output_dir=output_root / "failures" / mode,
                failure_archive_limit=failure_archive_limit,
                repo_root=repo_root,
                step_budget=step_budget,
                timeout_seconds=timeout_seconds,
            )
            evaluations[mode] = mode_result
            reports[mode] = mode_result["report"]

        summary = {
            "records_written": existing_summary.get("records_written"),
            "train": existing_summary.get("train"),
            "evaluations": evaluations,
            "baseline_delta": compare_mode_reports(reports),
            "eval_mode": "local_cpu_reevaluate",
        }
        existing_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def finalize_overfit_output(
    *,
    dataset_index: Path,
    output_root: Path,
    failure_archive_limit: int = 25,
    repo_root: Path = REPO_ROOT,
    step_budget: str = "50M",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    with finalize_session(output_root, scope="overfit") as session:
        existing_summary_path = output_root / "summary.json"
        existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8")) if existing_summary_path.exists() else {}

        session.write_state(status="running", current_mode="fine_tuned")
        fine_tuned = reevaluate_and_report_mode(
            reference_index=dataset_index,
            prediction_index=output_root / "predictions" / "fine_tuned" / "predictions.jsonl",
            prediction_output_dir=output_root / "predictions" / "fine_tuned",
            report_path=output_root / "reports" / "fine_tuned.json",
            failure_output_dir=output_root / "failures" / "fine_tuned",
            failure_archive_limit=failure_archive_limit,
            repo_root=repo_root,
            step_budget=step_budget,
            timeout_seconds=timeout_seconds,
        )
        summary = {
            "records_written": existing_summary.get("records_written"),
            "example_count": existing_summary.get("example_count"),
            "train": existing_summary.get("train"),
            "fine_tuned": fine_tuned,
            "eval_mode": "local_cpu_reevaluate",
        }
        existing_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
