from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _metric_line(report: dict[str, Any]) -> str:
    return (
        f"exact={report.get('exact_match', 0.0):.3f}, "
        f"assemblability={report.get('assemblability', 0.0):.3f}, "
        f"functional={report.get('functional_equivalence', 0.0):.3f}, "
        f"per_card={report.get('per_card_exact', 0.0):.3f}, "
        f"edit_distance={report.get('normalized_edit_distance', 0.0):.3f}"
    )


def write_research_report(*, summary_path: Path, output_path: Path) -> Path:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [
        "# IBM 650 IT -> PIT Training Report",
        "",
        "## Training",
        "",
        f"- records_written: {summary.get('records_written')}",
        f"- train_backend: {summary.get('train', {}).get('backend')}",
        f"- train_examples: {summary.get('train', {}).get('example_count')}",
        f"- model_path: {summary.get('train', {}).get('model_path')}",
        "",
        "## Mode Metrics",
        "",
    ]

    evaluations = summary.get("evaluations", {})
    for mode in ["zero_shot", "few_shot", "fine_tuned"]:
        payload = evaluations.get(mode, {})
        report = payload.get("report", {})
        lines.extend(
            [
                f"### {mode}",
                "",
                f"- metrics: {_metric_line(report)}",
                f"- report_path: {payload.get('report_path')}",
                f"- prediction_index: {payload.get('prediction_index')}",
                f"- failure_archive: {payload.get('failure_archive', {}).get('count', 0)} cases",
                f"- failure_taxonomy: {json.dumps(report.get('failure_taxonomy', {}), sort_keys=True)}",
                "",
            ]
        )

    lines.extend(["## Baseline Delta", ""])
    baseline_delta = summary.get("baseline_delta", {})
    for mode in ["zero_shot", "few_shot", "fine_tuned"]:
        payload = baseline_delta.get(mode, {})
        lines.append(f"### {mode}")
        lines.append("")
        for metric, values in payload.items():
            lines.append(
                f"- {metric}: value={values.get('value')}, delta_vs_zero_shot={values.get('delta_vs_zero_shot')}"
            )
        lines.append("")

    lines.extend(
        [
            "## Reproducibility",
            "",
            "- The canonical entry point is `python -m ibm650_it.cli train-eval ...`.",
            "- Remote GPU runs can be launched with `python3 scripts/runpod_train_eval.py ...`.",
            f"- Summary JSON: {summary_path}",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
