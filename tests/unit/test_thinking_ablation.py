from __future__ import annotations

from pathlib import Path

from ibm650_it.training import thinking_ablation


def test_run_thinking_ablation_runs_both_conditions(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, bool | None, bool]] = []

    def fake_run_inference(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_index = output_dir / "predictions.jsonl"
        prediction_index.write_text("", encoding="utf-8")
        calls.append((kwargs["prompt_style"], kwargs["enable_thinking"], kwargs["preserve_raw_completion"]))
        return {
            "prediction_index": str(prediction_index),
        }

    def fake_reevaluate_and_report_mode(**kwargs):
        report_path = Path(kwargs["report_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "exact_match": 0.4 if "thinking_on" in str(report_path) else 0.2,
            "assemblability": 1.0,
            "functional_equivalence": 0.5 if "thinking_on" in str(report_path) else 0.25,
            "per_card_exact": 0.8,
            "normalized_edit_distance": 0.1,
            "failure_taxonomy": {"assembles_but_misexecutes": 1},
        }
        return {
            "prediction_index": str(kwargs["prediction_index"]),
            "report_path": str(report_path),
            "report": report,
            "failure_archive": {"count": 1},
        }

    monkeypatch.setattr(thinking_ablation, "run_inference", fake_run_inference)
    monkeypatch.setattr(thinking_ablation, "reevaluate_and_report_mode", fake_reevaluate_and_report_mode)

    summary = thinking_ablation.run_thinking_ablation(
        reference_index=tmp_path / "dev.jsonl",
        output_root=tmp_path / "out",
        model_dir=tmp_path / "model",
    )

    assert calls == [
        ("chat", True, True),
        ("chat", False, True),
    ]
    assert summary["comparison"]["thinking_on"]["exact_match"] == 0.4
    assert summary["comparison"]["thinking_off"]["exact_match"] == 0.2
    assert summary["comparison"]["delta_thinking_on_minus_off"]["functional_equivalence"] == 0.25
