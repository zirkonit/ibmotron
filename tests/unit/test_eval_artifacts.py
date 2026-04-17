from __future__ import annotations

import json
from pathlib import Path

from ibm650_it.dataset.io import load_jsonl
from ibm650_it.eval.archive import archive_failures
from ibm650_it.eval.reevaluate import _reference_has_successful_run, reevaluate_prediction_records
from ibm650_it.eval.report import build_evaluation_report


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_build_evaluation_report_uses_stored_metrics_when_candidate_missing(tmp_path: Path) -> None:
    source_path = tmp_path / "source.it"
    source_path.write_text("+ 1 0 1 3 1730\n0001+ y1 z 1 f\n0002+ t y1 f\n0003+ h ff\n", encoding="utf-8")
    reference_pit = tmp_path / "reference_pit.dck"
    reference_pit.write_text("card1\ncard2\n", encoding="latin-1")

    reference_index = tmp_path / "reference.jsonl"
    _write_jsonl(
        reference_index,
        [
            {
                "id": "sample-1",
                "band": "B0",
                "source": {"it_text_v1": "source.it"},
                "reference": {"translate": {"pit_raw_canonical": "reference_pit.dck"}},
                "generator": {"features": ["punch"]},
            }
        ],
    )

    prediction_index = tmp_path / "predictions.jsonl"
    _write_jsonl(
        prediction_index,
        [
            {
                "id": "sample-1",
                "mode": "fine_tuned",
                "pit_raw_canonical": "missing_candidate.dck",
                "metrics": {
                    "exact_match": False,
                    "per_card_exact": 0.25,
                    "normalized_edit_distance": 0.75,
                },
                "assemblable": False,
                "functional": False,
                "failure_type": "returned_it_source_instead_of_pit",
            }
        ],
    )

    report = build_evaluation_report(
        reference_index=reference_index,
        prediction_index=prediction_index,
    )

    assert report["count"] == 1
    assert report["exact_match"] == 0.0
    assert report["per_card_exact"] == 0.25
    assert report["normalized_edit_distance"] == 0.75
    assert report["failure_taxonomy"] == {"returned_it_source_instead_of_pit": 1}


def test_archive_failures_writes_summary_when_candidate_missing(tmp_path: Path) -> None:
    source_path = tmp_path / "source.it"
    source_path.write_text("+ 1 0 1 3 1730\n0001+ y1 z 1 f\n0002+ t y1 f\n0003+ h ff\n", encoding="utf-8")
    reference_pit = tmp_path / "reference_pit.dck"
    reference_pit.write_text("card1\ncard2\n", encoding="latin-1")

    reference_index = tmp_path / "reference.jsonl"
    _write_jsonl(
        reference_index,
        [
            {
                "id": "sample-1",
                "band": "B0",
                "source": {"it_text_v1": "source.it"},
                "reference": {"translate": {"pit_raw_canonical": "reference_pit.dck"}},
                "generator": {"features": ["punch"]},
            }
        ],
    )

    prediction_index = tmp_path / "predictions.jsonl"
    _write_jsonl(
        prediction_index,
        [
            {
                "id": "sample-1",
                "mode": "fine_tuned",
                "pit_raw_canonical": "missing_candidate.dck",
                "metrics": {
                    "exact_match": False,
                    "per_card_exact": 0.0,
                    "normalized_edit_distance": 1.0,
                },
                "assemblable": False,
                "functional": False,
                "failure_type": "returned_it_source_instead_of_pit",
                "retrieval": {"backend": "transformers_qlora"},
            }
        ],
    )

    manifest = archive_failures(
        reference_index=reference_index,
        prediction_index=prediction_index,
        output_dir=tmp_path / "failures",
    )

    case_dir = tmp_path / "failures" / "0001_sample-1"
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    assert manifest == {"count": 1, "by_failure_type": {"returned_it_source_instead_of_pit": 1}}
    assert summary["candidate_missing"] is True
    assert not (case_dir / "candidate_pit.dck").exists()


def test_reevaluate_predictions_persists_full_exception_metadata_for_missing_candidate(tmp_path: Path) -> None:
    source_path = tmp_path / "source.it"
    source_path.write_text("+ 1 0 1 3 1730\n0001+ y1 z 1 f\n0002+ t y1 f\n0003+ h ff\n", encoding="utf-8")
    reference_pit = tmp_path / "reference_pit.dck"
    reference_pit.write_text("card1\ncard2\n", encoding="latin-1")

    reference_index = tmp_path / "reference.jsonl"
    _write_jsonl(
        reference_index,
        [
            {
                "id": "sample-1",
                "band": "B0",
                "source": {"it_text_v1": "source.it"},
                "reference": {
                    "translate": {"pit_raw_canonical": "reference_pit.dck"},
                    "run": {"status": "ok", "output_deck": "reference_output.dck"},
                },
                "generator": {"features": ["punch"]},
            }
        ],
    )
    (tmp_path / "reference_output.dck").write_text("out\n", encoding="latin-1")

    prediction_index = tmp_path / "predictions.jsonl"
    _write_jsonl(
        prediction_index,
        [
            {
                "id": "sample-1",
                "mode": "fine_tuned",
                "pit_raw_canonical": "missing_candidate.dck",
                "metrics": {
                    "exact_match": False,
                    "per_card_exact": 0.0,
                    "normalized_edit_distance": 1.0,
                },
                "assemblable": False,
                "functional": False,
                "failure_type": "returned_it_source_instead_of_pit",
                "timings": {"generation_seconds": 0.5},
            }
        ],
    )

    summary = reevaluate_prediction_records(
        reference_index=reference_index,
        prediction_index=prediction_index,
        output_dir=tmp_path / "reevaluated",
    )
    reevaluated = load_jsonl(Path(summary["prediction_index"]))[0]

    assert reevaluated["assemble"]["status"] == "candidate_missing"
    assert reevaluated["assemble"]["error_type"] == "FileNotFoundError"
    assert "missing_candidate.dck" in reevaluated["assemble"]["error_message"]
    assert reevaluated["timings"]["generation_seconds"] == 0.5
    assert reevaluated["timings"]["evaluation_seconds"] >= 0.0


def test_reference_has_successful_run_requires_nonempty_output_deck(tmp_path: Path) -> None:
    empty_output = tmp_path / "empty_output.dck"
    empty_output.write_text("", encoding="latin-1")
    nonempty_output = tmp_path / "nonempty_output.dck"
    nonempty_output.write_text("card\n", encoding="latin-1")

    assert not _reference_has_successful_run(
        {"reference": {"run": {"status": "ok", "output_deck": "empty_output.dck"}}},
        tmp_path,
    )
    assert not _reference_has_successful_run(
        {"reference": {"run": {"status": "run_error", "output_deck": "nonempty_output.dck"}}},
        tmp_path,
    )
    assert _reference_has_successful_run(
        {"reference": {"run": {"status": "ok", "output_deck": "nonempty_output.dck"}}},
        tmp_path,
    )
