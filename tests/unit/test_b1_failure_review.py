from __future__ import annotations

import json
from pathlib import Path

from ibm650_it.dataset.io import load_jsonl
from ibm650_it.eval.b1_failure_review import build_b1_failure_review


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_build_b1_failure_review_classifies_and_archives_cases(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    prediction_root = tmp_path / "predictions"
    reference_root.mkdir()
    prediction_root.mkdir()

    source_one = reference_root / "source_one.it"
    source_two = reference_root / "source_two.it"
    source_three = reference_root / "source_three.it"
    for path in [source_one, source_two, source_three]:
        path.write_text("+ 1 0 1 3 1730\n0001+ h ff\n", encoding="utf-8")

    ref_one = reference_root / "ref_one.dck"
    ref_two = reference_root / "ref_two.dck"
    ref_three = reference_root / "ref_three.dck"
    ref_one.write_text("s0001\ns0002\na0001 10 0000 0050\na0002 20 0000 0050\na0003 30 0000 0050\n", encoding="latin-1")
    ref_two.write_text("s0001\ns0002\na0001 30 0000 0050\n", encoding="latin-1")
    ref_three.write_text("s0001\ns0002\n", encoding="latin-1")

    cand_one = prediction_root / "cand_one.dck"
    cand_two = prediction_root / "cand_two.dck"
    cand_three = prediction_root / "cand_three.dck"
    cand_one.write_text("s0001\ns0002\na0001 20 0000 0050\na0002 10 0000 0050\na0003 31 0000 0050\n", encoding="latin-1")
    cand_two.write_text("s0001\ns0002\na0001 31 0000 0050\n", encoding="latin-1")
    cand_three.write_text("0001+ y1 z 1j f\n0002+ h ff\n", encoding="latin-1")

    reference_index = tmp_path / "reference.jsonl"
    _write_jsonl(
        reference_index,
        [
                {
                    "id": "case-1",
                    "band": "B1",
                    "source": {"it_text_v1": str(source_one.relative_to(tmp_path))},
                    "reference": {"translate": {"pit_raw_canonical": str(ref_one.relative_to(tmp_path))}},
                },
                {
                    "id": "case-2",
                    "band": "B1",
                    "source": {"it_text_v1": str(source_two.relative_to(tmp_path))},
                    "reference": {"translate": {"pit_raw_canonical": str(ref_two.relative_to(tmp_path))}},
                },
                {
                    "id": "case-3",
                    "band": "B1",
                    "source": {"it_text_v1": str(source_three.relative_to(tmp_path))},
                    "reference": {"translate": {"pit_raw_canonical": str(ref_three.relative_to(tmp_path))}},
                },
            ],
        )

    prediction_index = prediction_root / "predictions.jsonl"
    _write_jsonl(
        prediction_index,
        [
            {
                "id": "case-1",
                "band": "B1",
                "pit_raw_canonical": "cand_one.dck",
                "metrics": {"exact_match": False, "per_card_exact": 0.8, "normalized_edit_distance": 0.2},
                "assemblable": True,
                "functional": False,
                "failure_type": "assembles_but_misexecutes",
            },
            {
                "id": "case-2",
                "band": "B1",
                "pit_raw_canonical": "cand_two.dck",
                "metrics": {"exact_match": False, "per_card_exact": 0.9, "normalized_edit_distance": 0.1},
                "assemblable": True,
                "functional": True,
                "failure_type": "functional_success_exact_failure",
            },
            {
                "id": "case-3",
                "band": "B1",
                "pit_raw_canonical": "cand_three.dck",
                "metrics": {"exact_match": False, "per_card_exact": 0.0, "normalized_edit_distance": 1.0},
                "assemblable": False,
                "functional": False,
                "failure_type": "malformed_pit_card",
            },
        ],
    )

    summary = build_b1_failure_review(
        reference_index=reference_index,
        prediction_index=prediction_index,
        output_root=tmp_path / "review",
    )

    cases = load_jsonl(tmp_path / "review" / "cases.jsonl")
    categories = {case["id"]: case["review_category"] for case in cases}

    assert summary["b1_non_exact_count"] == 3
    assert categories["case-1"] == "symbolic_tail_or_reservation_drift"
    assert categories["case-2"] == "functional_success_exact_failure"
    assert categories["case-3"] == "malformed_pit"
    assert (tmp_path / "review" / "selected_failures" / "0001_case-3" / "summary.json").exists()
