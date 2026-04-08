from __future__ import annotations

import json
from pathlib import Path

from ibm650_it.dataset.io import load_jsonl
from ibm650_it.eval.band_failure_review import build_band_failure_review


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_build_band_failure_review_classifies_tail_truncation(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    prediction_root = tmp_path / "predictions"
    reference_root.mkdir()
    prediction_root.mkdir()

    source_b2 = reference_root / "source_b2.it"
    source_b3 = reference_root / "source_b3.it"
    source_b2.write_text("+ 1 0 1 3 1730\n0001+ h ff\n", encoding="utf-8")
    source_b3.write_text("+ 1 1 1 3 1730\n0001+ h ff\n", encoding="utf-8")

    ref_b2 = reference_root / "ref_b2.dck"
    ref_b3 = reference_root / "ref_b3.dck"
    ref_b2.write_text(
        "s0001\n"
        "a0001 00 0000 0001\n"
        "4         i     u0001\n"
        "3         i0002  0002\n"
        "4         s     u0003\n"
        "3         s0004  0004\n"
        "4         a     u0005\n"
        "3         a0006  0006\n",
        encoding="latin-1",
    )
    ref_b3.write_text(
        "s0001\n"
        "a0001 00 0000 0001\n"
        "4         i     u0001\n"
        "3         i0002  0002\n"
        "4         y     u0003\n"
        "3         y0004  0004\n"
        "4         c     u0005\n"
        "3         c0006  0006\n",
        encoding="latin-1",
    )

    cand_b2 = prediction_root / "cand_b2.dck"
    cand_b3 = prediction_root / "cand_b3.dck"
    cand_b2.write_text(
        "s0001\n"
        "a0001 00 0000 0001\n"
        "4         i     u0001\n"
        "3         i0002  0002\n"
        "4         s     u0003\n"
        "3         s00\n",
        encoding="latin-1",
    )
    cand_b3.write_text(
        "s0001\n"
        "a0001 00 0000 0001\n"
        "4         i     u0001\n"
        "3         i0002  0002\n"
        "4         y     u0003\n"
        "3         y0004  0004\n"
        "4         c     u0005\n",
        encoding="latin-1",
    )

    reference_index = tmp_path / "reference.jsonl"
    _write_jsonl(
        reference_index,
        [
            {
                "id": "case-b2",
                "band": "B2",
                "source": {"it_text_v1": str(source_b2.relative_to(tmp_path))},
                "reference": {"translate": {"pit_raw_canonical": str(ref_b2.relative_to(tmp_path))}},
                "generator": {"features": ["goto", "if_goto", "multi_output"]},
            },
            {
                "id": "case-b3",
                "band": "B3",
                "source": {"it_text_v1": str(source_b3.relative_to(tmp_path))},
                "reference": {"translate": {"pit_raw_canonical": str(ref_b3.relative_to(tmp_path))}},
                "generator": {"features": ["iterate", "indexed_c", "multi_output"]},
            },
        ],
    )

    prediction_index = prediction_root / "predictions.jsonl"
    _write_jsonl(
        prediction_index,
        [
            {
                "id": "case-b2",
                "band": "B2",
                "pit_raw_canonical": "cand_b2.dck",
                "metrics": {"exact_match": False, "per_card_exact": 0.9, "normalized_edit_distance": 0.1},
                "assemblable": True,
                "functional": False,
                "failure_type": "assembles_but_misexecutes",
            },
            {
                "id": "case-b3",
                "band": "B3",
                "pit_raw_canonical": "cand_b3.dck",
                "metrics": {"exact_match": False, "per_card_exact": 0.92, "normalized_edit_distance": 0.08},
                "assemblable": True,
                "functional": False,
                "failure_type": "assembles_but_misexecutes",
            },
        ],
    )

    summary = build_band_failure_review(
        reference_index=reference_index,
        prediction_index=prediction_index,
        output_root=tmp_path / "review",
        bands=["B2", "B3"],
    )

    cases = load_jsonl(tmp_path / "review" / "cases.jsonl")
    categories = {case["id"]: case["review_category"] for case in cases}

    assert summary["non_exact_count"] == 2
    assert categories["case-b2"] == "partial_dictionary_tail_truncation"
    assert categories["case-b3"] == "partial_dictionary_tail_truncation"
    assert (tmp_path / "review" / "selected_failures" / "0001_case-b2" / "summary.json").exists()
    assert (tmp_path / "review" / "review.md").exists()
