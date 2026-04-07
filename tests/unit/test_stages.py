from __future__ import annotations

from pathlib import Path

from ibm650_it.dataset import corpus, stages
from ibm650_it.dataset.split import build_exact_splits


def test_stage_counts_match_declared_totals() -> None:
    for stage_name, spec in stages.STAGE_SPECS.items():
        band_counts = stages.stage_band_counts(stage_name)
        split_counts = stages.stage_split_counts(stage_name)

        assert sum(band_counts.values()) == spec.total_count
        assert split_counts["synthetic_train"] + split_counts["synthetic_dev"] + split_counts["synthetic_test"] == spec.total_count
        assert split_counts["synthetic_dev"] == spec.dev_count
        assert split_counts["synthetic_test"] == spec.test_count


def test_build_stage_corpus_forwards_stage_counts(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_build_pilot_corpus(**kwargs):
        captured.update(kwargs)
        return {"accepted_total": 2000}

    monkeypatch.setattr(corpus, "build_pilot_corpus", fake_build_pilot_corpus)

    summary = corpus.build_stage_corpus(
        stage="2k",
        output_root=tmp_path / "stage_2k",
        workers=2,
        include_historical_golden=False,
    )

    assert summary["stage"] == "2k"
    assert captured["band_counts"] == stages.stage_band_counts("2k")
    assert captured["split_counts"] == stages.stage_split_counts("2k")
    assert captured["include_historical_golden"] is False


def test_build_exact_splits_respects_requested_counts() -> None:
    records = []
    for band in ["B0", "B1", "B2", "B3"]:
        for index in range(3):
            records.append(
                {
                    "id": f"{band}-{index}",
                    "band": band,
                    "hashes": {"alpha_hash": f"{band}-alpha-{index}"},
                }
            )

    buckets = build_exact_splits(
        records,
        split_counts={
            "synthetic_train": 8,
            "synthetic_dev": 2,
            "synthetic_test": 2,
        },
    )

    assert len(buckets["synthetic_train"]) == 8
    assert len(buckets["synthetic_dev"]) == 2
    assert len(buckets["synthetic_test"]) == 2
