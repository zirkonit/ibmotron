import json
from pathlib import Path

from ibm650_it.dataset.subset import slice_dataset


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_slice_dataset_copies_only_selected_sample_roots(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    accepted_a = source_root / "accepted" / "B0" / "0001_000001"
    accepted_b = source_root / "accepted" / "B1" / "0001_000101"
    golden = source_root / "historical_golden" / "it_example_1"
    for path in [accepted_a, accepted_b, golden]:
        path.mkdir(parents=True, exist_ok=True)
        (path / "source.it").write_text("placeholder", encoding="utf-8")

    train_records = [
        {"id": "a", "source": {"it_text_v1": "accepted/B0/0001_000001/source.it"}},
        {"id": "b", "source": {"it_text_v1": "accepted/B1/0001_000101/source.it"}},
    ]
    dev_records = [
        {"id": "b", "source": {"it_text_v1": str(source_root / "accepted" / "B1" / "0001_000101" / "source.it")}},
    ]
    golden_records = [
        {"id": "g", "source": {"it_text_v1": "historical_golden/it_example_1/source.it"}},
    ]

    _write_jsonl(source_root / "splits" / "synthetic_train.jsonl", train_records)
    _write_jsonl(source_root / "splits" / "synthetic_dev.jsonl", dev_records)
    _write_jsonl(source_root / "splits" / "synthetic_test.jsonl", [])
    _write_jsonl(source_root / "splits" / "adversarial_test.jsonl", [])
    _write_jsonl(source_root / "splits" / "historical_golden.jsonl", golden_records)

    summary = slice_dataset(
        source_root=source_root,
        output_root=tmp_path / "subset",
        train_limit=1,
        dev_limit=1,
    )

    assert summary["record_count"] == 3
    assert (tmp_path / "subset" / "accepted" / "B0" / "0001_000001" / "source.it").exists()
    assert (tmp_path / "subset" / "accepted" / "B1" / "0001_000101" / "source.it").exists()
    assert (tmp_path / "subset" / "historical_golden" / "it_example_1" / "source.it").exists()
