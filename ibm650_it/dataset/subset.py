from __future__ import annotations

import json
import shutil
from pathlib import Path

from ibm650_it.dataset.io import load_jsonl, write_jsonl
from ibm650_it.dataset.sampling import stable_limit_records


SPLIT_NAMES = [
    "historical_golden",
    "synthetic_train",
    "synthetic_dev",
    "synthetic_test",
    "adversarial_test",
]


def _sample_root(record: dict[str, object]) -> Path:
    source_ref = Path(str(record["source"]["it_text_v1"]))  # type: ignore[index]
    parts = source_ref.parts
    if not parts:
        raise ValueError(f"record has empty source path: {record.get('id')}")
    if "accepted" in parts:
        index = parts.index("accepted")
        return Path(*parts[index : index + 3])
    if "historical_golden" in parts:
        index = parts.index("historical_golden")
        return Path(*parts[index : index + 2])
    raise ValueError(f"unsupported sample root for record {record.get('id')}: {source_ref}")


def _rewrite_record_paths(record: dict[str, object], *, source_root: Path) -> dict[str, object]:
    source_root_resolved = source_root.resolve()
    source_root_prefix = source_root.as_posix().rstrip("/") + "/"

    def transform(value: object) -> object:
        if isinstance(value, dict):
            return {key: transform(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [transform(inner) for inner in value]
        if isinstance(value, str):
            path = Path(value)
            if path.is_absolute():
                try:
                    return str(path.resolve().relative_to(source_root_resolved))
                except ValueError:
                    return value
            if value.startswith(source_root_prefix):
                return value[len(source_root_prefix) :]
        return value

    return transform(record)  # type: ignore[return-value]


def slice_dataset(
    *,
    source_root: Path,
    output_root: Path,
    train_limit: int | None = None,
    dev_limit: int | None = None,
    test_limit: int | None = None,
    adversarial_limit: int | None = None,
    include_historical_golden: bool = True,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    split_limits = {
        "synthetic_train": train_limit,
        "synthetic_dev": dev_limit,
        "synthetic_test": test_limit,
        "adversarial_test": adversarial_limit,
    }
    selected_by_split: dict[str, list[dict[str, object]]] = {}
    selected_records: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    sample_roots: set[Path] = set()

    for split_name in SPLIT_NAMES:
        split_path = source_root / "splits" / f"{split_name}.jsonl"
        records = load_jsonl(split_path)
        if split_name == "historical_golden":
            if not include_historical_golden:
                records = []
        else:
            limit = split_limits[split_name]
            if limit is not None:
                records = stable_limit_records(
                    records,
                    limit,
                    salt=f"slice_dataset:{source_root}:{split_name}",
                )
        rewritten = [_rewrite_record_paths(record, source_root=source_root) for record in records]
        selected_by_split[split_name] = rewritten
        for record in rewritten:
            sample_roots.add(_sample_root(record))
            record_id = str(record["id"])
            if record_id in selected_ids:
                continue
            selected_ids.add(record_id)
            selected_records.append(record)

    for sample_root in sorted(sample_roots):
        source_path = source_root / sample_root
        destination_path = output_root / sample_root
        if destination_path.exists():
            shutil.rmtree(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, destination_path)

    split_outputs: dict[str, str] = {}
    for split_name, records in selected_by_split.items():
        split_output = output_root / "splits" / f"{split_name}.jsonl"
        write_jsonl(split_output, records)
        split_outputs[split_name] = str(split_output)

    index_path = output_root / "index.jsonl"
    write_jsonl(index_path, selected_records)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "record_count": len(selected_records),
        "split_counts": {name: len(records) for name, records in selected_by_split.items()},
        "sample_root_count": len(sample_roots),
        "index_path": str(index_path),
        "split_outputs": split_outputs,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
