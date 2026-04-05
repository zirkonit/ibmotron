from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ibm650_it import REPO_ROOT


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def relativize_record_paths(record: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    def transform(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: transform(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [transform(inner) for inner in value]
        if isinstance(value, str) and value.startswith("/"):
            return os.path.relpath(value, dataset_root)
        return value

    return transform(record)


def resolve_record_path(path_ref: str, base_dir: Path) -> Path:
    path = Path(path_ref)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    repo_relative = (REPO_ROOT / path).resolve()
    if repo_relative.exists():
        return repo_relative
    return candidate


def resolve_record_base(index_path: Path) -> Path:
    if index_path.parent.name == "splits":
        return index_path.parent.parent
    return index_path.parent
