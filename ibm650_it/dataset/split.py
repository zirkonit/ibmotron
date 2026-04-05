from __future__ import annotations

import hashlib


def split_by_alpha_hash(record: dict[str, object], train_ratio: float = 0.8, dev_ratio: float = 0.1) -> str:
    alpha_hash = str(record["hashes"]["alpha_hash"])  # type: ignore[index]
    bucket = int(hashlib.sha256(alpha_hash.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "synthetic_train"
    if bucket < train_ratio + dev_ratio:
        return "synthetic_dev"
    return "synthetic_test"


def build_split_map(records: list[dict[str, object]]) -> dict[str, str]:
    return {str(record["id"]): split_by_alpha_hash(record) for record in records}
