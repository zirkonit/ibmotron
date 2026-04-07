from __future__ import annotations

import hashlib
import math


SYNTHETIC_SPLITS = ["synthetic_train", "synthetic_dev", "synthetic_test"]


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


def _stable_band_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        records,
        key=lambda record: (
            hashlib.sha256(str(record["hashes"]["alpha_hash"]).encode("utf-8")).hexdigest(),
            str(record["id"]),
        ),
    )


def _allocate_band_quotas(records_by_band: dict[str, list[dict[str, object]]], total_count: int) -> dict[str, int]:
    total_records = sum(len(records) for records in records_by_band.values())
    if total_count < 0 or total_count > total_records:
        raise ValueError(f"cannot allocate {total_count} records across {total_records} available records")
    if total_records == 0:
        return {band: 0 for band in records_by_band}

    quotas = {band: 0 for band in records_by_band}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for band, records in sorted(records_by_band.items()):
        exact = total_count * len(records) / total_records
        base = math.floor(exact)
        quotas[band] = min(base, len(records))
        allocated += quotas[band]
        remainders.append((exact - base, band))

    remaining = total_count - allocated
    for _, band in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if quotas[band] >= len(records_by_band[band]):
            continue
        quotas[band] += 1
        remaining -= 1
    return quotas


def build_exact_splits(
    records: list[dict[str, object]],
    *,
    split_counts: dict[str, int],
) -> dict[str, list[dict[str, object]]]:
    synthetic_records = [record for record in records if record["band"] != "historical_golden"]
    historical_golden = [record for record in records if record["band"] == "historical_golden"]
    requested = sum(split_counts.get(name, 0) for name in SYNTHETIC_SPLITS)
    if requested != len(synthetic_records):
        raise ValueError(
            f"split counts {split_counts} do not match synthetic record count {len(synthetic_records)}"
        )

    remaining_by_band: dict[str, list[dict[str, object]]] = {}
    for record in synthetic_records:
        remaining_by_band.setdefault(str(record["band"]), []).append(record)
    for band in remaining_by_band:
        remaining_by_band[band] = _stable_band_records(remaining_by_band[band])

    buckets: dict[str, list[dict[str, object]]] = {
        "historical_golden": historical_golden,
        "synthetic_train": [],
        "synthetic_dev": [],
        "synthetic_test": [],
        "adversarial_test": [],
    }

    for split_name in ["synthetic_dev", "synthetic_test"]:
        quotas = _allocate_band_quotas(remaining_by_band, split_counts.get(split_name, 0))
        for band in sorted(remaining_by_band):
            take = quotas.get(band, 0)
            buckets[split_name].extend(remaining_by_band[band][:take])
            remaining_by_band[band] = remaining_by_band[band][take:]

    for band in sorted(remaining_by_band):
        buckets["synthetic_train"].extend(remaining_by_band[band])

    for split_name in SYNTHETIC_SPLITS:
        expected = split_counts.get(split_name, 0)
        actual = len(buckets[split_name])
        if actual != expected:
            raise ValueError(f"{split_name} count mismatch: expected {expected}, got {actual}")
    return buckets
