from __future__ import annotations

import hashlib
from collections import deque
from typing import Any


def _record_identity(record: dict[str, Any]) -> str:
    alpha_hash = record.get("hashes", {}).get("alpha_hash")
    if alpha_hash:
        return str(alpha_hash)
    source_ref = record.get("source", {}).get("it_text_v1")
    if source_ref:
        return str(source_ref)
    return str(record.get("id", ""))


def _stable_record_key(record: dict[str, Any], *, salt: str) -> tuple[str, str]:
    identity = _record_identity(record)
    record_id = str(record.get("id", ""))
    digest = hashlib.sha256(f"{salt}:{identity}:{record_id}".encode("utf-8")).hexdigest()
    return digest, record_id


def stable_limit_records(
    records: list[dict[str, Any]],
    limit: int | None,
    *,
    salt: str = "stable_limit_records_v1",
) -> list[dict[str, Any]]:
    """Return a deterministic limited sample without front-of-file curriculum bias.

    When multiple bands are present, records are hash-shuffled *within each band*
    and then drawn round-robin across bands. This keeps limited evals and smoke
    runs balanced across bands while remaining stable across reruns so partial
    outputs can resume safely.
    """
    if limit is None or limit >= len(records):
        return list(records)
    if limit <= 0:
        return []

    bands = {str(record.get("band", "")) for record in records if str(record.get("band", ""))}
    if len(bands) <= 1:
        ordered = sorted(records, key=lambda record: _stable_record_key(record, salt=salt))
        return ordered[:limit]

    queues: dict[str, deque[dict[str, Any]]] = {}
    for band in sorted(bands):
        queues[band] = deque(
            sorted(
                [record for record in records if str(record.get("band", "")) == band],
                key=lambda record: _stable_record_key(record, salt=f"{salt}:{band}"),
            )
        )

    selected: list[dict[str, Any]] = []
    active_bands = list(sorted(queues))
    while active_bands and len(selected) < limit:
        next_active: list[str] = []
        for band in active_bands:
            queue = queues[band]
            if not queue:
                continue
            selected.append(queue.popleft())
            if queue:
                next_active.append(band)
            if len(selected) >= limit:
                break
        active_bands = next_active
    return selected
