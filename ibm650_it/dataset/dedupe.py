from __future__ import annotations

from collections.abc import Iterable


def dedupe_by_hash(records: Iterable[dict[str, object]], key: str = "alpha_hash") -> list[dict[str, object]]:
    seen: set[str] = set()
    result: list[dict[str, object]] = []
    for record in records:
        value = str(record["hashes"][key])  # type: ignore[index]
        if value in seen:
            continue
        seen.add(value)
        result.append(record)
    return result
