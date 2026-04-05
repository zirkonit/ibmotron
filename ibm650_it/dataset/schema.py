from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class DatasetRecord:
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)["payload"]
