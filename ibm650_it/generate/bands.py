from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BandSpec:
    name: str
    min_statements: int
    max_statements: int
    allow_control_flow: bool
    allow_iteration: bool


BANDS = {
    "B0": BandSpec("B0", 2, 4, False, False),
    "B1": BandSpec("B1", 4, 8, False, False),
    "B2": BandSpec("B2", 4, 10, True, False),
    "B3": BandSpec("B3", 4, 10, True, True),
    "B4": BandSpec("B4", 8, 20, True, True),
    "B5": BandSpec("B5", 8, 20, True, True),
}
