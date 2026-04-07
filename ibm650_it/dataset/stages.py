from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    total_count: int
    dev_count: int
    test_count: int
    band_weights: dict[str, int]

    @property
    def train_count(self) -> int:
        return self.total_count - self.dev_count - self.test_count


STAGE_SPECS = {
    "2k": StageSpec(
        name="2k",
        total_count=2000,
        dev_count=200,
        test_count=200,
        band_weights={"B0": 20, "B1": 50, "B2": 20, "B3": 10},
    ),
    "5k": StageSpec(
        name="5k",
        total_count=5000,
        dev_count=500,
        test_count=500,
        band_weights={"B0": 15, "B1": 45, "B2": 25, "B3": 15},
    ),
    "10k": StageSpec(
        name="10k",
        total_count=10000,
        dev_count=1000,
        test_count=1000,
        band_weights={"B0": 10, "B1": 40, "B2": 30, "B3": 20},
    ),
}


def _largest_remainder_counts(total: int, weights: dict[str, int]) -> dict[str, int]:
    weight_total = sum(weights.values())
    counts = {band: (total * weight) // weight_total for band, weight in weights.items()}
    allocated = sum(counts.values())
    remainders = sorted(
        ((total * weight / weight_total - counts[band], band) for band, weight in weights.items()),
        key=lambda item: (-item[0], item[1]),
    )
    remaining = total - allocated
    for _, band in remainders:
        if remaining <= 0:
            break
        counts[band] += 1
        remaining -= 1
    return counts


def get_stage_spec(name: str) -> StageSpec:
    if name not in STAGE_SPECS:
        raise KeyError(f"unknown stage: {name}")
    return STAGE_SPECS[name]


def stage_band_counts(name: str) -> dict[str, int]:
    spec = get_stage_spec(name)
    return _largest_remainder_counts(spec.total_count, spec.band_weights)


def stage_split_counts(name: str) -> dict[str, int]:
    spec = get_stage_spec(name)
    return {
        "synthetic_train": spec.train_count,
        "synthetic_dev": spec.dev_count,
        "synthetic_test": spec.test_count,
    }
