from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True, slots=True)
class ExactMatchMetrics:
    exact_match: bool
    per_card_exact: float
    normalized_edit_distance: float


def compute_exact_match(reference_cards: list[str], candidate_cards: list[str]) -> ExactMatchMetrics:
    max_len = max(len(reference_cards), len(candidate_cards), 1)
    matches = sum(
        1
        for ref, cand in zip(reference_cards, candidate_cards, strict=False)
        if ref == cand
    )
    ratio = SequenceMatcher(a="\n".join(reference_cards), b="\n".join(candidate_cards)).ratio()
    return ExactMatchMetrics(
        exact_match=reference_cards == candidate_cards,
        per_card_exact=matches / max_len,
        normalized_edit_distance=1.0 - ratio,
    )
