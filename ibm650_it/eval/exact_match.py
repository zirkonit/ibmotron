from __future__ import annotations

from pathlib import Path

from ibm650_it.pit.diff import compute_exact_match
from ibm650_it.simh.deckio import read_deck_cards


def compare_pit_files(reference: Path, candidate: Path) -> dict[str, object]:
    metrics = compute_exact_match(read_deck_cards(reference), read_deck_cards(candidate))
    return {
        "exact_match": metrics.exact_match,
        "per_card_exact": metrics.per_card_exact,
        "normalized_edit_distance": metrics.normalized_edit_distance,
    }
