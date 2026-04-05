from __future__ import annotations

from pathlib import Path

from ibm650_it.simh.deckio import read_deck_cards


def compare_run_outputs(reference: Path, candidate: Path) -> bool:
    return read_deck_cards(reference) == read_deck_cards(candidate)
