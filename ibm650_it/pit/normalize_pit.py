from __future__ import annotations

from pathlib import Path

from ibm650_it.simh.deckio import canonicalize_deck_file


def canonicalize_pit_file(src: Path, dst: Path) -> Path:
    return canonicalize_deck_file(
        src,
        dst,
        drop_edge_blank_cards=True,
        collapse_internal_spaces=False,
    )
