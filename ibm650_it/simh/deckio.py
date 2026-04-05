from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path


def read_deck_cards(path: Path) -> list[str]:
    text = path.read_text(encoding="latin-1")
    return text.splitlines()


def write_deck_cards(path: Path, cards: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(cards)
    if cards:
        payload += "\n"
    path.write_text(payload, encoding="latin-1")
    return path


def copy_raw_deck(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return dst


def split_tail_cards(src: Path, tail_count: int, body_path: Path, tail_path: Path) -> tuple[Path, Path]:
    cards = read_deck_cards(src)
    if len(cards) < tail_count:
        raise ValueError(f"deck has {len(cards)} cards, need at least {tail_count}")
    body_cards = cards[:-tail_count]
    tail_cards = cards[-tail_count:]
    write_deck_cards(body_path, body_cards)
    write_deck_cards(tail_path, tail_cards)
    return body_path, tail_path


def join_decks(parts: list[Path], output_path: Path) -> Path:
    cards: list[str] = []
    for part in parts:
        cards.extend(read_deck_cards(part))
    return write_deck_cards(output_path, cards)


def canonicalize_deck_lines(
    cards: list[str],
    *,
    drop_edge_blank_cards: bool = False,
    collapse_internal_spaces: bool = False,
) -> list[str]:
    normalized = [card.rstrip() for card in cards]
    if collapse_internal_spaces:
        normalized = [re.sub(r" {2,}", " ", card) if card.strip() else "" for card in normalized]
    if drop_edge_blank_cards:
        while normalized and not normalized[0].strip():
            normalized.pop(0)
        while normalized and not normalized[-1].strip():
            normalized.pop()
    return normalized


def canonicalize_deck_file(
    src: Path,
    dst: Path,
    *,
    drop_edge_blank_cards: bool = False,
    collapse_internal_spaces: bool = False,
) -> Path:
    cards = read_deck_cards(src)
    cards = canonicalize_deck_lines(
        cards,
        drop_edge_blank_cards=drop_edge_blank_cards,
        collapse_internal_spaces=collapse_internal_spaces,
    )
    return write_deck_cards(dst, cards)


def deck_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
