from pathlib import Path

from ibm650_it.simh.deckio import canonicalize_deck_file, join_decks, read_deck_cards, split_tail_cards, write_deck_cards


def test_split_tail_cards_and_join(tmp_path: Path) -> None:
    source = tmp_path / "source.dck"
    write_deck_cards(source, [f"card-{index}" for index in range(1, 16)])
    body, tail = split_tail_cards(source, 10, tmp_path / "body.dck", tmp_path / "tail.dck")
    assert read_deck_cards(body) == [f"card-{index}" for index in range(1, 6)]
    assert read_deck_cards(tail) == [f"card-{index}" for index in range(6, 16)]

    joined = join_decks([tail, body], tmp_path / "joined.dck")
    assert read_deck_cards(joined) == [f"card-{index}" for index in range(6, 16)] + [f"card-{index}" for index in range(1, 6)]


def test_canonicalize_drops_edge_blank_cards_only(tmp_path: Path) -> None:
    source = tmp_path / "pit_raw.dck"
    write_deck_cards(source, ["", "  ", "card 1   ", "card 2", ""])
    canonical = canonicalize_deck_file(source, tmp_path / "pit_canonical.dck", drop_edge_blank_cards=True)
    assert read_deck_cards(canonical) == ["card 1", "card 2"]
