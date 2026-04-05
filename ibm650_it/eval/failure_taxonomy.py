from __future__ import annotations

import re


IT_HEADER_RE = re.compile(r"^\+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s*$")
IT_STATEMENT_RE = re.compile(r"^\d{4}\+\s+.+\s+f{1,2}\s*$", flags=re.IGNORECASE)


def _looks_like_it_source(candidate_cards: list[str]) -> bool:
    if not candidate_cards:
        return False
    it_like = 0
    for card in candidate_cards:
        stripped = card.strip()
        if IT_HEADER_RE.match(stripped) or IT_STATEMENT_RE.match(stripped):
            it_like += 1
    return it_like >= max(2, len(candidate_cards) // 2)


def classify_failure(
    *,
    candidate_cards: list[str],
    exact_match: bool,
    assemblable: bool,
    functional: bool,
    assemble_status: str | None = None,
) -> str:
    if exact_match:
        return "exact_match"
    joined = "\n".join(candidate_cards)
    if "<IT>" in joined or "User:" in joined or "Assistant:" in joined:
        return "malformed_source_echo_in_output"
    if _looks_like_it_source(candidate_cards):
        return "returned_it_source_instead_of_pit"
    if len(candidate_cards) < 10:
        return "malformed_pit_card"
    if not assemblable:
        if assemble_status == "assemble_error" and len(candidate_cards) >= 10:
            return "wrong_reservation_handling_or_symbolic_output"
        return "unassemblable_output"
    if functional:
        return "functional_success_exact_failure"
    return "assembles_but_misexecutes"
