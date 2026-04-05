from __future__ import annotations


def classify_failure(
    *,
    exact_match: bool,
    assemblable: bool,
    functional: bool,
) -> str:
    if exact_match:
        return "exact_match"
    if not assemblable:
        return "unassemblable_output"
    if functional:
        return "functional_success_exact_failure"
    return "assembles_but_misexecutes"
