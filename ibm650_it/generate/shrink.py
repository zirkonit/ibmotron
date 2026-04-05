from __future__ import annotations


def shrink_source_lines(lines: list[str]) -> list[list[str]]:
    if len(lines) <= 2:
        return []
    candidates: list[list[str]] = []
    for index in range(1, len(lines) - 1):
        candidates.append(lines[:index] + lines[index + 1 :])
    return candidates
