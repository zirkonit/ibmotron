from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedSoapCard:
    raw: str
    symbolic_location: int | None


LOCATION_RE = re.compile(r"\b(\d{4})\s+[\d\-]{2}\s+\d{4}\s+\d{4}\s*$")


def parse_symbolic_location(card: str) -> ParsedSoapCard:
    match = LOCATION_RE.search(card)
    return ParsedSoapCard(raw=card, symbolic_location=int(match.group(1)) if match else None)
