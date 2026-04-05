from __future__ import annotations

import re


class NormalizeITError(ValueError):
    pass


HEADER_RE = re.compile(r"^\+\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
PACKED_HEADER_RE = re.compile(r"^\+(\d{10})(\d{10})(\d{10})(\d{10})(\d{10})$")
STATEMENT_RE = re.compile(r"^(?P<number>\d{1,4})\+?\s+(?P<body>.+?)\s+(?P<term>f|ff)\s*$", re.IGNORECASE)


def normalize_header(line: str) -> str:
    stripped = line.strip()
    packed = PACKED_HEADER_RE.match(stripped)
    if packed:
        numbers = [int(part) for part in packed.groups()]
        return f"+ {numbers[0]} {numbers[1]} {numbers[2]} {numbers[3]} {numbers[4]}"
    match = HEADER_RE.match(stripped)
    if not match:
        raise NormalizeITError(f"unrecognized header line: {line!r}")
    n_i, n_y, n_c, n_s, free = (int(group) for group in match.groups())
    return f"+ {n_i} {n_y} {n_c} {n_s} {free}"


def normalize_it_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise NormalizeITError("empty IT source")
    normalized = [normalize_header(lines[0])]
    seen_final = False
    for line in lines[1:]:
        if seen_final:
            continue
        if line.lstrip().startswith("+"):
            raise NormalizeITError("continuation-card statements are out of scope for it_text_v1")
        match = STATEMENT_RE.match(line.strip())
        if not match:
            raise NormalizeITError(f"unrecognized statement line: {line!r}")
        lower = f"{int(match.group('number')):04d}+ {re.sub(r'\s+', ' ', match.group('body').strip().lower())} {match.group('term').lower()}"
        if lower.endswith(" ff"):
            seen_final = True
        normalized.append(lower)
    if not seen_final:
        raise NormalizeITError("canonical IT source must end in a final ff statement")
    return "\n".join(normalized) + "\n"
