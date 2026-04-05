from __future__ import annotations

from ibm650_it.source.ast import Program
from ibm650_it.source.render_it_text import render_program, render_statement_body


class Card80RenderError(ValueError):
    pass


def render_card80(program: Program) -> str:
    lines = render_program(program).splitlines()
    output: list[str] = []
    for index, line in enumerate(lines):
        if index == 0:
            output.append(line[:80].ljust(80))
            output.append("".ljust(80))
            continue
        parts = line.split()
        stmt_no = parts[0]
        body = "".join(parts[1:-1])
        terminator = parts[-1].upper()
        if len(body) > 14:
            raise Card80RenderError(f"statement body exceeds 14 characters: {body}")
        card = f"{stmt_no}+".ljust(41) + body.upper().ljust(15) + terminator.rjust(2)
        output.append(card[:80].ljust(80))
    return "\n".join(output) + "\n"


def render_simh_source_deck(program: Program) -> str:
    header = program.header
    if header is None:
        text = render_program(program)
        header_line = text.splitlines()[0]
    else:
        header_line = f"+{header.n_i:>9}{header.n_y:>10}{header.n_c:>10}{header.n_s:>10}{header.N:>16}"
    output = [header_line, ""]
    for index, statement in enumerate(program.statements):
        prefix = f"{statement.number:04d}+" if statement.number != 0 else "    +"
        term = "ff" if index == len(program.statements) - 1 else "f"
        body = render_statement_body(statement)
        output.append(prefix.ljust(42) + body.ljust(26) + term.rjust(2))
    return "\n".join(output) + "\n"
