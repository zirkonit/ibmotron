from __future__ import annotations

from ibm650_it.source.ast import (
    Add,
    Assign,
    FloatConst,
    Halt,
    IntConst,
    Program,
    Punch,
    Var,
)


def straight_line_template() -> Program:
    return Program(
        statements=(
            Assign(1, Var("y", IntConst(1)), FloatConst("1j")),
            Assign(2, Var("y", IntConst(2)), Add(Var("y", IntConst(1)), FloatConst("2j"))),
            Punch(3, (Var("y", IntConst(2)),)),
            Halt(4),
        )
    )
