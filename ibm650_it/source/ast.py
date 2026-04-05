from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class Header:
    n_i: int
    n_y: int
    n_c: int
    n_s: int
    n_pkg: int = 265
    N: int = 0


class Expr:
    pass


@dataclass(frozen=True, slots=True)
class IntConst(Expr):
    value: int


@dataclass(frozen=True, slots=True)
class FloatConst(Expr):
    literal: str


@dataclass(frozen=True, slots=True)
class Var(Expr):
    cls: Literal["i", "y", "c"]
    index: Expr


@dataclass(frozen=True, slots=True)
class Neg(Expr):
    expr: Expr


@dataclass(frozen=True, slots=True)
class Add(Expr):
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True, slots=True)
class Sub(Expr):
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True, slots=True)
class Mul(Expr):
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True, slots=True)
class Div(Expr):
    lhs: Expr
    rhs: Expr


class Statement:
    number: int


@dataclass(frozen=True, slots=True)
class Assign(Statement):
    number: int
    target: Var
    expr: Expr


@dataclass(frozen=True, slots=True)
class Goto(Statement):
    number: int
    target_stmt: int


@dataclass(frozen=True, slots=True)
class IfGoto(Statement):
    number: int
    target_stmt: int
    lhs: Expr
    relation: str
    rhs: Expr


@dataclass(frozen=True, slots=True)
class Punch(Statement):
    number: int
    vars: tuple[Var, ...]


@dataclass(frozen=True, slots=True)
class Halt(Statement):
    number: int


@dataclass(frozen=True, slots=True)
class Iterate(Statement):
    number: int
    end_stmt: int
    loop_var: Var
    start_expr: Expr
    step_expr: Expr
    stop_expr: Expr


@dataclass(frozen=True, slots=True)
class Read(Statement):
    number: int
    vars: tuple[Var, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Program:
    statements: tuple[Statement, ...]
    header: Header | None = None
    runtime_package: str = "P1"
