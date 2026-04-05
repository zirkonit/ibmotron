from __future__ import annotations

from ibm650_it.source.ast import (
    Add,
    Assign,
    Div,
    Expr,
    FloatConst,
    Goto,
    Halt,
    Header,
    IfGoto,
    IntConst,
    Iterate,
    Mul,
    Neg,
    Program,
    Punch,
    Read,
    Statement,
    Sub,
    Var,
)
from ibm650_it.source.bounds import compute_header


def _render_header(header: Header) -> str:
    return f"+ {header.n_i} {header.n_y} {header.n_c} {header.n_s} {header.N}"


def render_expr(expr: Expr) -> str:
    if isinstance(expr, IntConst):
        return str(expr.value)
    if isinstance(expr, FloatConst):
        return expr.literal.lower()
    if isinstance(expr, Var):
        index = render_expr(expr.index)
        if isinstance(expr.index, IntConst):
            return f"{expr.cls}{index}"
        return f"{expr.cls}{index}"
    if isinstance(expr, Neg):
        return f"m{render_expr(expr.expr)}"
    if isinstance(expr, Add):
        return f"{render_expr(expr.lhs)} s {render_expr(expr.rhs)}"
    if isinstance(expr, Sub):
        return f"{render_expr(expr.lhs)} m {render_expr(expr.rhs)}"
    if isinstance(expr, Mul):
        return f"{render_expr(expr.lhs)} x {render_expr(expr.rhs)}"
    if isinstance(expr, Div):
        return f"{render_expr(expr.lhs)} d {render_expr(expr.rhs)}"
    raise TypeError(f"unsupported expression: {expr!r}")


def render_statement_body(statement: Statement) -> str:
    if isinstance(statement, Assign):
        return f"{render_expr(statement.target)} z {render_expr(statement.expr)}".lower()
    if isinstance(statement, Goto):
        return f"g {statement.target_stmt}".lower()
    if isinstance(statement, IfGoto):
        return f"g {statement.target_stmt} if {render_expr(statement.lhs)} {statement.relation.lower()} {render_expr(statement.rhs)}".lower()
    if isinstance(statement, Punch):
        return " ".join(f"t {render_expr(var)}" for var in statement.vars).lower()
    if isinstance(statement, Halt):
        return "h"
    if isinstance(statement, Iterate):
        return (
            f"{statement.end_stmt}k {render_expr(statement.loop_var)}k "
            f"{render_expr(statement.start_expr)}k {render_expr(statement.step_expr)}k {render_expr(statement.stop_expr)}k"
        ).lower()
    if isinstance(statement, Read):
        return ("read" if not statement.vars else " ".join(["read", *[render_expr(var) for var in statement.vars]])).lower()
    raise TypeError(f"unsupported statement: {statement!r}")


def render_statement(statement: Statement, *, final: bool = False) -> str:
    prefix = f"{statement.number:04d}"
    suffix = "ff" if final else "f"
    return f"{prefix}+ {render_statement_body(statement)} {suffix}"


def render_program(program: Program) -> str:
    header = program.header or compute_header(program)
    lines = [_render_header(header)]
    for index, statement in enumerate(program.statements):
        lines.append(render_statement(statement, final=index == len(program.statements) - 1))
    return "\n".join(lines) + "\n"
