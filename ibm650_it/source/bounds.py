from __future__ import annotations

from dataclasses import dataclass

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
    Sub,
    Var,
)

P1_FOOTPRINT = 265


class BoundAnalysisError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Interval:
    lo: int
    hi: int

    def add(self, other: "Interval") -> "Interval":
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def sub(self, other: "Interval") -> "Interval":
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def mul_const(self, const: int) -> "Interval":
        values = (self.lo * const, self.hi * const)
        return Interval(min(values), max(values))

    def div_const(self, const: int) -> "Interval":
        if const == 0:
            raise BoundAnalysisError("division by zero in fixed-point bounds analysis")
        values = (self.lo // const, self.hi // const)
        return Interval(min(values), max(values))


def _i_var_name(var: Var) -> str:
    if var.cls != "i" or not isinstance(var.index, IntConst):
        raise BoundAnalysisError("only simple I variables can define fixed-point bounds in v1")
    return f"i{var.index.value}"


def eval_fixed_interval(expr: Expr, env: dict[str, Interval]) -> Interval:
    if isinstance(expr, IntConst):
        return Interval(expr.value, expr.value)
    if isinstance(expr, FloatConst):
        raise BoundAnalysisError("floating point values are not admissible in fixed-point bounds")
    if isinstance(expr, Var):
        if expr.cls != "i":
            raise BoundAnalysisError("only I variables are admissible in fixed-point bounds")
        key = _i_var_name(expr)
        if key not in env:
            raise BoundAnalysisError(f"unknown fixed-point variable bound: {key}")
        return env[key]
    if isinstance(expr, Neg):
        child = eval_fixed_interval(expr.expr, env)
        return Interval(-child.hi, -child.lo)
    if isinstance(expr, Add):
        return eval_fixed_interval(expr.lhs, env).add(eval_fixed_interval(expr.rhs, env))
    if isinstance(expr, Sub):
        return eval_fixed_interval(expr.lhs, env).sub(eval_fixed_interval(expr.rhs, env))
    if isinstance(expr, Mul):
        if isinstance(expr.lhs, IntConst):
            return eval_fixed_interval(expr.rhs, env).mul_const(expr.lhs.value)
        if isinstance(expr.rhs, IntConst):
            return eval_fixed_interval(expr.lhs, env).mul_const(expr.rhs.value)
        raise BoundAnalysisError("v1 bounds only support multiplication by a constant")
    if isinstance(expr, Div):
        if isinstance(expr.rhs, IntConst):
            return eval_fixed_interval(expr.lhs, env).div_const(expr.rhs.value)
        raise BoundAnalysisError("v1 bounds only support division by a constant")
    raise BoundAnalysisError(f"unsupported expression: {expr!r}")


def _record_var_max(var: Var, env: dict[str, Interval], maxima: dict[str, int]) -> None:
    interval = eval_fixed_interval(var.index, env)
    if interval.hi < 0:
        raise BoundAnalysisError(f"negative subscript upper bound for {var.cls}: {interval.hi}")
    maxima[var.cls] = max(maxima[var.cls], interval.hi)


def compute_header(program: Program, n_pkg: int = P1_FOOTPRINT) -> Header:
    env: dict[str, Interval] = {}
    maxima = {"i": 0, "y": 0, "c": 0}
    for statement in program.statements:
        if isinstance(statement, Assign):
            _record_var_max(statement.target, env, maxima)
            _collect_expr_var_max(statement.expr, env, maxima)
            if statement.target.cls == "i" and isinstance(statement.target.index, IntConst):
                try:
                    env[_i_var_name(statement.target)] = eval_fixed_interval(statement.expr, env)
                except BoundAnalysisError:
                    pass
        elif isinstance(statement, IfGoto):
            for expr in (statement.lhs, statement.rhs):
                for cls in ("i", "y", "c"):
                    _collect_expr_var_max(expr, env, maxima)
        elif isinstance(statement, Punch):
            for var in statement.vars:
                _record_var_max(var, env, maxima)
        elif isinstance(statement, Iterate):
            if statement.loop_var.cls != "i":
                raise BoundAnalysisError("iteration loop variable must be fixed-point I in v1")
            _record_var_max(statement.loop_var, env, maxima)
            start = eval_fixed_interval(statement.start_expr, env)
            step = eval_fixed_interval(statement.step_expr, env)
            stop = eval_fixed_interval(statement.stop_expr, env)
            if step.lo != step.hi or step.lo == 0:
                raise BoundAnalysisError("iteration step must be a non-zero constant in v1")
            loop_interval = Interval(min(start.lo, stop.hi), max(start.hi, stop.lo))
            env[_i_var_name(statement.loop_var)] = loop_interval
        elif isinstance(statement, Read):
            for var in statement.vars:
                _record_var_max(var, env, maxima)
        elif isinstance(statement, (Goto, Halt)):
            pass
        else:
            raise BoundAnalysisError(f"unsupported statement for bounds: {statement!r}")
    n_s = max(statement.number for statement in program.statements)
    N = compute_free_storage(
        n_i=maxima["i"],
        n_y=maxima["y"],
        n_c=maxima["c"],
        n_s=n_s,
        n_pkg=n_pkg,
    )
    if N <= 0:
        raise BoundAnalysisError(f"computed non-positive free storage N={N}")
    return Header(
        n_i=maxima["i"],
        n_y=maxima["y"],
        n_c=maxima["c"],
        n_s=n_s,
        n_pkg=n_pkg,
        N=N,
    )


def compute_free_storage(*, n_i: int, n_y: int, n_c: int, n_s: int, n_pkg: int = P1_FOOTPRINT) -> int:
    return 1999 - (n_i + n_y + n_c + n_s + n_pkg)


def _collect_expr_var_max(expr: Expr, env: dict[str, Interval], maxima: dict[str, int]) -> None:
    if isinstance(expr, Var):
        _record_var_max(expr, env, maxima)
    elif isinstance(expr, (Neg,)):
        _collect_expr_var_max(expr.expr, env, maxima)
    elif isinstance(expr, (Add, Sub, Mul, Div)):
        _collect_expr_var_max(expr.lhs, env, maxima)
        _collect_expr_var_max(expr.rhs, env, maxima)
    elif isinstance(expr, (IntConst, FloatConst)):
        return
    else:
        raise BoundAnalysisError(f"unsupported expression: {expr!r}")
