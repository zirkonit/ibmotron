from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ibm650_it.dataset.build_records import build_record
from ibm650_it.generate.bands import BANDS
from ibm650_it.simh.runner import SimhRunner
from ibm650_it.source.ast import Add, Assign, FloatConst, Goto, Halt, IfGoto, IntConst, Iterate, Mul, Program, Punch, Statement, Var
from ibm650_it.source.render_it_card80 import render_simh_source_deck
from ibm650_it.source.bounds import compute_header
from ibm650_it.source.render_it_text import render_program


def _float_literal(rng: random.Random) -> FloatConst:
    return FloatConst(f"{rng.randint(1, 9)}j")


def _int_literal(rng: random.Random, lo: int = 1, hi: int = 6) -> IntConst:
    return IntConst(rng.randint(lo, hi))


def _loop_bounds(rng: random.Random, *, start_hi: int = 3, step_hi: int = 2, iterations_hi: int = 5) -> tuple[IntConst, IntConst, IntConst]:
    start = rng.randint(1, start_hi)
    step = rng.randint(1, step_hi)
    iterations = rng.randint(2, iterations_hi)
    stop = start + step * (iterations - 1)
    return IntConst(start), IntConst(step), IntConst(stop)


def _loop_bounds_wide(rng: random.Random) -> tuple[IntConst, IntConst, IntConst]:
    """Wider iterate bounds that stress the model on larger symbol-table tails.

    The 20260408 eval's only non-exact case was a B3 iterate where the model got
    a single constant in the dictionary tail off by 1. The narrow _loop_bounds()
    range (start 1-3, step 1-2, iterations 2-5) gives the model limited variety
    on the iterate surface it sees during SFT; widening some families produces
    PIT decks whose accumulator/constant positions vary more, which gives the
    model more signal for generalising the symbol-table arithmetic.
    """
    start = rng.randint(1, 5)
    step = rng.randint(1, 3)
    iterations = rng.randint(3, 8)
    stop = start + step * (iterations - 1)
    return IntConst(start), IntConst(step), IntConst(stop)


def _generate_b0(seed: int) -> Program:
    rng = random.Random(seed)
    assignment_count = rng.randint(1, 3)
    statements = [Assign(1, Var("y", IntConst(1)), _float_literal(rng))]
    if assignment_count >= 2:
        statements.append(Assign(2, Var("c", IntConst(1)), Add(Var("y", IntConst(1)), _float_literal(rng))))
    if assignment_count >= 3:
        statements.append(Assign(3, Var("y", IntConst(2)), Add(Var("c", IntConst(1)), _float_literal(rng))))
    punch_var = Var("y", IntConst(2 if assignment_count >= 3 else 1))
    statements.append(Punch(len(statements) + 1, (punch_var,)))
    statements.append(Halt(len(statements) + 1))
    program = Program(statements=tuple(statements))
    return Program(statements=program.statements, header=compute_header(program))


def _generate_b1(seed: int) -> Program:
    rng = random.Random(seed)
    statements: list[Statement] = [
        Assign(1, Var("i", IntConst(1)), _int_literal(rng, 1, 4)),
        Assign(2, Var("c", IntConst(1)), _float_literal(rng)),
        Assign(3, Var("y", IntConst(1)), Add(Var("c", IntConst(1)), _float_literal(rng))),
        Assign(4, Var("y", IntConst(2)), Mul(Var("y", IntConst(1)), _float_literal(rng))),
        Assign(5, Var("c", IntConst(2)), Add(Var("y", IntConst(2)), Var("c", IntConst(1)))),
    ]
    next_stmt = 6
    output_vars = [Var("y", IntConst(2)), Var("c", IntConst(2))]
    variant = rng.choice(["extra_i", "extra_y", "extra_c"])
    if variant == "extra_i":
        statements.append(Assign(next_stmt, Var("i", IntConst(2)), Add(Var("i", IntConst(1)), _int_literal(rng, 1, 3))))
        output_vars.append(Var("i", IntConst(2)))
    elif variant == "extra_y":
        statements.append(Assign(next_stmt, Var("y", IntConst(3)), Add(Var("c", IntConst(2)), _float_literal(rng))))
        output_vars.append(Var("y", IntConst(3)))
    else:
        statements.append(Assign(next_stmt, Var("c", IntConst(3)), Add(Var("c", IntConst(2)), _float_literal(rng))))
        output_vars.append(Var("c", IntConst(3)))
    next_stmt += 1

    statements.append(Punch(next_stmt, tuple(output_vars[:4])))
    statements.append(Halt(next_stmt + 1))
    program = Program(statements=tuple(statements))
    return Program(statements=program.statements, header=compute_header(program))


def _generate_b2(seed: int) -> Program:
    rng = random.Random(seed)
    limit = rng.randint(4, 7)
    branch_point = rng.randint(2, limit - 1)
    inc_primary = _float_literal(rng)
    inc_secondary = _float_literal(rng)
    family = rng.choice(["branch_accumulate", "dual_accumulator", "post_branch_mix"])
    if family == "branch_accumulate":
        statements = [
            Assign(1, Var("i", IntConst(1)), IntConst(0)),
            Assign(2, Var("c", IntConst(1)), FloatConst("0j")),
            Assign(3, Var("i", IntConst(1)), Add(Var("i", IntConst(1)), IntConst(1))),
            IfGoto(4, 9, Var("i", IntConst(1)), "w", IntConst(limit)),
            IfGoto(5, 7, Var("i", IntConst(1)), "u", IntConst(branch_point)),
            Assign(6, Var("c", IntConst(1)), Add(Var("c", IntConst(1)), inc_primary)),
            Assign(7, Var("c", IntConst(1)), Add(Var("c", IntConst(1)), inc_secondary)),
            Goto(8, 3),
            Punch(9, (Var("i", IntConst(1)), Var("c", IntConst(1)))),
            Halt(10),
        ]
    elif family == "dual_accumulator":
        y_seed = _float_literal(rng)
        statements = [
            Assign(1, Var("i", IntConst(1)), IntConst(0)),
            Assign(2, Var("c", IntConst(1)), FloatConst("0j")),
            Assign(3, Var("y", IntConst(1)), y_seed),
            Assign(4, Var("i", IntConst(1)), Add(Var("i", IntConst(1)), IntConst(1))),
            IfGoto(5, 11, Var("i", IntConst(1)), "w", IntConst(limit)),
            IfGoto(6, 8, Var("i", IntConst(1)), "u", IntConst(branch_point)),
            Assign(7, Var("y", IntConst(1)), Add(Var("y", IntConst(1)), inc_primary)),
            Assign(8, Var("c", IntConst(1)), Add(Var("c", IntConst(1)), inc_secondary)),
            Assign(9, Var("c", IntConst(2)), Add(Var("y", IntConst(1)), Var("c", IntConst(1)))),
            Goto(10, 4),
            Punch(11, (Var("i", IntConst(1)), Var("y", IntConst(1)), Var("c", IntConst(2)))),
            Halt(12),
        ]
    else:
        y_seed = rng.choice([FloatConst("0j"), _float_literal(rng)])
        bias = _float_literal(rng)
        statements = [
            Assign(1, Var("i", IntConst(1)), IntConst(0)),
            Assign(2, Var("c", IntConst(1)), FloatConst("0j")),
            Assign(3, Var("y", IntConst(1)), y_seed),
            Assign(4, Var("i", IntConst(1)), Add(Var("i", IntConst(1)), IntConst(1))),
            IfGoto(5, 12, Var("i", IntConst(1)), "w", IntConst(limit)),
            IfGoto(6, 9, Var("i", IntConst(1)), "u", IntConst(branch_point)),
            Assign(7, Var("c", IntConst(1)), Add(Var("c", IntConst(1)), inc_primary)),
            Goto(8, 10),
            Assign(9, Var("y", IntConst(1)), Add(Var("y", IntConst(1)), inc_secondary)),
            Assign(10, Var("c", IntConst(2)), Add(Var("c", IntConst(1)), Var("y", IntConst(1)))),
            Goto(11, 4),
            Assign(12, Var("y", IntConst(2)), Add(Var("c", IntConst(2)), bias)),
            Punch(13, (Var("i", IntConst(1)), Var("c", IntConst(2)), Var("y", IntConst(2)))),
            Halt(14),
        ]
    program = Program(statements=tuple(statements))
    return Program(statements=program.statements, header=compute_header(program))


def _generate_b3(seed: int) -> Program:
    rng = random.Random(seed)
    family = rng.choice(
        [
            "indexed_sum",
            "progressive_store",
            "postmix_store",
            "indexed_feedback",
            "indexed_pair_store",
            # New families added 20260409 to widen the iterate surface distribution
            # after the b23x/postfix run had a single B3 case miss exact match by
            # one symbol-table constant.
            "wide_indexed_sum",
            "indexed_y_sum",
            "dual_post_reduction",
        ]
    )
    bounds_fn = _loop_bounds_wide if family in {"wide_indexed_sum", "indexed_y_sum", "dual_post_reduction"} else _loop_bounds
    start_expr, step_expr, stop_expr = bounds_fn(rng)

    if family == "indexed_sum":
        base = rng.choice([FloatConst("0j"), _float_literal(rng), _float_literal(rng)])
        increment = _float_literal(rng)
        bias = _float_literal(rng)
        statements = [
            Assign(1, Var("y", IntConst(1)), base),
            Iterate(2, 5, Var("i", IntConst(1)), start_expr, step_expr, stop_expr),
            Assign(3, Var("c", Var("i", IntConst(1))), increment),
            Assign(4, Var("y", IntConst(1)), Add(Var("y", IntConst(1)), Var("c", Var("i", IntConst(1))))),
            Assign(5, Var("c", IntConst(1)), Add(Var("y", IntConst(1)), bias)),
            Punch(6, (Var("y", IntConst(1)), Var("c", IntConst(1)))),
            Halt(7),
        ]
    elif family == "progressive_store":
        current = _float_literal(rng)
        delta = _float_literal(rng)
        base = rng.choice([FloatConst("0j"), _float_literal(rng)])
        statements = [
            Assign(1, Var("c", IntConst(1)), current),
            Assign(2, Var("y", IntConst(1)), base),
            Iterate(3, 6, Var("i", IntConst(1)), start_expr, step_expr, stop_expr),
            Assign(4, Var("c", Var("i", IntConst(1))), Var("c", IntConst(1))),
            Assign(5, Var("y", IntConst(1)), Add(Var("y", IntConst(1)), Var("c", Var("i", IntConst(1))))),
            Assign(6, Var("c", IntConst(1)), Add(Var("c", IntConst(1)), delta)),
            Punch(7, (Var("y", IntConst(1)), Var("c", IntConst(1)))),
            Halt(8),
        ]
    elif family == "postmix_store":
        base = rng.choice([FloatConst("0j"), _float_literal(rng)])
        increment = _float_literal(rng)
        mix = _float_literal(rng)
        statements = [
            Assign(1, Var("y", IntConst(1)), base),
            Assign(2, Var("c", IntConst(1)), mix),
            Iterate(3, 5, Var("i", IntConst(1)), start_expr, step_expr, stop_expr),
            Assign(4, Var("c", Var("i", IntConst(1))), Add(Var("c", IntConst(1)), increment)),
            Assign(5, Var("y", IntConst(1)), Add(Var("y", IntConst(1)), Var("c", Var("i", IntConst(1))))),
            Assign(6, Var("y", IntConst(2)), Add(Var("y", IntConst(1)), Var("c", IntConst(1)))),
            Punch(7, (Var("y", IntConst(1)), Var("y", IntConst(2)))),
            Halt(8),
        ]
    elif family == "indexed_feedback":
        base = rng.choice([FloatConst("0j"), _float_literal(rng)])
        seed_value = _float_literal(rng)
        delta = _float_literal(rng)
        statements = [
            Assign(1, Var("y", IntConst(1)), base),
            Assign(2, Var("c", IntConst(1)), seed_value),
            Iterate(3, 7, Var("i", IntConst(1)), start_expr, step_expr, stop_expr),
            Assign(4, Var("y", Var("i", IntConst(1))), Add(Var("y", IntConst(1)), Var("c", IntConst(1)))),
            Assign(5, Var("c", IntConst(2)), Add(Var("y", Var("i", IntConst(1))), delta)),
            Assign(6, Var("c", IntConst(1)), Var("c", IntConst(2))),
            Assign(7, Var("y", IntConst(1)), Var("y", Var("i", IntConst(1)))),
            Assign(8, Var("c", IntConst(3)), Add(Var("y", IntConst(1)), Var("c", IntConst(1)))),
            Punch(9, (Var("y", IntConst(1)), Var("c", IntConst(1)), Var("c", IntConst(3)))),
            Halt(10),
        ]
    elif family == "indexed_pair_store":
        base = rng.choice([FloatConst("0j"), _float_literal(rng)])
        increment = _float_literal(rng)
        bias = _float_literal(rng)
        statements = [
            Assign(1, Var("c", IntConst(1)), base),
            Assign(2, Var("y", IntConst(1)), FloatConst("0j")),
            Iterate(3, 6, Var("i", IntConst(1)), start_expr, step_expr, stop_expr),
            Assign(4, Var("c", Var("i", IntConst(1))), Add(Var("c", IntConst(1)), increment)),
            Assign(5, Var("y", Var("i", IntConst(1))), Add(Var("y", IntConst(1)), Var("c", Var("i", IntConst(1))))),
            Assign(6, Var("y", IntConst(1)), Var("y", Var("i", IntConst(1)))),
            Assign(7, Var("c", IntConst(2)), Add(Var("y", IntConst(1)), bias)),
            Punch(8, (Var("y", IntConst(1)), Var("c", IntConst(1)), Var("c", IntConst(2)))),
            Halt(9),
        ]
    elif family == "wide_indexed_sum":
        # Same structural shape as the ff563f47 failing case (indexed_sum), but
        # driven with wider loop bounds so each seed produces a different symbol-
        # table tail length. The 20260408 miss was one constant off-by-one on a
        # narrow-bounds (stop=5) instance; seeing wider bounds during SFT gives
        # the model more signal to fit the constant arithmetic.
        base = rng.choice([FloatConst("0j"), _float_literal(rng)])
        increment = _float_literal(rng)
        bias = _float_literal(rng)
        statements = [
            Assign(1, Var("y", IntConst(1)), base),
            Iterate(2, 5, Var("i", IntConst(1)), start_expr, step_expr, stop_expr),
            Assign(3, Var("c", Var("i", IntConst(1))), increment),
            Assign(4, Var("y", IntConst(1)), Add(Var("y", IntConst(1)), Var("c", Var("i", IntConst(1))))),
            Assign(5, Var("c", IntConst(1)), Add(Var("y", IntConst(1)), bias)),
            Punch(6, (Var("y", IntConst(1)), Var("c", IntConst(1)))),
            Halt(7),
        ]
    elif family == "indexed_y_sum":
        # Mirror of indexed_sum using y-indexed instead of c-indexed so the model
        # sees iterate loops that write into the y table as well. Prior B3 families
        # were c-heavy; this rebalances the loop-body variable class.
        base = rng.choice([FloatConst("0j"), _float_literal(rng)])
        increment = _float_literal(rng)
        bias = _float_literal(rng)
        statements = [
            Assign(1, Var("c", IntConst(1)), base),
            Iterate(2, 5, Var("i", IntConst(1)), start_expr, step_expr, stop_expr),
            Assign(3, Var("y", Var("i", IntConst(1))), increment),
            Assign(4, Var("c", IntConst(1)), Add(Var("c", IntConst(1)), Var("y", Var("i", IntConst(1))))),
            Assign(5, Var("y", IntConst(1)), Add(Var("c", IntConst(1)), bias)),
            Punch(6, (Var("c", IntConst(1)), Var("y", IntConst(1)))),
            Halt(7),
        ]
    else:  # dual_post_reduction
        # Two sequential post-loop reductions instead of one. The dictionary tail
        # is visibly longer on these, giving the model training pressure on the
        # trailing cards that were the 20260406 per-card miss before the token-cap
        # fix and the 20260408 off-by-one afterward.
        base = rng.choice([FloatConst("0j"), _float_literal(rng)])
        increment = _float_literal(rng)
        bias_a = _float_literal(rng)
        bias_b = _float_literal(rng)
        statements = [
            Assign(1, Var("y", IntConst(1)), base),
            Iterate(2, 5, Var("i", IntConst(1)), start_expr, step_expr, stop_expr),
            Assign(3, Var("c", Var("i", IntConst(1))), increment),
            Assign(4, Var("y", IntConst(1)), Add(Var("y", IntConst(1)), Var("c", Var("i", IntConst(1))))),
            Assign(5, Var("c", IntConst(1)), Add(Var("y", IntConst(1)), bias_a)),
            Assign(6, Var("c", IntConst(2)), Add(Var("c", IntConst(1)), bias_b)),
            Punch(7, (Var("y", IntConst(1)), Var("c", IntConst(1)), Var("c", IntConst(2)))),
            Halt(8),
        ]
    program = Program(statements=tuple(statements))
    return Program(statements=program.statements, header=compute_header(program))


def generate_band_program(band: str, *, seed: int) -> Program:
    if band not in BANDS:
        raise KeyError(f"unknown band: {band}")
    if band == "B0":
        return _generate_b0(seed)
    if band == "B1":
        return _generate_b1(seed)
    if band == "B2":
        return _generate_b2(seed)
    if band == "B3":
        return _generate_b3(seed)
    raise NotImplementedError(f"{band} generation is not implemented yet")


def _collect_var_features(expr: object, features: set[str]) -> None:
    if isinstance(expr, Var):
        if not isinstance(expr.index, IntConst):
            features.add(f"indexed_{expr.cls}")
        _collect_var_features(expr.index, features)
    elif isinstance(expr, (Add, Mul)):
        _collect_var_features(expr.lhs, features)
        _collect_var_features(expr.rhs, features)


def infer_features(program: Program) -> list[str]:
    features: set[str] = set()
    for statement in program.statements:
        if isinstance(statement, Punch):
            features.add("punch")
            if len(statement.vars) > 1:
                features.add("multi_output")
            for var in statement.vars:
                _collect_var_features(var, features)
        if isinstance(statement, Assign):
            features.add(f"assign_{statement.target.cls}")
            if isinstance(statement.expr, Add):
                features.add("add")
            if isinstance(statement.expr, Mul):
                features.add("mul")
            _collect_var_features(statement.target, features)
            _collect_var_features(statement.expr, features)
        if isinstance(statement, Goto):
            features.add("goto")
        if isinstance(statement, IfGoto):
            features.add("if_goto")
            _collect_var_features(statement.lhs, features)
            _collect_var_features(statement.rhs, features)
        if isinstance(statement, Iterate):
            features.add("iterate")
            _collect_var_features(statement.loop_var, features)
            _collect_var_features(statement.start_expr, features)
            _collect_var_features(statement.step_expr, features)
            _collect_var_features(statement.stop_expr, features)
    return sorted(features)


def generate_accepted_programs(
    *,
    runner: SimhRunner,
    band: str,
    count: int,
    output_dir: Path,
    start_seed: int = 1,
    max_attempts: int | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_attempts = max_attempts or max(count * 20, 20)
    attempts = 0
    accepted = 0
    seed = start_seed
    rejection_counts: Counter[str] = Counter()
    index_path = output_dir / "index.jsonl"
    if index_path.exists():
        index_path.unlink()

    while accepted < count and attempts < max_attempts:
        attempts += 1
        sample_id = f"{band.lower()}_{accepted + 1:04d}_seed{seed:06d}"
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        program = generate_band_program(band, seed=seed)
        source_path = sample_dir / "source.it"
        source_deck_path = sample_dir / "source.simh.txt"
        source_path.write_text(render_program(program), encoding="utf-8")
        source_deck_path.write_text(render_simh_source_deck(program), encoding="utf-8")
        ast_path = sample_dir / "ast.json"
        bounds_path = sample_dir / "bounds.json"
        ast_path.write_text(json.dumps(asdict(program), indent=2), encoding="utf-8")
        bounds_path.write_text(json.dumps(asdict(program.header), indent=2), encoding="utf-8")
        try:
            pipeline = runner.reference_pipeline(
                source_deck=source_deck_path,
                output_dir=sample_dir / "pipeline",
            )
        except Exception as exc:
            rejection_counts[type(exc).__name__] += 1
            seed += 1
            continue

        record = build_record(
            band=band,
            seed=seed,
            source_path=source_path,
            pipeline=pipeline,
            header=asdict(program.header),
            ast_path=ast_path,
            bounds_path=bounds_path,
            features=infer_features(program),
        )
        (sample_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        accepted += 1
        seed += 1

    summary = {
        "band": band,
        "accepted": accepted,
        "attempts": attempts,
        "rejected": attempts - accepted,
        "rejection_counts": dict(rejection_counts),
        "index_path": str(index_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
