from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ibm650_it.dataset.build_records import build_record
from ibm650_it.generate.bands import BANDS
from ibm650_it.simh.runner import SimhRunner
from ibm650_it.source.ast import Add, Assign, FloatConst, Halt, IntConst, Mul, Program, Punch, Var
from ibm650_it.source.render_it_card80 import render_simh_source_deck
from ibm650_it.source.bounds import compute_header
from ibm650_it.source.render_it_text import render_program


def _float_literal(rng: random.Random) -> FloatConst:
    return FloatConst(f"{rng.randint(1, 9)}j")


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
    statements = [
        Assign(1, Var("i", IntConst(1)), IntConst(rng.randint(1, 4))),
        Assign(2, Var("c", IntConst(1)), _float_literal(rng)),
        Assign(3, Var("y", IntConst(1)), Add(Var("c", IntConst(1)), _float_literal(rng))),
        Assign(4, Var("y", IntConst(2)), Mul(Var("y", IntConst(1)), _float_literal(rng))),
    ]
    if rng.choice([True, False]):
        statements.append(Assign(5, Var("c", IntConst(2)), Add(Var("y", IntConst(2)), Var("c", IntConst(1)))))
        output_vars = (Var("y", IntConst(2)), Var("c", IntConst(2)))
    else:
        statements.append(Assign(5, Var("y", IntConst(3)), Add(Var("y", IntConst(2)), _float_literal(rng))))
        output_vars = (Var("y", IntConst(2)), Var("y", IntConst(3)))
    statements.append(Punch(6, output_vars))
    statements.append(Halt(7))
    program = Program(statements=tuple(statements))
    return Program(statements=program.statements, header=compute_header(program))


def generate_band_program(band: str, *, seed: int) -> Program:
    if band not in BANDS:
        raise KeyError(f"unknown band: {band}")
    if band == "B0":
        return _generate_b0(seed)
    if band == "B1":
        return _generate_b1(seed)
    raise NotImplementedError(f"{band} generation is not implemented yet")


def infer_features(program: Program) -> list[str]:
    features: set[str] = set()
    for statement in program.statements:
        if isinstance(statement, Punch):
            features.add("punch")
            if len(statement.vars) > 1:
                features.add("multi_output")
        if isinstance(statement, Assign):
            features.add(f"assign_{statement.target.cls}")
            if isinstance(statement.expr, Add):
                features.add("add")
            if isinstance(statement.expr, Mul):
                features.add("mul")
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
