from __future__ import annotations

from ibm650_it.dataset.build_records import alpha_normalize_source
from ibm650_it.generate.sample_program import generate_band_program, infer_features
from ibm650_it.source.render_it_text import render_program
from ibm650_it.source.ast import Goto, IfGoto, Iterate


def test_generate_b2_contains_control_flow_and_positive_header() -> None:
    program = generate_band_program("B2", seed=7)

    assert any(isinstance(statement, Goto) for statement in program.statements)
    assert any(isinstance(statement, IfGoto) for statement in program.statements)
    assert program.header is not None
    assert program.header.N > 0
    assert "goto" in infer_features(program)
    assert "if_goto" in infer_features(program)


def test_generate_b3_contains_iteration_and_indexed_features() -> None:
    program = generate_band_program("B3", seed=11)

    assert any(isinstance(statement, Iterate) for statement in program.statements)
    assert program.header is not None
    assert program.header.N > 0
    features = infer_features(program)
    assert "iterate" in features
    assert "indexed_c" in features


def test_generate_b3_has_large_alpha_unique_space() -> None:
    alpha_sources = {
        alpha_normalize_source(render_program(generate_band_program("B3", seed=seed)))
        for seed in range(1, 201)
    }

    assert len(alpha_sources) >= 150
