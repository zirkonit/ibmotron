from __future__ import annotations

from ibm650_it.dataset.build_records import alpha_normalize_source
from ibm650_it.generate.sample_program import _normalize_it_read_input_deck, generate_band_program, generate_band_sample, infer_features
from ibm650_it.source.render_it_text import render_program
from ibm650_it.source.ast import Goto, IfGoto, Iterate, Punch, Read


def test_generate_b0_has_large_alpha_unique_space() -> None:
    alpha_sources = {
        alpha_normalize_source(render_program(generate_band_program("B0", seed=seed)))
        for seed in range(1, 1201)
    }

    assert len(alpha_sources) >= 1000


def test_generate_b2_contains_control_flow_and_positive_header() -> None:
    program = generate_band_program("B2", seed=7)

    assert any(isinstance(statement, Goto) for statement in program.statements)
    assert any(isinstance(statement, IfGoto) for statement in program.statements)
    assert program.header is not None
    assert program.header.N > 0
    assert "goto" in infer_features(program)
    assert "if_goto" in infer_features(program)


def test_generate_b2_has_large_alpha_unique_space() -> None:
    alpha_sources = {
        alpha_normalize_source(render_program(generate_band_program("B2", seed=seed)))
        for seed in range(1, 201)
    }

    assert len(alpha_sources) >= 180


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

    assert len(alpha_sources) >= 180


def test_generate_b3_every_seed_contains_iterate_and_indexed_features() -> None:
    """Guard against adding a B3 family that accidentally drops the iterate/indexed
    invariants — the band spec says B3 programs must contain both, and downstream
    band-stratified training relies on that."""
    for seed in range(1, 201):
        program = generate_band_program("B3", seed=seed)
        assert any(isinstance(statement, Iterate) for statement in program.statements), f"seed {seed} missing Iterate"
        features = infer_features(program)
        assert "iterate" in features, f"seed {seed} missing iterate feature"
        assert any(f.startswith("indexed_") for f in features), (
            f"seed {seed} missing any indexed_ feature; got {features}"
        )


def test_generate_b3_families_span_wider_loop_bounds() -> None:
    """The 20260409 iterate-diversity additions (wide_indexed_sum, indexed_y_sum,
    dual_post_reduction) should occasionally produce loops with stop > 11, which
    is the historical max from the narrow _loop_bounds(). Over 300 seeds we expect
    to see at least one such case — if we don't, the new families have not been
    wired into the rotation and the diversity knob is dead."""
    max_stop_seen = 0
    for seed in range(1, 301):
        program = generate_band_program("B3", seed=seed)
        for statement in program.statements:
            if isinstance(statement, Iterate):
                stop = statement.stop_expr
                # stop_expr is an IntConst on the generator output
                value = getattr(stop, "value", None)
                if value is not None and int(value) > max_stop_seen:
                    max_stop_seen = int(value)
    assert max_stop_seen > 11, (
        f"expected wider-bounds iterate to produce stop > 11 at least once over 300 seeds; "
        f"max observed was {max_stop_seen}. Either the new B3 families are not in the rotation "
        f"or _loop_bounds_wide() is too narrow."
    )


def test_generate_b4_contains_mixed_control_flow_iteration_and_multiple_outputs() -> None:
    program = generate_band_program("B4", seed=21)

    assert any(isinstance(statement, Iterate) for statement in program.statements)
    assert any(isinstance(statement, Goto) for statement in program.statements)
    assert any(isinstance(statement, IfGoto) for statement in program.statements)
    assert sum(isinstance(statement, Punch) for statement in program.statements) >= 2
    assert program.header is not None
    assert program.header.N > 0
    features = infer_features(program)
    assert "iterate" in features
    assert "goto" in features
    assert "if_goto" in features
    assert "multi_output" in features


def test_generate_b4_has_large_alpha_unique_space() -> None:
    alpha_sources = {
        alpha_normalize_source(render_program(generate_band_program("B4", seed=seed)))
        for seed in range(1, 201)
    }

    assert len(alpha_sources) >= 180


def test_generate_b4_every_seed_contains_mixed_invariants() -> None:
    for seed in range(1, 201):
        program = generate_band_program("B4", seed=seed)
        features = infer_features(program)
        assert any(isinstance(statement, Iterate) for statement in program.statements), f"seed {seed} missing Iterate"
        assert any(isinstance(statement, Goto) for statement in program.statements), f"seed {seed} missing Goto"
        assert any(isinstance(statement, IfGoto) for statement in program.statements), f"seed {seed} missing IfGoto"
        assert sum(isinstance(statement, Punch) for statement in program.statements) >= 2, (
            f"seed {seed} expected at least two Punch statements"
        )
        assert "multi_output" in features, f"seed {seed} missing multi_output feature"


def test_generate_b5_contains_read_and_input_program() -> None:
    generated = generate_band_sample("B5", seed=31)
    program = generated.program

    assert any(isinstance(statement, Read) for statement in program.statements)
    assert generated.input_program is not None
    assert any(isinstance(statement, Punch) for statement in generated.input_program.statements)
    assert not any(isinstance(statement, Read) for statement in generated.input_program.statements)
    assert program.header is not None
    assert program.header.N > 0
    features = infer_features(program)
    assert "read" in features
    assert "iterate" in features
    assert "if_goto" in features
    assert "multi_output" in features


def test_generate_b5_has_large_alpha_unique_space() -> None:
    alpha_sources = {
        alpha_normalize_source(render_program(generate_band_program("B5", seed=seed)))
        for seed in range(1, 201)
    }

    assert len(alpha_sources) >= 180


def test_normalize_it_read_input_deck_emits_textual_type1_cards(tmp_path) -> None:
    raw_output = tmp_path / "raw_output.dck"
    raw_output.write_text(
        "02?001000540?0000050030001000590000000500100010005000000000402000200059000000050\n",
        encoding="latin-1",
    )

    output_path = _normalize_it_read_input_deck(raw_output, tmp_path / "input_data.it.txt")

    assert output_path.read_text(encoding="latin-1") == (
        "02+001000540?0000050\n"
        "03+00100059000000050\n"
        "01+00100050000000004\n"
        "02+002000N9000000050\n"
    )
