from ibm650_it.source.ast import Assign, FloatConst, Halt, IntConst, Program, Punch, Var
from ibm650_it.source.bounds import compute_header
from ibm650_it.source.render_it_card80 import render_simh_source_deck


def test_render_simh_source_deck_places_terminator_in_column_70() -> None:
    program = Program(
        statements=(
            Assign(1, Var("y", IntConst(1)), FloatConst("2j")),
            Punch(2, (Var("y", IntConst(1)),)),
            Halt(3),
        )
    )
    program = Program(statements=program.statements, header=compute_header(program))
    lines = render_simh_source_deck(program).splitlines()
    assert lines[2][69] == "f"
    assert lines[3][69] == "f"
    assert lines[4][68:70] == "ff"
