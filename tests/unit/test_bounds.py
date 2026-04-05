from ibm650_it.source.ast import Add, Assign, FloatConst, Halt, IntConst, Program, Punch, Var
from ibm650_it.source.bounds import compute_free_storage, compute_header


def test_prime_example_header_math() -> None:
    assert compute_free_storage(n_i=2, n_y=0, n_c=50, n_s=10, n_pkg=265) == 1672


def test_compute_header_for_simple_program() -> None:
    program = Program(
        statements=(
            Assign(1, Var("y", IntConst(1)), FloatConst("1j")),
            Assign(2, Var("y", IntConst(2)), Add(Var("y", IntConst(1)), FloatConst("2j"))),
            Punch(3, (Var("y", IntConst(2)),)),
            Halt(4),
        )
    )
    header = compute_header(program)
    assert header.n_y == 2
    assert header.n_s == 4
    assert header.N > 0


def test_compute_header_counts_source_side_variable_usage() -> None:
    program = Program(
        statements=(
            Assign(1, Var("y", IntConst(1)), Add(Var("c", IntConst(3)), FloatConst("1j"))),
            Punch(2, (Var("y", IntConst(1)),)),
            Halt(3),
        )
    )
    header = compute_header(program)
    assert header.n_y == 1
    assert header.n_c == 3
