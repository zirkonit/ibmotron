from ibm650_it.dataset.build_records import alpha_normalize_source
from ibm650_it.source.normalize_it import normalize_header, normalize_it_text


def test_normalize_header_packed_and_spaced() -> None:
    assert normalize_header("+        2         0        50        10            1672") == "+ 2 0 50 10 1672"
    assert normalize_header("+00000000020000000000000000005000000000100000001672") == "+ 2 0 50 10 1672"


def test_normalize_it_text_enforces_final_ff() -> None:
    text = "+ 0 1 0 2 1731\n0001 y1 z 1j f\n0002 h ff\n"
    assert normalize_it_text(text) == "+ 0 1 0 2 1731\n0001+ y1 z 1j f\n0002+ h ff\n"


def test_alpha_normalize_renames_statement_numbers_and_variables() -> None:
    source = "+ 0 2 1 2 1730\n0007+ y7 z c9 s y7 f\n0020+ t y7 ff\n"
    assert alpha_normalize_source(source) == "+ 0 2 1 2 1730\n0001+ y1 z c1 s y1 f\n0002+ t y1 ff\n"
