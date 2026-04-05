from ibm650_it.simh.runner import SimhRunner


def test_parse_accup_handles_trailing_sign() -> None:
    assert SimhRunner._parse_accup("ACCUP:\t 0000000000+\n") == 0
    assert SimhRunner._parse_accup("ACCUP:\t 6600000000-\n") == -6600000000
