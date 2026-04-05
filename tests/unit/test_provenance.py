from ibm650_it.dataset.provenance import build_provenance


def test_provenance_is_complete() -> None:
    provenance = build_provenance()
    assert provenance["simh_source"] == "open-simh/simh"
    assert provenance["simh_commit_or_checksum"]
    assert provenance["generator_version"]
    assert provenance["normalizer_version"]
