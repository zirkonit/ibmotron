from pathlib import Path

from ibm650_it.simh.runner import SimhRunner


def test_parse_accup_handles_trailing_sign() -> None:
    assert SimhRunner._parse_accup("ACCUP:\t 0000000000+\n") == 0
    assert SimhRunner._parse_accup("ACCUP:\t 6600000000-\n") == -6600000000


def test_runner_normalizes_repo_root_to_absolute_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "third_party" / "simh" / "BIN").mkdir(parents=True)
    (tmp_path / "ibm650_it" / "simh" / "ini_templates").mkdir(parents=True)
    binary = tmp_path / "third_party" / "simh" / "BIN" / "i650"
    binary.write_text("", encoding="utf-8")

    runner = SimhRunner(repo_root=Path("."))

    assert runner.repo_root == tmp_path.resolve()
    assert runner.simh_binary == binary.resolve()
    assert runner.template_root == (tmp_path / "ibm650_it" / "simh" / "ini_templates").resolve()
