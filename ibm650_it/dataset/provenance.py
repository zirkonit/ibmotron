from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from ibm650_it import REPO_ROOT


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_sha(repo_root: Path) -> str | None:
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def load_sources_lock(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    lock_path = repo_root / "sources.lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"missing sources lock: {lock_path}")
    return json.loads(lock_path.read_text(encoding="utf-8"))


def get_simh_lock_entry(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    lock = load_sources_lock(repo_root)
    for entry in lock["sources"]:  # type: ignore[index]
        if entry["source_id"] == "open-simh/simh":  # type: ignore[index]
            return entry  # type: ignore[return-value]
    raise KeyError("open-simh/simh not found in sources.lock.json")


def compute_generator_version(repo_root: Path = REPO_ROOT) -> str:
    git_sha = _git_sha(repo_root)
    if git_sha is not None:
        return git_sha
    paths = [
        repo_root / "ibm650_it/generate/bands.py",
        repo_root / "ibm650_it/generate/sample_program.py",
        repo_root / "ibm650_it/source/ast.py",
        repo_root / "ibm650_it/source/bounds.py",
        repo_root / "ibm650_it/source/render_it_text.py",
        repo_root / "ibm650_it/source/render_it_card80.py",
    ]
    return f"workspace-sha256:{_sha256_files(paths)}"


def compute_normalizer_version(repo_root: Path = REPO_ROOT) -> str:
    git_sha = _git_sha(repo_root)
    if git_sha is not None:
        return git_sha
    paths = [
        repo_root / "ibm650_it/source/normalize_it.py",
        repo_root / "ibm650_it/pit/normalize_pit.py",
        repo_root / "ibm650_it/simh/deckio.py",
        repo_root / "ibm650_it/dataset/build_records.py",
    ]
    return f"workspace-sha256:{_sha256_files(paths)}"


def build_provenance(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    simh = get_simh_lock_entry(repo_root)
    return {
        "simh_source": str(simh["source_id"]),
        "simh_commit_or_checksum": str(simh["commit"]),
        "generator_version": compute_generator_version(repo_root),
        "normalizer_version": compute_normalizer_version(repo_root),
    }
