#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party"
SOURCE_CACHE = THIRD_PARTY / "source_cache"
LOCKFILE = REPO_ROOT / "sources.lock.json"

SIMH_REPO_URL = "https://github.com/open-simh/simh.git"
CARNEGIE_PDF_URL = "https://archive.org/download/bitsavers_ibm650Carntor_16304233/CarnegieInternalTranslator.pdf"
CARNEGIE_TXT_URL = "https://archive.org/stream/bitsavers_ibm650Carntor_16304233/CarnegieInternalTranslator_djvu.txt"
IBM650_MANUAL_URL = "https://archive.computerhistory.org/resources/access/text/2012/07/102726995-05-01-acc.pdf"
HF_API_URL = "https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
HF_README_URL = "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/raw/main/README.md"
UNSLOTH_NEMOTRON_URL = "https://unsloth.ai/docs/models/nemotron-3.md"
UNSLOTH_REQUIREMENTS_URL = "https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        path.write_bytes(response.read())


def ensure_file(url: str, path: Path, *, refresh: bool) -> None:
    if refresh or not path.exists():
        download(url, path)


def ensure_simh(refresh: bool) -> tuple[Path, str]:
    simh_root = THIRD_PARTY / "simh"
    if not simh_root.exists():
        subprocess.run(["git", "clone", SIMH_REPO_URL, str(simh_root)], check=True)
    elif refresh:
        subprocess.run(["git", "-C", str(simh_root), "fetch", "--all", "--tags"], check=True)
    commit = (
        subprocess.run(
            ["git", "-C", str(simh_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    return simh_root, commit


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def build_lock(refresh: bool) -> dict[str, object]:
    fetched_at = utcnow()
    simh_root, simh_commit = ensure_simh(refresh=refresh)

    history_dir = SOURCE_CACHE / "history"
    training_dir = SOURCE_CACHE / "training"
    ensure_file(CARNEGIE_PDF_URL, history_dir / "CarnegieInternalTranslator.pdf", refresh=refresh)
    ensure_file(CARNEGIE_TXT_URL, history_dir / "CarnegieInternalTranslator_djvu.txt", refresh=refresh)
    ensure_file(IBM650_MANUAL_URL, history_dir / "IBM650_Manual_102726995-05-01-acc.pdf", refresh=refresh)
    ensure_file(HF_API_URL, training_dir / "nvidia_NVIDIA-Nemotron-3-Nano-4B-BF16.api.json", refresh=refresh)
    ensure_file(HF_README_URL, training_dir / "nvidia_NVIDIA-Nemotron-3-Nano-4B-BF16_README.md", refresh=refresh)
    ensure_file(UNSLOTH_NEMOTRON_URL, training_dir / "unsloth_nemotron_3.md", refresh=refresh)
    ensure_file(UNSLOTH_REQUIREMENTS_URL, training_dir / "unsloth_requirements.md", refresh=refresh)

    simh_required = [
        "I650/sw/run_it.ini",
        "I650/sw/run_soap.ini",
        "I650/sw/it/it_compiler.dck",
        "I650/sw/it/soapII.dck",
        "I650/sw/it/soapII_patch.dck",
        "I650/sw/it/it_reservation_p1.dck",
        "I650/sw/it/it_package_p1.dck",
        "I650/sw/it/it_example_1_src.txt",
        "I650/sw/it/it_example_1_data.txt",
        "I650/sw/it/it_example_2_src.txt",
    ]
    simh_files = []
    for rel_path in simh_required:
        file_path = simh_root / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"missing required SIMH file: {file_path}")
        simh_files.append(
            {
                "path": rel(file_path),
                "sha256": sha256_file(file_path),
            }
        )

    hf_api = json.loads((training_dir / "nvidia_NVIDIA-Nemotron-3-Nano-4B-BF16.api.json").read_text(encoding="utf-8"))
    return {
        "generated_at": fetched_at,
        "sources": [
            {
                "source_id": "open-simh/simh",
                "family": "simh",
                "commit": simh_commit,
                "fetch_date": fetched_at,
                "local_path": rel(simh_root),
                "files": simh_files,
            },
            {
                "source_id": "CarnegieInternalTranslator.pdf",
                "family": "historical_doc",
                "url": CARNEGIE_PDF_URL,
                "checksum_sha256": sha256_file(history_dir / "CarnegieInternalTranslator.pdf"),
                "fetch_date": fetched_at,
                "local_path": rel(history_dir / "CarnegieInternalTranslator.pdf"),
                "notes_path": rel(history_dir / "CarnegieInternalTranslator_djvu.txt"),
            },
            {
                "source_id": "IBM650_Manual_102726995-05-01-acc.pdf",
                "family": "historical_doc",
                "url": IBM650_MANUAL_URL,
                "checksum_sha256": sha256_file(history_dir / "IBM650_Manual_102726995-05-01-acc.pdf"),
                "fetch_date": fetched_at,
                "local_path": rel(history_dir / "IBM650_Manual_102726995-05-01-acc.pdf"),
            },
            {
                "source_id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                "family": "training_stack",
                "url": HF_README_URL,
                "commit_or_checksum": hf_api["sha"],
                "last_modified": hf_api.get("lastModified"),
                "fetch_date": fetched_at,
                "local_path": rel(training_dir / "nvidia_NVIDIA-Nemotron-3-Nano-4B-BF16_README.md"),
                "api_path": rel(training_dir / "nvidia_NVIDIA-Nemotron-3-Nano-4B-BF16.api.json"),
                "checksum_sha256": sha256_file(training_dir / "nvidia_NVIDIA-Nemotron-3-Nano-4B-BF16_README.md"),
            },
            {
                "source_id": "unsloth_nemotron_3",
                "family": "training_stack",
                "url": UNSLOTH_NEMOTRON_URL,
                "checksum_sha256": sha256_file(training_dir / "unsloth_nemotron_3.md"),
                "fetch_date": fetched_at,
                "local_path": rel(training_dir / "unsloth_nemotron_3.md"),
            },
            {
                "source_id": "unsloth_requirements",
                "family": "training_stack",
                "url": UNSLOTH_REQUIREMENTS_URL,
                "checksum_sha256": sha256_file(training_dir / "unsloth_requirements.md"),
                "fetch_date": fetched_at,
                "local_path": rel(training_dir / "unsloth_requirements.md"),
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-fetch remote artifacts before locking them")
    args = parser.parse_args()
    lock = build_lock(refresh=args.refresh)
    LOCKFILE.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(LOCKFILE)


if __name__ == "__main__":
    main()
