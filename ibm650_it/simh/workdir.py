from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class JobWorkdir:
    root: Path

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)


def create_job_workdir(base_dir: Path | None = None, prefix: str = "ibm650_it_") -> JobWorkdir:
    if base_dir is None:
        root = Path(tempfile.mkdtemp(prefix=prefix))
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        root = base_dir
        root.mkdir(parents=True, exist_ok=True)
    return JobWorkdir(root=root)


def stage_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return dst
    shutil.copy2(src, dst)
    return dst


def stage_text(text: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return dst
