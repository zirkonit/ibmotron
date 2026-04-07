from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator


FINALIZE_LOCK_FILENAME = ".finalize.lock"
FINALIZE_STATE_FILENAME = ".finalize_state.json"


class FinalizeLockError(RuntimeError):
    pass


@dataclass(slots=True)
class FinalizeSession:
    output_root: Path
    scope: str
    pid: int
    lock_path: Path
    state_path: Path

    def write_state(
        self,
        *,
        status: str,
        current_mode: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        payload = {
            "scope": self.scope,
            "pid": self.pid,
            "status": status,
            "current_mode": current_mode,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if error_type is not None:
            payload["error_type"] = error_type
        if error_message is not None:
            payload["error_message"] = error_message
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _lock_payload(*, scope: str, pid: int) -> dict[str, object]:
    return {
        "scope": scope,
        "pid": pid,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _read_lock(lock_path: Path) -> dict[str, object] | None:
    if not lock_path.exists():
        return None
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        return None


@contextmanager
def finalize_session(output_root: Path, *, scope: str) -> Iterator[FinalizeSession]:
    output_root.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    lock_path = output_root / FINALIZE_LOCK_FILENAME
    state_path = output_root / FINALIZE_STATE_FILENAME
    lock_payload = _lock_payload(scope=scope, pid=pid)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        existing = _read_lock(lock_path)
        owner = f"pid={existing.get('pid')}" if existing else "unknown owner"
        raise FinalizeLockError(
            f"finalize already in progress for {output_root} ({owner})"
        ) from exc

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(lock_payload, handle, indent=2)

    session = FinalizeSession(
        output_root=output_root,
        scope=scope,
        pid=pid,
        lock_path=lock_path,
        state_path=state_path,
    )
    session.write_state(status="running")
    try:
        yield session
    except Exception as exc:
        session.write_state(
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc).strip() or repr(exc),
        )
        raise
    else:
        if state_path.exists():
            state_path.unlink()
    finally:
        if lock_path.exists():
            lock_path.unlink()
