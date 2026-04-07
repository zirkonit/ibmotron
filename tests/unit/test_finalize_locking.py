from __future__ import annotations

import json
from pathlib import Path

import pytest

from ibm650_it.eval.locking import FINALIZE_LOCK_FILENAME, FINALIZE_STATE_FILENAME, FinalizeLockError, finalize_session


def test_finalize_session_rejects_concurrent_writer(tmp_path: Path) -> None:
    output_root = tmp_path / "eval"

    with finalize_session(output_root, scope="train_eval"):
        with pytest.raises(FinalizeLockError):
            with finalize_session(output_root, scope="train_eval"):
                raise AssertionError("unreachable")


def test_finalize_session_writes_failure_state_and_clears_lock(tmp_path: Path) -> None:
    output_root = tmp_path / "eval"

    with pytest.raises(RuntimeError, match="boom"):
        with finalize_session(output_root, scope="train_eval") as session:
            session.write_state(status="running", current_mode="fine_tuned")
            raise RuntimeError("boom")

    assert not (output_root / FINALIZE_LOCK_FILENAME).exists()
    state = json.loads((output_root / FINALIZE_STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["error_type"] == "RuntimeError"
    assert state["error_message"] == "boom"
