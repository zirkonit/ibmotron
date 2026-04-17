from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import runpod_progressive_qwen


class _FakeCtl:
    def __init__(self, returncodes: list[int]) -> None:
        self.returncodes = list(returncodes)
        self.wait_calls: list[tuple[str, int, int]] = []
        self.ssh_commands: list[str] = []

    def wait_for_ssh(self, pod_id: str, *, timeout_seconds: int, poll_seconds: int) -> dict[str, str]:
        self.wait_calls.append((pod_id, timeout_seconds, poll_seconds))
        return {"pod_id": pod_id}

    def ssh(self, info: dict[str, str], remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        self.ssh_commands.append(remote_command)
        returncode = self.returncodes.pop(0) if self.returncodes else 1
        return subprocess.CompletedProcess(["ssh"], returncode, "", "")


def test_wait_for_ready_marker_retries_until_marker_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    ctl = _FakeCtl([1, 1, 0])
    monkeypatch.setattr(runpod_progressive_qwen.time, "sleep", lambda _: None)

    ssh_info = runpod_progressive_qwen._wait_for_ready_marker(
        ctl=ctl,
        pod_id="pod-123",
        timeout_seconds=30,
        poll_seconds=0,
    )

    assert ssh_info == {"pod_id": "pod-123"}
    assert ctl.ssh_commands == [
        f"test -f {runpod_progressive_qwen.READY_MARKER}",
        f"test -f {runpod_progressive_qwen.READY_MARKER}",
        f"test -f {runpod_progressive_qwen.READY_MARKER}",
    ]
    assert ctl.wait_calls[0] == ("pod-123", 30, 1)


def test_wait_for_ready_marker_times_out_when_marker_never_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    ctl = _FakeCtl([1, 1, 1, 1])
    ticks = iter([0.0, 0.4, 0.8, 1.1, 1.1])
    monkeypatch.setattr(runpod_progressive_qwen.time, "time", lambda: next(ticks))
    monkeypatch.setattr(runpod_progressive_qwen.time, "sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="did not publish ready marker"):
        runpod_progressive_qwen._wait_for_ready_marker(
            ctl=ctl,
            pod_id="pod-456",
            timeout_seconds=1,
            poll_seconds=0,
        )


def _record(record_id: str, band: str) -> dict[str, object]:
    return {
        "id": record_id,
        "band": band,
        "hashes": {"alpha_hash": f"alpha-{record_id}"},
        "source": {"it_text_v1": f"source/{record_id}.txt"},
        "reference": {"translate": {"pit_raw_canonical": f"target/{record_id}.dck"}},
    }


def test_prepare_stage_dataset_supports_cumulative_focus_bands(tmp_path: Path) -> None:
    train_records = [
        _record(f"{band}-{idx}", band)
        for band in ["B0", "B1", "B2", "B3", "B4", "B5"]
        for idx in range(5)
    ]
    eval_records = [
        _record(f"eval-{band}-{idx}", band)
        for band in ["B0", "B1", "B2", "B3", "B4", "B5"]
        for idx in range(2)
    ]
    stage = runpod_progressive_qwen.ProgressiveStage(
        train_count=12,
        eval_count=12,
        focus_bands=("B0", "B1"),
    )

    summary = runpod_progressive_qwen._prepare_stage_dataset(
        source_root=tmp_path,
        output_root=tmp_path / "dataset",
        stage=stage,
        train_records=train_records,
        eval_records=eval_records,
    )

    assert summary["focus_bands"] == ["B0", "B1"]
    assert summary["focus_band_weights"] == {"B0": 4, "B1": 4}
    assert summary["train_band_counts"] == {"B0": 4, "B1": 4, "B2": 1, "B3": 1, "B4": 1, "B5": 1}
    assert summary["eval_band_counts"] == {"B0": 2, "B1": 2, "B2": 2, "B3": 2, "B4": 2, "B5": 2}


def test_prediction_timing_metrics_reads_fine_tuned_timings(tmp_path: Path) -> None:
    prediction_root = tmp_path / "predictions"
    prediction_root.mkdir()
    prediction_index = prediction_root / "predictions.jsonl"
    prediction_index.write_text(
        "\n".join(
            [
                json.dumps({"timings": {"generation_seconds": 2.0, "evaluation_seconds": 0.5, "total_seconds": 2.5}}),
                json.dumps({"timings": {"generation_seconds": 4.0, "evaluation_seconds": 1.5, "total_seconds": 5.5}}),
                json.dumps({"timings": {"generation_seconds": 6.0, "evaluation_seconds": 2.5, "total_seconds": 8.5}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "evaluations": {
            "fine_tuned": {
                "prediction_index": str(prediction_index),
            }
        }
    }

    metrics = runpod_progressive_qwen._prediction_timing_metrics(summary)

    assert metrics["prediction_count"] == 3
    assert metrics["generation_total_seconds"] == pytest.approx(12.0)
    assert metrics["generation_avg_seconds"] == pytest.approx(4.0)
    assert metrics["generation_median_seconds"] == pytest.approx(4.0)
    assert metrics["evaluation_total_seconds"] == pytest.approx(4.5)
    assert metrics["evaluation_avg_seconds"] == pytest.approx(1.5)
    assert metrics["per_example_total_seconds"] == pytest.approx((2.5 + 5.5 + 8.5) / 3)
    assert metrics["per_example_total_median_seconds"] == pytest.approx(5.5)
