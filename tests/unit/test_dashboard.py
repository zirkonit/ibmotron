from __future__ import annotations

import json
from pathlib import Path

from ibm650_it import dashboard


def test_parse_launcher_args_extracts_key_flags() -> None:
    command = (
        "python3 scripts/runpod_train_eval.py "
        "--name ibmotron-a40-rerun128-e3 "
        "--run-mode train-eval "
        "--dataset-name pilot_remote_128_20 "
        "--gpu-id NVIDIA A40 "
        "--limit 20 "
        "--remote-output artifacts/eval_reports/out "
        "--local-output artifacts/eval_reports/out_local"
    )

    parsed = dashboard._parse_launcher_args(command)

    assert parsed["name"] == "ibmotron-a40-rerun128-e3"
    assert parsed["run_mode"] == "train-eval"
    assert parsed["dataset_name"] == "pilot_remote_128_20"
    assert parsed["gpu_id"] == "NVIDIA A40"
    assert parsed["limit"] == "20"
    assert parsed["remote_output"] == "artifacts/eval_reports/out"


def test_derive_phase_uses_remote_progress() -> None:
    job = {"pid": 123, "run_mode": "train-eval", "limit": 20}
    remote = {
        "tgz_exists": True,
        "repo_exists": True,
        "output_exists": True,
        "active_process": True,
        "progress": {
            "zero_shot": {"lines": 20, "expected": 20},
            "few_shot": {"lines": 20, "expected": 20},
            "fine_tuned": {"lines": 7, "expected": 20},
        },
    }

    phase = dashboard._derive_phase(job, remote, None)

    assert phase == "remote_generate"


def test_derive_phase_handles_reused_workspace_without_tgz() -> None:
    job = {"pid": 123, "run_mode": "train-eval", "limit": 20}
    remote = {
        "tgz_exists": False,
        "repo_exists": True,
        "output_exists": True,
        "active_process": True,
        "progress": {
            "zero_shot": {"lines": 20, "expected": 20},
            "few_shot": {"lines": 20, "expected": 20},
            "fine_tuned": {"lines": 4, "expected": 20},
        },
    }

    phase = dashboard._derive_phase(job, remote, None)

    assert phase == "remote_generate"


def test_match_pod_prefers_explicit_pod_id() -> None:
    job = {"name": "run-name", "pod_id_arg": "warm-pod-123"}
    pods_by_id = {
        "warm-pod-123": {"id": "warm-pod-123", "name": "shared-warm-pod"},
    }
    pods_by_name = {
        "run-name": {"id": "cold-pod-999", "name": "run-name"},
    }

    pod = dashboard._match_pod(job, pods_by_id, pods_by_name)

    assert pod == {"id": "warm-pod-123", "name": "shared-warm-pod"}


def test_collect_recent_runs_reads_top_level_summaries(tmp_path: Path, monkeypatch) -> None:
    eval_root = tmp_path / "artifacts" / "eval_reports"
    run_dir = eval_root / "run_one"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "eval_mode": "local_cpu_reevaluate",
                "train": {"backend": "transformers_qlora", "qlora_bits": 0},
                "fine_tuned": {"report": {"exact_match": 0.5, "assemblability": 1.0, "functional_equivalence": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "REPO_ROOT", tmp_path)

    runs = dashboard.collect_recent_runs(limit=5)

    assert len(runs) == 1
    assert runs[0]["path"] == "artifacts/eval_reports/run_one"
    assert runs[0]["eval_mode"] == "local_cpu_reevaluate"
    assert runs[0]["train"]["backend"] == "transformers_qlora"


def test_collect_recent_runs_reads_nested_sweep_summaries(tmp_path: Path, monkeypatch) -> None:
    eval_root = tmp_path / "artifacts" / "eval_reports" / "sweeps" / "example" / "e5_lr0p0002"
    eval_root.mkdir(parents=True)
    (eval_root / "summary.json").write_text(
        json.dumps(
            {
                "eval_mode": "local_cpu_reevaluate",
                "train": {"backend": "transformers_qlora", "qlora_bits": 0},
                "evaluations": {"fine_tuned": {"report": {"exact_match": 0.4, "assemblability": 1.0, "functional_equivalence": 0.6}}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "REPO_ROOT", tmp_path)

    runs = dashboard.collect_recent_runs(limit=5)

    assert len(runs) == 1
    assert runs[0]["path"] == "artifacts/eval_reports/sweeps/example/e5_lr0p0002"


def test_inspect_local_output_reports_local_reevaluate_when_finalize_state_exists(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "artifacts" / "eval_reports" / "run_one"
    output_root.mkdir(parents=True)
    (output_root / ".finalize_state.json").write_text(
        json.dumps({"status": "running", "current_mode": "fine_tuned"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "REPO_ROOT", tmp_path)

    local = dashboard._inspect_local_output("artifacts/eval_reports/run_one", "train-eval")

    assert local is not None
    assert local["phase"] == "local_reevaluate"
    assert local["state"]["current_mode"] == "fine_tuned"


def test_collect_dashboard_status_matches_warm_pod_by_id(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "collect_active_wrappers",
        lambda: [
            {
                "pid": 123,
                "elapsed": "00:10",
                "command": "python scripts/runpod_train_eval.py",
                "name": "run-three",
                "run_mode": "train-eval",
                "remote_output": "artifacts/eval_reports/out",
                "local_output": "artifacts/eval_reports/out",
                "limit": 20,
                "example_count": None,
                "pod_id_arg": "warm-pod-123",
            }
        ],
    )
    monkeypatch.setattr(
        dashboard,
        "collect_pods",
        lambda: [
            {
                "id": "warm-pod-123",
                "name": "shared-warm-pod",
                "gpu": "NVIDIA A40",
                "cost_per_hr": 0.4,
                "status": "RUNNING",
            }
        ],
    )
    monkeypatch.setattr(
        dashboard,
        "_ssh_remote_status",
        lambda *args, **kwargs: {
            "tgz_exists": False,
            "repo_exists": True,
            "output_exists": True,
            "active_process": True,
            "progress": {
                "zero_shot": {"lines": 20, "expected": 20},
                "few_shot": {"lines": 20, "expected": 20},
                "fine_tuned": {"lines": 3, "expected": 20},
            },
            "gpu": "96, 30113, 46068",
        },
    )
    monkeypatch.setattr(dashboard, "_read_archive_size", lambda pid: None)

    status = dashboard.collect_dashboard_status(dashboard.DashboardConfig())

    assert len(status["active_jobs"]) == 1
    job = status["active_jobs"][0]
    assert job["pod_id"] == "warm-pod-123"
    assert job["phase"] == "remote_generate"
    assert job["remote"]["phase"] == "fine_tuned"
    assert status["orphan_pods"] == []
