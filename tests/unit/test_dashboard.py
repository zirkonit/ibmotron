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

    phase = dashboard._derive_phase(job, remote)

    assert phase == "fine_tuned"


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
