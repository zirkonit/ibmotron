from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from scripts import runpod_train_eval


def test_remote_train_command_uses_no_same_owner() -> None:
    args = argparse.Namespace(
        run_mode="train-eval",
        dataset_name="pilot_remote_128_20",
        dataset_index=None,
        backend="transformers_qlora",
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        qlora_bits=0,
        learning_rate=1e-4,
        epochs=1,
        max_seq_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        few_shot_k=4,
        train_split="synthetic_train.jsonl",
        eval_split="synthetic_dev.jsonl",
        limit=20,
        max_examples=128,
        max_new_tokens=1024,
        failure_archive_limit=25,
        step_budget="50M",
        timeout_seconds=30,
        example_count=32,
    )
    command = runpod_train_eval.remote_train_command(args, "artifacts/eval_reports/out")
    assert "tar --no-same-owner --no-same-permissions -xzf /workspace/ibmotron-base.tgz -C /workspace" in command
    assert "tar --no-same-owner --no-same-permissions -xzf /workspace/ibmotron-dataset.tgz -C /workspace" in command
    assert "./scripts/build_simh.sh" not in command
    assert "--eval-mode skip" in command


def test_remote_prepare_command_bootstraps_runtime_only() -> None:
    command = runpod_train_eval.remote_prepare_command()

    assert "apt-get install -y build-essential git python3-pip python3-venv" in command
    assert "pip install -e ." in command
    assert "python -m ibm650_it.cli train-eval" not in command
    assert "python -m ibm650_it.cli overfit-sanity" not in command
    assert runpod_train_eval.READY_MARKER in command


def test_remote_train_command_reuse_workspace_skips_bootstrap() -> None:
    args = argparse.Namespace(
        run_mode="train-eval",
        dataset_name="pilot_remote_128_20",
        dataset_index=None,
        backend="transformers_qlora",
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        qlora_bits=0,
        learning_rate=1e-4,
        epochs=1,
        max_seq_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        few_shot_k=4,
        train_split="synthetic_train.jsonl",
        eval_split="synthetic_dev.jsonl",
        limit=20,
        max_examples=128,
        max_new_tokens=1024,
        failure_archive_limit=25,
        step_budget="50M",
        timeout_seconds=30,
        example_count=32,
    )

    command = runpod_train_eval.remote_train_command(args, "artifacts/eval_reports/out", reuse_workspace=True)

    assert "ibmotron-base.tgz" not in command
    assert "ibmotron-dataset.tgz" not in command
    assert "apt-get update" not in command
    assert "cd /workspace/ibmotron" in command
    assert "python -m ibm650_it.cli train-eval" in command


def test_remote_overfit_command_uses_overfit_sanity() -> None:
    args = argparse.Namespace(
        run_mode="overfit-sanity",
        dataset_name="pilot_remote_128_20",
        dataset_index="artifacts/datasets/pilot_remote_128_20/splits/synthetic_train.jsonl",
        backend="transformers_qlora",
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        qlora_bits=0,
        learning_rate=1e-4,
        epochs=1,
        max_seq_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        few_shot_k=4,
        train_split="synthetic_train.jsonl",
        eval_split="synthetic_dev.jsonl",
        limit=20,
        max_examples=128,
        max_new_tokens=1024,
        failure_archive_limit=25,
        step_budget="50M",
        timeout_seconds=30,
        example_count=32,
    )
    command = runpod_train_eval.remote_train_command(args, "artifacts/eval_reports/out")
    assert "python -m ibm650_it.cli overfit-sanity" in command
    assert "--dataset-index artifacts/datasets/pilot_remote_128_20/splits/synthetic_train.jsonl" in command
    assert "--example-count 32" in command
    assert "--eval-mode skip" in command


def test_normalize_tarinfo_clears_ownership_metadata() -> None:
    info = tarfile.TarInfo("sample.txt")
    info.uid = 501
    info.gid = 50
    info.uname = "i"
    info.gname = "staff"

    normalized = runpod_train_eval._normalize_tarinfo(info)

    assert normalized.uid == 0
    assert normalized.gid == 0
    assert normalized.uname == ""
    assert normalized.gname == ""


def test_collect_remote_dataset_repo_paths_keeps_only_split_and_training_targets(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    dataset_root = repo_root / "artifacts" / "datasets" / "mini"
    accepted = dataset_root / "accepted" / "B0" / "0001_000001"
    splits = dataset_root / "splits"
    accepted.mkdir(parents=True)
    (accepted / "pipeline" / "translate").mkdir(parents=True)
    splits.mkdir(parents=True)
    (accepted / "source.it").write_text("+ 0 0 1 1 1732\n0001+ h ff\n", encoding="utf-8")
    (accepted / "pipeline" / "translate" / "pit_raw_canonical.dck").write_text("s0001 00 0000 laaaa\n", encoding="utf-8")
    record = {
        "id": "sample-1",
        "band": "B0",
        "source": {"it_text_v1": "accepted/B0/0001_000001/source.it"},
        "reference": {"translate": {"pit_raw_canonical": "accepted/B0/0001_000001/pipeline/translate/pit_raw_canonical.dck"}},
    }
    for split_name in ["synthetic_train.jsonl", "synthetic_dev.jsonl"]:
        (splits / split_name).write_text(json.dumps(record) + "\n", encoding="utf-8")
    (dataset_root / "summary.json").write_text("{}", encoding="utf-8")
    (accepted / "pipeline" / "run" / "spit_p1.dck").parent.mkdir(parents=True)
    (accepted / "pipeline" / "run" / "spit_p1.dck").write_text("should_not_be_uploaded\n", encoding="utf-8")

    monkeypatch.setattr(runpod_train_eval, "REPO_ROOT", repo_root)
    monkeypatch.setattr(runpod_train_eval, "ARCHIVE_CACHE_DIR", repo_root / "cache")

    args = argparse.Namespace(
        run_mode="train-eval",
        dataset_name="mini",
        dataset_index=None,
        train_split="synthetic_train.jsonl",
        eval_split="synthetic_dev.jsonl",
    )

    repo_paths = runpod_train_eval._collect_remote_dataset_repo_paths(args)
    rel_paths = [str(path.relative_to(repo_root)) for path in repo_paths]

    assert "artifacts/datasets/mini/splits/synthetic_train.jsonl" in rel_paths
    assert "artifacts/datasets/mini/splits/synthetic_dev.jsonl" in rel_paths
    assert "artifacts/datasets/mini/accepted/B0/0001_000001/source.it" in rel_paths
    assert "artifacts/datasets/mini/accepted/B0/0001_000001/pipeline/translate/pit_raw_canonical.dck" in rel_paths
    assert "artifacts/datasets/mini/accepted/B0/0001_000001/pipeline/run/spit_p1.dck" not in rel_paths
