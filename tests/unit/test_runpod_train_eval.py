from __future__ import annotations

import argparse
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
        epochs=1,
        max_seq_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        few_shot_k=4,
        limit=20,
        max_examples=128,
        max_new_tokens=1024,
        failure_archive_limit=25,
        timeout_seconds=30,
        example_count=32,
    )
    command = runpod_train_eval.remote_train_command(args, "artifacts/eval_reports/out")
    assert "tar --no-same-owner --no-same-permissions -xzf /workspace/ibmotron.tgz -C /workspace" in command


def test_remote_overfit_command_uses_overfit_sanity() -> None:
    args = argparse.Namespace(
        run_mode="overfit-sanity",
        dataset_name="pilot_remote_128_20",
        dataset_index="artifacts/datasets/pilot_remote_128_20/splits/synthetic_train.jsonl",
        backend="transformers_qlora",
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        qlora_bits=0,
        epochs=1,
        max_seq_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        few_shot_k=4,
        limit=20,
        max_examples=128,
        max_new_tokens=1024,
        failure_archive_limit=25,
        timeout_seconds=30,
        example_count=32,
    )
    command = runpod_train_eval.remote_train_command(args, "artifacts/eval_reports/out")
    assert "python -m ibm650_it.cli overfit-sanity" in command
    assert "--dataset-index artifacts/datasets/pilot_remote_128_20/splits/synthetic_train.jsonl" in command
    assert "--example-count 32" in command


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
