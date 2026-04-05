from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibm650_it import REPO_ROOT
from ibm650_it.cloud import RunpodCtl


ACCELERATE_VERSION = "1.10.0"
BITSANDBYTES_VERSION = "0.47.0"
DATASETS_VERSION = "4.0.0"
PEFT_VERSION = "0.17.0"
TRANSFORMERS_VERSION = "4.56.0"
MAMBA_WHEEL_URL = (
    "https://github.com/state-spaces/mamba/releases/download/v2.3.1/"
    "mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
CAUSAL_CONV1D_WHEEL_URL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/"
    "causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)


INCLUDE_PATHS = [
    "README.md",
    "Makefile",
    "SPEC.md",
    "pyproject.toml",
    "sources.lock.json",
    "docker",
    "docs",
    "scripts",
    "ibm650_it",
    "tests",
    "third_party/simh",
]

EXCLUDE_PARTS = {
    ".env",
    ".git",
    ".gitignore",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    ".DS_Store",
    "artifacts/eval_reports",
    "artifacts/models",
    "artifacts/failures",
}


def _include_path(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_PARTS:
            return False
    normalized = str(path)
    return not any(normalized.startswith(prefix) for prefix in EXCLUDE_PARTS)


def _normalize_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def build_archive(archive_path: Path, *, dataset_name: str) -> Path:
    include_paths = [*INCLUDE_PATHS, f"artifacts/datasets/{dataset_name}"]
    with tarfile.open(archive_path, "w:gz") as tar:
        for relative in include_paths:
            target = REPO_ROOT / relative
            if not target.exists():
                continue
            if target.is_file():
                tar.add(target, arcname=f"ibmotron/{relative}", filter=_normalize_tarinfo)
                continue
            for path in target.rglob("*"):
                if not path.exists() or not _include_path(path.relative_to(REPO_ROOT)):
                    continue
                tar.add(path, arcname=f"ibmotron/{path.relative_to(REPO_ROOT)}", filter=_normalize_tarinfo)
    return archive_path


def remote_train_command(args: argparse.Namespace, remote_output: str) -> str:
    if args.run_mode == "overfit-sanity":
        dataset_index = args.dataset_index or f"artifacts/datasets/{args.dataset_name}/splits/synthetic_train.jsonl"
        return " && ".join(
            [
                "export DEBIAN_FRONTEND=noninteractive",
                "apt-get update",
                "apt-get install -y build-essential git python3-pip python3-venv",
                "mkdir -p /workspace",
                "rm -rf /workspace/ibmotron",
                "tar --no-same-owner --no-same-permissions -xzf /workspace/ibmotron.tgz -C /workspace",
                "cd /workspace/ibmotron",
                "python3 -m venv .venv --system-site-packages",
                ". .venv/bin/activate",
                "export HF_HOME=/workspace/.cache/huggingface",
                "mkdir -p \"$HF_HOME\"",
                "python -m pip install --upgrade pip",
                "pip install -e \".[dev]\"",
                (
                    "pip install "
                    f"accelerate=={ACCELERATE_VERSION} "
                    f"bitsandbytes=={BITSANDBYTES_VERSION} "
                    f"datasets=={DATASETS_VERSION} "
                    f"peft=={PEFT_VERSION} "
                    f"transformers=={TRANSFORMERS_VERSION}"
                ),
                f"pip install {MAMBA_WHEEL_URL}",
                f"pip install {CAUSAL_CONV1D_WHEEL_URL}",
                "./scripts/build_simh.sh",
                (
                    "python -m ibm650_it.cli overfit-sanity "
                    f"--dataset-index {dataset_index} "
                    f"--output {remote_output} "
                    f"--example-count {args.example_count} "
                    f"--backend {args.backend} "
                    f"--model-name {args.model_name} "
                    f"--qlora-bits {args.qlora_bits} "
                    f"--epochs {args.epochs} "
                    f"--max-seq-length {args.max_seq_length} "
                    f"--per-device-train-batch-size {args.per_device_train_batch_size} "
                    f"--gradient-accumulation-steps {args.gradient_accumulation_steps} "
                    f"--max-new-tokens {args.max_new_tokens} "
                    f"--failure-archive-limit {args.failure_archive_limit} "
                    f"--timeout-seconds {args.timeout_seconds}"
                ).strip(),
            ]
        )
    return " && ".join(
        [
            "export DEBIAN_FRONTEND=noninteractive",
            "apt-get update",
            "apt-get install -y build-essential git python3-pip python3-venv",
            "mkdir -p /workspace",
            "rm -rf /workspace/ibmotron",
            "tar --no-same-owner --no-same-permissions -xzf /workspace/ibmotron.tgz -C /workspace",
            "cd /workspace/ibmotron",
            "python3 -m venv .venv --system-site-packages",
            ". .venv/bin/activate",
            "export HF_HOME=/workspace/.cache/huggingface",
            "mkdir -p \"$HF_HOME\"",
            "python -m pip install --upgrade pip",
            "pip install -e \".[dev]\"",
            (
                "pip install "
                f"accelerate=={ACCELERATE_VERSION} "
                f"bitsandbytes=={BITSANDBYTES_VERSION} "
                f"datasets=={DATASETS_VERSION} "
                f"peft=={PEFT_VERSION} "
                f"transformers=={TRANSFORMERS_VERSION}"
            ),
            f"pip install {MAMBA_WHEEL_URL}",
            f"pip install {CAUSAL_CONV1D_WHEEL_URL}",
            "./scripts/build_simh.sh",
            (
                "python -m ibm650_it.cli train-eval "
                f"--dataset-root artifacts/datasets/{args.dataset_name} "
                f"--output {remote_output} "
                f"--backend {args.backend} "
                f"--model-name {args.model_name} "
                f"--qlora-bits {args.qlora_bits} "
                f"--epochs {args.epochs} "
                f"--max-seq-length {args.max_seq_length} "
                f"--per-device-train-batch-size {args.per_device_train_batch_size} "
                f"--gradient-accumulation-steps {args.gradient_accumulation_steps} "
                f"--few-shot-k {args.few_shot_k} "
                + (f"--limit {args.limit} " if args.limit is not None else "")
                + (f"--max-examples {args.max_examples} " if args.max_examples is not None else "")
                + f"--max-new-tokens {args.max_new_tokens} "
                + f"--failure-archive-limit {args.failure_archive_limit} "
                f"--timeout-seconds {args.timeout_seconds}"
            ).strip(),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ibmotron-runpod-train")
    parser.add_argument("--pod-id")
    parser.add_argument("--run-mode", choices=["train-eval", "overfit-sanity"], default="train-eval")
    parser.add_argument("--gpu-id", default="NVIDIA RTX A6000")
    parser.add_argument("--cloud-type", default="COMMUNITY")
    parser.add_argument("--image", default="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")
    parser.add_argument("--dataset-name", default="pilot_1000")
    parser.add_argument("--dataset-index")
    parser.add_argument("--backend", default="transformers_qlora")
    parser.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    parser.add_argument("--qlora-bits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--few-shot-k", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--failure-archive-limit", type=int, default=25)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--remote-output", default="artifacts/eval_reports/runpod_train_eval")
    parser.add_argument("--local-output", default="artifacts/eval_reports/runpod_train_eval")
    parser.add_argument("--example-count", type=int, default=32)
    parser.add_argument("--keep-pod", action="store_true")
    args = parser.parse_args()

    ctl = RunpodCtl(repo_root=REPO_ROOT)
    ctl.ensure_ssh_key()

    pod = (
        ctl.get_pod(args.pod_id)
        if args.pod_id
        else ctl.create_pod(
            name=args.name,
            gpu_id=args.gpu_id,
            image=args.image,
            cloud_type=args.cloud_type,
            public_ip=args.cloud_type.upper() == "COMMUNITY",
        )
    )
    pod_id = pod["id"]
    archive_path = Path(tempfile.gettempdir()) / "ibmotron-runpod.tgz"
    try:
        build_archive(archive_path, dataset_name=args.dataset_name)
        ssh_info = ctl.wait_for_ssh(pod_id, timeout_seconds=600, poll_seconds=10)
        ctl.scp_to(ssh_info, archive_path, "/workspace/ibmotron.tgz")
        remote_command = remote_train_command(args, args.remote_output)
        proc = ctl.ssh(ssh_info, remote_command, check=False)
        logs_dir = REPO_ROOT / "artifacts" / "logs" / "runpod"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / f"{pod_id}.stdout.log").write_text(proc.stdout, encoding="utf-8")
        (logs_dir / f"{pod_id}.stderr.log").write_text(proc.stderr, encoding="utf-8")
        remote_output_path = f"/workspace/ibmotron/{args.remote_output}"
        exists = ctl.ssh(ssh_info, f"test -e {remote_output_path}", check=False)
        if exists.returncode != 0:
            summary = {
                "pod_id": pod_id,
                "pod": pod,
                "ssh": ssh_info,
                "returncode": proc.returncode,
                "remote_output_missing": remote_output_path,
                "local_output": str(REPO_ROOT / args.local_output),
            }
            print(json.dumps(summary, indent=2))
            raise SystemExit(proc.returncode or 1)
        ctl.scp_from(ssh_info, remote_output_path, REPO_ROOT / args.local_output)
        summary = {
            "pod_id": pod_id,
            "pod": pod,
            "ssh": ssh_info,
            "returncode": proc.returncode,
            "local_output": str(REPO_ROOT / args.local_output),
        }
        print(json.dumps(summary, indent=2))
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
    finally:
        if not args.keep_pod and not args.pod_id:
            try:
                ctl.delete_pod(pod_id)
            except Exception:
                pass


if __name__ == "__main__":
    main()
