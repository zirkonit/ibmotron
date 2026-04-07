from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibm650_it import REPO_ROOT
from ibm650_it.cloud import RunpodCtl
from ibm650_it.dataset.io import load_jsonl, resolve_record_base, resolve_record_path
from ibm650_it.eval.finalize import finalize_overfit_output, finalize_train_eval_output


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
READY_MARKER = "/workspace/ibmotron/.ibmotron_ready.json"
ARCHIVE_CACHE_DIR = REPO_ROOT / "artifacts" / "cache" / "runpod_archives"
REAL_BASELINE_QLORA_BITS = 0
REAL_BASELINE_LEARNING_RATE = 2e-4
REAL_BASELINE_EPOCHS = 5
REAL_BASELINE_MAX_NEW_TOKENS = 1024


REMOTE_BASE_INCLUDE_PATHS = [
    "README.md",
    "pyproject.toml",
    "ibm650_it",
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


def _iter_repo_paths(include_paths: list[str]) -> list[Path]:
    repo_paths: list[Path] = []
    for relative in include_paths:
        target = REPO_ROOT / relative
        if not target.exists():
            continue
        if target.is_file():
            repo_paths.append(target)
            continue
        for path in target.rglob("*"):
            if path.exists() and _include_path(path.relative_to(REPO_ROOT)):
                repo_paths.append(path)
    return sorted(repo_paths)


def _archive_fingerprint(repo_paths: list[Path]) -> str:
    payload = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in repo_paths
        if path.is_file()
    ]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def _write_archive(archive_path: Path, repo_paths: list[Path]) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in repo_paths:
            if not path.is_file():
                continue
            tar.add(path, arcname=f"ibmotron/{path.relative_to(REPO_ROOT)}", filter=_normalize_tarinfo)
    return archive_path


def build_base_archive() -> Path:
    repo_paths = _iter_repo_paths(REMOTE_BASE_INCLUDE_PATHS)
    fingerprint = _archive_fingerprint(repo_paths)
    archive_path = ARCHIVE_CACHE_DIR / f"base_{fingerprint}.tgz"
    if archive_path.exists():
        return archive_path
    return _write_archive(archive_path, repo_paths)


def _dataset_root_from_args(args: argparse.Namespace) -> Path:
    if args.run_mode == "overfit-sanity" and args.dataset_index:
        return resolve_record_base(Path(args.dataset_index))
    return REPO_ROOT / "artifacts" / "datasets" / args.dataset_name


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_dataset_caps(args: argparse.Namespace) -> tuple[int | None, int | None]:
    if args.run_mode == "overfit-sanity":
        return args.example_count, args.example_count
    if args.run_mode == "inference-only":
        if args.limit is not None:
            return None, args.limit
        eval_index = _dataset_index_paths(args)[0]
        return None, _count_jsonl_records(eval_index)
    dataset_root = _dataset_root_from_args(args)
    train_index = dataset_root / "splits" / args.train_split
    eval_index = dataset_root / "splits" / args.eval_split
    max_examples = args.max_examples if args.max_examples is not None else _count_jsonl_records(train_index)
    limit = args.limit if args.limit is not None else _count_jsonl_records(eval_index)
    return max_examples, limit


def _dataset_index_paths(args: argparse.Namespace) -> list[Path]:
    dataset_root = _dataset_root_from_args(args)
    if args.run_mode == "overfit-sanity":
        if args.dataset_index is None:
            return [dataset_root / "splits" / "synthetic_train.jsonl"]
        return [Path(args.dataset_index)]
    if args.run_mode == "inference-only":
        if args.reference_index is not None:
            return [Path(args.reference_index)]
        return [dataset_root / "splits" / args.eval_split]
    return [
        dataset_root / "splits" / args.train_split,
        dataset_root / "splits" / args.eval_split,
    ]


def _repo_relative(path: Path) -> Path:
    return path.resolve().relative_to(REPO_ROOT)


def _collect_remote_dataset_repo_paths(args: argparse.Namespace) -> list[Path]:
    dataset_root = _dataset_root_from_args(args)
    repo_paths: set[Path] = set()
    for index_path in _dataset_index_paths(args):
        resolved_index = index_path.resolve()
        repo_paths.add(REPO_ROOT / _repo_relative(resolved_index))
        base_dir = resolve_record_base(index_path)
        for record in load_jsonl(index_path):
            source_path = resolve_record_path(str(record["source"]["it_text_v1"]), base_dir)
            target_path = resolve_record_path(str(record["reference"]["translate"]["pit_raw_canonical"]), base_dir)
            repo_paths.add(REPO_ROOT / _repo_relative(source_path))
            repo_paths.add(REPO_ROOT / _repo_relative(target_path))
    summary_path = dataset_root / "summary.json"
    if summary_path.exists():
        repo_paths.add(summary_path)
    if args.run_mode == "inference-only":
        output_root = REPO_ROOT / args.local_output
        for relative in [
            output_root / "model",
            output_root / "predictions" / args.inference_mode,
            output_root / "summary.json",
        ]:
            if relative.is_file():
                repo_paths.add(relative)
            elif relative.is_dir():
                for path in relative.rglob("*"):
                    if path.is_file():
                        repo_paths.add(path)
    return sorted(repo_paths)


def build_dataset_archive(args: argparse.Namespace) -> Path:
    repo_paths = _collect_remote_dataset_repo_paths(args)
    dataset_root = _dataset_root_from_args(args)
    fingerprint = _archive_fingerprint(repo_paths)
    archive_path = ARCHIVE_CACHE_DIR / f"{dataset_root.name}_{fingerprint}.tgz"
    if archive_path.exists():
        return archive_path
    return _write_archive(archive_path, repo_paths)


def _base_runtime_steps() -> list[str]:
    return [
        "cd /workspace/ibmotron",
        ". .venv/bin/activate",
        "export HF_HOME=/workspace/.cache/huggingface",
        "mkdir -p \"$HF_HOME\"",
    ]


def remote_prepare_command() -> str:
    return " && ".join(
        [
            "export DEBIAN_FRONTEND=noninteractive",
            "apt-get update",
            "apt-get install -y build-essential git python3-pip python3-venv",
            "mkdir -p /workspace",
            "mkdir -p /workspace/ibmotron",
            f"rm -f {READY_MARKER}",
            "rm -rf /workspace/ibmotron/.venv",
            "tar --no-same-owner --no-same-permissions -xzf /workspace/ibmotron-base.tgz -C /workspace",
            "tar --no-same-owner --no-same-permissions -xzf /workspace/ibmotron-dataset.tgz -C /workspace",
            "cd /workspace/ibmotron",
            "python3 -m venv .venv --system-site-packages",
            ". .venv/bin/activate",
            "export HF_HOME=/workspace/.cache/huggingface",
            "mkdir -p \"$HF_HOME\"",
            "python -m pip install --upgrade pip",
            "pip install -e .",
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
            (
                "python -c "
                "\"import json; from pathlib import Path; "
                f"Path({READY_MARKER!r}).write_text(json.dumps({{'ready': True}}), encoding='utf-8')\""
            ),
        ]
    )


def remote_train_command(args: argparse.Namespace, remote_output: str, *, reuse_workspace: bool = False) -> str:
    prefix = _base_runtime_steps() if reuse_workspace else [remote_prepare_command(), *_base_runtime_steps()]
    max_examples, limit = resolve_dataset_caps(args)
    if args.run_mode == "overfit-sanity":
        dataset_index = args.dataset_index or f"artifacts/datasets/{args.dataset_name}/splits/synthetic_train.jsonl"
        return " && ".join(
            [
                *prefix,
                (
                    "python -m ibm650_it.cli overfit-sanity "
                    f"--dataset-index {dataset_index} "
                f"--output {remote_output} "
                f"--example-count {args.example_count} "
                f"--backend {args.backend} "
                f"--model-name {args.model_name} "
                f"--qlora-bits {args.qlora_bits} "
                    f"--learning-rate {args.learning_rate} "
                    f"--epochs {args.epochs} "
                    f"--max-seq-length {args.max_seq_length} "
                    f"--per-device-train-batch-size {args.per_device_train_batch_size} "
                f"--gradient-accumulation-steps {args.gradient_accumulation_steps} "
                f"--max-new-tokens {args.max_new_tokens} "
                "--eval-mode skip "
                f"--failure-archive-limit {args.failure_archive_limit} "
                f"--timeout-seconds {args.timeout_seconds}"
            ).strip(),
        ]
    )
    if args.run_mode == "inference-only":
        reference_index = args.reference_index or f"artifacts/datasets/{args.dataset_name}/splits/{args.eval_split}"
        model_path = args.model_path or f"{remote_output}/model"
        return " && ".join(
            [
                *prefix,
                (
                    "python -m ibm650_it.cli run-inference "
                    f"--reference-index {reference_index} "
                    f"--output {remote_output}/predictions/{args.inference_mode} "
                    f"--mode {args.inference_mode} "
                    f"--model {model_path} "
                    + (f"--limit {limit} " if limit is not None else "")
                    + f"--max-new-tokens {args.max_new_tokens} "
                    + "--eval-mode skip "
                    f"--timeout-seconds {args.timeout_seconds}"
                ).strip(),
            ]
        )
    return " && ".join(
        [
            *prefix,
            (
                "python -m ibm650_it.cli train-eval "
                f"--dataset-root artifacts/datasets/{args.dataset_name} "
                f"--output {remote_output} "
                f"--backend {args.backend} "
                f"--model-name {args.model_name} "
                f"--qlora-bits {args.qlora_bits} "
                f"--learning-rate {args.learning_rate} "
                f"--epochs {args.epochs} "
                f"--max-seq-length {args.max_seq_length} "
                f"--per-device-train-batch-size {args.per_device_train_batch_size} "
                f"--gradient-accumulation-steps {args.gradient_accumulation_steps} "
                f"--few-shot-k {args.few_shot_k} "
                + (f"--limit {limit} " if limit is not None else "")
                + (f"--max-examples {max_examples} " if max_examples is not None else "")
                + f"--max-new-tokens {args.max_new_tokens} "
                + "--eval-mode skip "
                + f"--failure-archive-limit {args.failure_archive_limit} "
                f"--timeout-seconds {args.timeout_seconds}"
            ).strip(),
        ]
    )


def _sync_remote_output(ctl: RunpodCtl, ssh_info: dict[str, object], remote_output_path: str, local_output_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        temp_root = Path(tempdir)
        ctl.scp_from(ssh_info, remote_output_path, temp_root)
        downloaded_root = temp_root / Path(remote_output_path).name
        if not downloaded_root.exists():
            raise FileNotFoundError(f"downloaded output missing: {downloaded_root}")
        shutil.copytree(downloaded_root, local_output_root, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ibmotron-runpod-train")
    parser.add_argument("--pod-id")
    parser.add_argument("--run-mode", choices=["train-eval", "overfit-sanity", "inference-only"], default="train-eval")
    parser.add_argument("--gpu-id", default="NVIDIA RTX A6000")
    parser.add_argument("--cloud-type", default="COMMUNITY")
    parser.add_argument("--image", default="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")
    parser.add_argument("--dataset-name", default="pilot_1000")
    parser.add_argument("--dataset-index")
    parser.add_argument("--reference-index")
    parser.add_argument("--backend", default="transformers_qlora")
    parser.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    parser.add_argument("--qlora-bits", type=int, default=REAL_BASELINE_QLORA_BITS)
    parser.add_argument("--learning-rate", type=float, default=REAL_BASELINE_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=REAL_BASELINE_EPOCHS)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--few-shot-k", type=int, default=4)
    parser.add_argument("--train-split", default="synthetic_train.jsonl")
    parser.add_argument("--eval-split", default="synthetic_dev.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--inference-mode", choices=["zero_shot", "few_shot", "fine_tuned"], default="fine_tuned")
    parser.add_argument("--model-path")
    parser.add_argument("--max-new-tokens", type=int, default=REAL_BASELINE_MAX_NEW_TOKENS)
    parser.add_argument("--failure-archive-limit", type=int, default=25)
    parser.add_argument("--step-budget", default="50M")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--remote-output", default="artifacts/eval_reports/runpod_train_eval")
    parser.add_argument("--local-output", default="artifacts/eval_reports/runpod_train_eval")
    parser.add_argument("--example-count", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-pod-workspace", action="store_true")
    parser.add_argument("--keep-pod", action="store_true")
    args = parser.parse_args()

    if args.prepare_only and not (args.keep_pod or args.pod_id):
        raise SystemExit("--prepare-only requires --keep-pod or --pod-id so the prepared pod is retained")
    if args.reuse_pod_workspace and not args.pod_id:
        raise SystemExit("--reuse-pod-workspace requires --pod-id")

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
    try:
        ssh_info = ctl.wait_for_ssh(pod_id, timeout_seconds=600, poll_seconds=10)
        if args.prepare_only or not args.reuse_pod_workspace:
            base_archive = build_base_archive()
            dataset_archive = build_dataset_archive(args)
            ctl.scp_to(ssh_info, base_archive, "/workspace/ibmotron-base.tgz")
            ctl.scp_to(ssh_info, dataset_archive, "/workspace/ibmotron-dataset.tgz")
        if args.reuse_pod_workspace:
            ready = ctl.ssh(ssh_info, f"test -f {READY_MARKER}", check=False)
            if ready.returncode != 0:
                raise SystemExit(f"remote workspace on pod {pod_id} is not prepared; missing {READY_MARKER}")
        remote_command = remote_prepare_command() if args.prepare_only else remote_train_command(
            args,
            args.remote_output,
            reuse_workspace=args.reuse_pod_workspace,
        )
        proc = ctl.ssh(ssh_info, remote_command, check=False)
        logs_dir = REPO_ROOT / "artifacts" / "logs" / "runpod"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_stem = f"{pod_id}.{args.name}"
        (logs_dir / f"{log_stem}.stdout.log").write_text(proc.stdout, encoding="utf-8")
        (logs_dir / f"{log_stem}.stderr.log").write_text(proc.stderr, encoding="utf-8")
        if args.prepare_only:
            summary = {
                "pod_id": pod_id,
                "pod": pod,
                "ssh": ssh_info,
                "returncode": proc.returncode,
                "prepared": proc.returncode == 0,
                "ready_marker": READY_MARKER,
            }
            print(json.dumps(summary, indent=2))
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)
            return
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
        local_output_root = REPO_ROOT / args.local_output
        _sync_remote_output(ctl, ssh_info, remote_output_path, local_output_root)
        subprocess.run(["./scripts/build_simh.sh"], cwd=REPO_ROOT, check=True)
        if args.run_mode == "overfit-sanity":
            finalize_overfit_output(
                dataset_index=REPO_ROOT / args.dataset_index,
                output_root=local_output_root,
                failure_archive_limit=args.failure_archive_limit,
                repo_root=REPO_ROOT,
                step_budget=args.step_budget,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.run_mode == "inference-only":
            finalize_train_eval_output(
                dataset_root=REPO_ROOT / f"artifacts/datasets/{args.dataset_name}",
                output_root=local_output_root,
                eval_split=args.eval_split,
                failure_archive_limit=args.failure_archive_limit,
                repo_root=REPO_ROOT,
                step_budget=args.step_budget,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            finalize_train_eval_output(
                dataset_root=REPO_ROOT / f"artifacts/datasets/{args.dataset_name}",
                output_root=local_output_root,
                eval_split=args.eval_split,
                failure_archive_limit=args.failure_archive_limit,
                repo_root=REPO_ROOT,
                step_budget=args.step_budget,
                timeout_seconds=args.timeout_seconds,
            )
        summary = {
            "pod_id": pod_id,
            "pod": pod,
            "ssh": ssh_info,
            "returncode": proc.returncode,
            "local_output": str(local_output_root),
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
