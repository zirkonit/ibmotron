from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
import shlex

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibm650_it import REPO_ROOT
from ibm650_it.cloud import RunpodCtl
from ibm650_it.dataset.io import load_jsonl, resolve_record_base, resolve_record_path
from ibm650_it.eval.finalize import finalize_overfit_output, finalize_train_eval_output
from ibm650_it.training.hf_qlora import DEFAULT_TRANSFORMERS_QLORA_MODEL


ACCELERATE_VERSION = "1.10.0"
BITSANDBYTES_VERSION = "0.47.0"
DATASETS_VERSION = "4.0.0"
PEFT_VERSION = "0.18.1"
TRANSFORMERS_VERSION = "4.56.0"
# Qwen3.5 support has not landed in any PyPI release yet — install from git main.
QWEN35_TRANSFORMERS_REQ = "transformers @ git+https://github.com/huggingface/transformers.git@main"
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
LOCAL_JOB_METADATA_FILENAME = ".runpod_job.json"
REMOTE_JOB_SCRIPT_FILENAME = ".runpod_remote_job.sh"
REMOTE_JOB_STATE_FILENAME = ".runpod_remote_state.json"
REMOTE_JOB_LOG_FILENAME = ".runpod_remote.log"
COLLECT_WATCHER_LOG_FILENAME = ".runpod_collect_watcher.log"
REAL_BASELINE_QLORA_BITS = 0
REAL_BASELINE_LEARNING_RATE = 2e-4
REAL_BASELINE_EPOCHS = 5
# The B4/B5 expansion materially lengthens PIT deck tails beyond the B2/B3-only
# stage_2k profile. 2048 keeps greedy eval out of the truncation path while still
# fitting comfortably on the A40/A6000 class pods used by the repo.
REAL_BASELINE_MAX_NEW_TOKENS = 2048


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


def _requires_mamba_runtime(model_name: str) -> bool:
    return "nemotron" in model_name.lower()


def _requires_qwen35_transformers_pin(model_name: str) -> bool:
    normalized = model_name.lower()
    return normalized.startswith("qwen/qwen3.5") or "/qwen3.5" in normalized


def _transformers_requirement(model_name: str) -> str:
    if _requires_qwen35_transformers_pin(model_name):
        return f"'{QWEN35_TRANSFORMERS_REQ}'"
    return f"transformers=={TRANSFORMERS_VERSION}"


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
        if args.model_path is not None:
            model_root = (REPO_ROOT / args.model_path).resolve()
            if model_root.is_file():
                repo_paths.add(REPO_ROOT / _repo_relative(model_root))
            elif model_root.is_dir():
                for path in model_root.rglob("*"):
                    if path.is_file():
                        repo_paths.add(path)
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


def _dataset_refresh_steps() -> list[str]:
    return [
        "mkdir -p /workspace",
        "tar --no-same-owner --no-same-permissions -xzf /workspace/ibmotron-dataset.tgz -C /workspace",
    ]


def _band_repeat_flags(args: argparse.Namespace) -> str:
    flags: list[str] = []
    preset = getattr(args, "band_repeat_preset", None)
    if preset:
        flags.append(f"--band-repeat-preset {preset}")
    for value in getattr(args, "band_repeat", []) or []:
        flags.append(f"--band-repeat {value}")
    return " ".join(flags)


def _mode_flags(args: argparse.Namespace) -> str:
    modes = list(getattr(args, "modes", []) or [])
    if not modes:
        return ""
    return " ".join(f"--modes {mode}" for mode in modes)


def remote_prepare_command(model_name: str = DEFAULT_TRANSFORMERS_QLORA_MODEL) -> str:
    steps = [
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
            f"{_transformers_requirement(model_name)}"
        ),
    ]
    if _requires_mamba_runtime(model_name):
        steps.extend(
            [
                f"pip install {MAMBA_WHEEL_URL}",
                f"pip install {CAUSAL_CONV1D_WHEEL_URL}",
            ]
        )
    steps.append(
        (
            "python -c "
            "\"import json; from pathlib import Path; "
            f"Path({READY_MARKER!r}).write_text(json.dumps({{'ready': True}}), encoding='utf-8')\""
        )
    )
    return " && ".join(steps)


def remote_train_command(args: argparse.Namespace, remote_output: str, *, reuse_workspace: bool = False) -> str:
    prefix = (
        [*_dataset_refresh_steps(), *_base_runtime_steps()]
        if reuse_workspace
        else [remote_prepare_command(args.model_name), *_base_runtime_steps()]
    )
    max_examples, limit = resolve_dataset_caps(args)
    band_repeat_flags = _band_repeat_flags(args)
    mode_flags = _mode_flags(args)
    batch_size = int(getattr(args, "inference_batch_size", 1) or 1)
    batch_flag = f" --inference-batch-size {batch_size}" if batch_size > 1 else ""
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
                    + (f"{band_repeat_flags} " if band_repeat_flags else "")
                    + f"--max-new-tokens {args.max_new_tokens}"
                    + batch_flag
                    + " --eval-mode skip "
                    + f"--failure-archive-limit {args.failure_archive_limit} "
                    + f"--timeout-seconds {args.timeout_seconds}"
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
                    + f"--max-new-tokens {args.max_new_tokens}"
                    + batch_flag
                    + " --eval-mode skip "
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
                f"--train-split {args.train_split} "
                f"--eval-split {args.eval_split} "
                + (f"{mode_flags} " if mode_flags else "")
                + (f"{band_repeat_flags} " if band_repeat_flags else "")
                + (f"--limit {limit} " if limit is not None else "")
                + (f"--max-examples {max_examples} " if max_examples is not None else "")
                + f"--max-new-tokens {args.max_new_tokens}"
                + batch_flag
                + " --eval-mode skip "
                + f"--failure-archive-limit {args.failure_archive_limit} "
                + f"--timeout-seconds {args.timeout_seconds}"
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


def _metadata_path(local_output_root: Path) -> Path:
    return local_output_root / LOCAL_JOB_METADATA_FILENAME


def _write_job_metadata(local_output_root: Path, payload: dict[str, object]) -> Path:
    local_output_root.mkdir(parents=True, exist_ok=True)
    path = _metadata_path(local_output_root)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_job_metadata(local_output_root: Path) -> dict[str, object] | None:
    path = _metadata_path(local_output_root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _remote_output_paths(remote_output: str) -> dict[str, str]:
    remote_root = f"/workspace/ibmotron/{remote_output}"
    return {
        "root": remote_root,
        "state": f"{remote_root}/{REMOTE_JOB_STATE_FILENAME}",
        "log": f"{remote_root}/{REMOTE_JOB_LOG_FILENAME}",
        "script": f"{remote_root}/{REMOTE_JOB_SCRIPT_FILENAME}",
    }


def _job_metadata_payload(args: argparse.Namespace, *, pod_id: str, detached: bool, status: str) -> dict[str, object]:
    return {
        "name": args.name,
        "pod_id": pod_id,
        "run_mode": args.run_mode,
        "dataset_name": args.dataset_name,
        "dataset_index": args.dataset_index,
        "reference_index": args.reference_index,
        "eval_split": args.eval_split,
        "inference_mode": args.inference_mode,
        "modes": list(getattr(args, "modes", []) or []),
        "remote_output": args.remote_output,
        "local_output": args.local_output,
        "band_repeat": list(getattr(args, "band_repeat", []) or []),
        "band_repeat_preset": getattr(args, "band_repeat_preset", None),
        "max_examples": args.max_examples,
        "limit": args.limit,
        "detached": detached,
        "status": status,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _watcher_log_path(local_output_root: Path) -> Path:
    return local_output_root / COLLECT_WATCHER_LOG_FILENAME


def _spawn_collect_watcher(args: argparse.Namespace, local_output_root: Path) -> int:
    local_output_root.mkdir(parents=True, exist_ok=True)
    log_path = _watcher_log_path(local_output_root)
    with log_path.open("ab") as log_handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--watch-collect",
                "--local-output",
                args.local_output,
                "--auto-collect-poll-seconds",
                str(args.auto_collect_poll_seconds),
            ],
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def _build_remote_job_script(args: argparse.Namespace, remote_command: str) -> str:
    remote_paths = _remote_output_paths(args.remote_output)
    state_path = remote_paths["state"]
    output_root = remote_paths["root"]
    running_writer = (
        "import json; "
        "from datetime import datetime; "
        "from pathlib import Path; "
        f"path = Path({state_path!r}); "
        "path.write_text(json.dumps({"
        f"'status': 'running', 'run_mode': {args.run_mode!r}, 'name': {args.name!r}, "
        "'updated_at': datetime.now().astimezone().isoformat(timespec='seconds')"
        "}, indent=2), encoding='utf-8')"
    )
    complete_writer = (
        "import json, os; "
        "from datetime import datetime; "
        "from pathlib import Path; "
        f"path = Path({state_path!r}); "
        "status = int(os.environ.get('IBMOTRON_JOB_STATUS', '0')); "
        "path.write_text(json.dumps({"
        "'status': 'complete' if status == 0 else 'failed', "
        f"'run_mode': {args.run_mode!r}, 'name': {args.name!r}, "
        "'exit_code': status, "
        "'updated_at': datetime.now().astimezone().isoformat(timespec='seconds')"
        "}, indent=2), encoding='utf-8')"
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set +e",
            f"mkdir -p {shlex.quote(output_root)}",
            "python3 -c " + shlex.quote(running_writer),
            f"bash -lc {shlex.quote(remote_command)}",
            "status=$?",
            'export IBMOTRON_JOB_STATUS="$status"',
            "python3 -c " + shlex.quote(complete_writer),
            "exit \"$status\"",
        ]
    )


def _launch_detached_remote_job(
    ctl: RunpodCtl,
    ssh_info: dict[str, object],
    *,
    args: argparse.Namespace,
    remote_command: str,
) -> dict[str, object]:
    remote_paths = _remote_output_paths(args.remote_output)
    script_text = _build_remote_job_script(args, remote_command)
    write_script_command = " && ".join(
        [
            f"mkdir -p {shlex.quote(remote_paths['root'])}",
            "python3 -c "
            + shlex.quote(
                (
                    "from pathlib import Path; "
                    f"Path({remote_paths['script']!r}).write_text({script_text!r}, encoding='utf-8')"
                )
            ),
            f"chmod +x {shlex.quote(remote_paths['script'])}",
            (
                f"nohup bash {shlex.quote(remote_paths['script'])} "
                f"> {shlex.quote(remote_paths['log'])} 2>&1 < /dev/null & echo $!"
            ),
        ]
    )
    proc = ctl.ssh(ssh_info, write_script_command, check=False)
    job_pid = (proc.stdout or "").strip().splitlines()[-1] if (proc.stdout or "").strip() else ""
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "remote_pid": job_pid,
        "remote_paths": remote_paths,
    }


def _initial_delete_pod_on_exit(args: argparse.Namespace) -> bool:
    """Decide whether a freshly-created pod should be torn down in the finally block.

    The decision is pure args-based and made ONCE before any work happens, so that
    any subsequent exception, SystemExit, SSH drop, sync error, or finalize error
    still takes the pod down through the finally block. The prior logic only flipped
    the flag at the very end of the happy path, which leaked pods on every non-zero
    ssh returncode — the failure mode we hit in 20260408.

    Returns False for modes where we want the pod to persist beyond this process:
    - ``--pod-id``: we're reusing an existing pod, its lifecycle is not ours
    - ``--keep-pod``: caller explicitly asked us to leave it running
    - ``--detach-remote``: a watcher subprocess will handle collection and deletion
    """
    return not args.keep_pod and not args.pod_id and not args.detach_remote


def _remote_job_status(ctl: RunpodCtl, ssh_info: dict[str, object], *, remote_output: str) -> dict[str, object]:
    remote_paths = _remote_output_paths(remote_output)
    remote_script = f"""
import json
import subprocess
from pathlib import Path

state_path = Path({remote_paths["state"]!r})
root = Path({remote_paths["root"]!r})
payload = {{
    "state_exists": state_path.exists(),
    "output_exists": root.exists(),
}}
if state_path.exists():
    try:
        payload["state"] = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        payload["state_error"] = str(exc)
ps_text = subprocess.run(
    ["bash", "-lc", "ps -eo pid,command | egrep 'python -m ibm650_it.cli (train-eval|overfit-sanity|run-inference)' | grep -v grep || true"],
    text=True,
    capture_output=True,
    check=False,
).stdout.strip()
payload["active_process"] = bool(ps_text)
payload["process"] = ps_text
print(json.dumps(payload))
"""
    proc = ctl.ssh(ssh_info, f"python3 - <<'PY'\n{remote_script}\nPY", check=False)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or proc.stdout.strip() or f"ssh failed: {proc.returncode}"}
    return json.loads(proc.stdout)


def _finalize_local_output(args: argparse.Namespace, local_output_root: Path) -> dict[str, object]:
    subprocess.run(["./scripts/build_simh.sh"], cwd=REPO_ROOT, check=True)
    requested_modes = list(getattr(args, "modes", []) or [])
    if args.run_mode == "overfit-sanity":
        return finalize_overfit_output(
            dataset_index=REPO_ROOT / args.dataset_index,
            output_root=local_output_root,
            failure_archive_limit=args.failure_archive_limit,
            repo_root=REPO_ROOT,
            step_budget=args.step_budget,
            timeout_seconds=args.timeout_seconds,
        )
    if args.run_mode == "inference-only":
        return finalize_train_eval_output(
            dataset_root=REPO_ROOT / f"artifacts/datasets/{args.dataset_name}",
            output_root=local_output_root,
            eval_split=args.eval_split,
            modes=[args.inference_mode],
            failure_archive_limit=args.failure_archive_limit,
            repo_root=REPO_ROOT,
            step_budget=args.step_budget,
            timeout_seconds=args.timeout_seconds,
        )
    return finalize_train_eval_output(
        dataset_root=REPO_ROOT / f"artifacts/datasets/{args.dataset_name}",
        output_root=local_output_root,
        eval_split=args.eval_split,
        modes=requested_modes or None,
        failure_archive_limit=args.failure_archive_limit,
        repo_root=REPO_ROOT,
        step_budget=args.step_budget,
        timeout_seconds=args.timeout_seconds,
    )


def _hydrate_args_from_metadata(args: argparse.Namespace, *, force: bool = False) -> None:
    local_output_root = REPO_ROOT / args.local_output
    metadata = _load_job_metadata(local_output_root)
    if metadata is None:
        return
    for field in [
        "pod_id",
        "run_mode",
        "dataset_name",
        "dataset_index",
        "reference_index",
        "eval_split",
        "inference_mode",
        "modes",
        "remote_output",
        "band_repeat",
        "band_repeat_preset",
    ]:
        if force or getattr(args, field, None) in (None, False):
            value = metadata.get(field)
            if value not in (None, ""):
                setattr(args, field, value)


def _run_collect_watcher(args: argparse.Namespace) -> int:
    _hydrate_args_from_metadata(args, force=True)
    local_output_root = REPO_ROOT / args.local_output
    while True:
        metadata = _load_job_metadata(local_output_root) or {}
        if (local_output_root / "summary.json").exists():
            metadata["status"] = "complete"
            metadata["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            _write_job_metadata(local_output_root, metadata)
            return 0
        try:
            ctl = RunpodCtl(repo_root=REPO_ROOT)
            pod_id = str(metadata.get("pod_id") or args.pod_id or "")
            if not pod_id:
                print("watch-collect: missing pod_id in metadata", file=sys.stderr)
                return 2
            ssh_info = ctl.wait_for_ssh(pod_id, timeout_seconds=60, poll_seconds=5)
            remote_status = _remote_job_status(ctl, ssh_info, remote_output=str(metadata.get("remote_output") or args.remote_output))
            if remote_status.get("error"):
                print(f"watch-collect: remote status error: {remote_status['error']}", flush=True)
                time.sleep(args.auto_collect_poll_seconds)
                continue
            state = remote_status.get("state") or {}
            if remote_status.get("active_process") or state.get("status") not in {"complete", "failed"}:
                print(
                    json.dumps(
                        {
                            "watch": "waiting",
                            "pod_id": pod_id,
                            "state": state,
                            "active_process": remote_status.get("active_process"),
                            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                        }
                    ),
                    flush=True,
                )
                time.sleep(args.auto_collect_poll_seconds)
                continue
            print(
                json.dumps(
                    {
                        "watch": "collecting",
                        "pod_id": pod_id,
                        "state": state,
                        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    }
                ),
                flush=True,
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--collect-only",
                    "--local-output",
                    args.local_output,
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            if proc.returncode == 0:
                metadata = _load_job_metadata(local_output_root) or metadata
                metadata["status"] = "complete"
                metadata["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                _write_job_metadata(local_output_root, metadata)
                return 0
            print(f"watch-collect: collect-only failed with rc={proc.returncode}", flush=True)
        except Exception as exc:
            print(f"watch-collect: exception: {exc}", flush=True)
        time.sleep(args.auto_collect_poll_seconds)


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
    parser.add_argument("--model-name", default=DEFAULT_TRANSFORMERS_QLORA_MODEL)
    parser.add_argument("--qlora-bits", type=int, default=REAL_BASELINE_QLORA_BITS)
    parser.add_argument("--learning-rate", type=float, default=REAL_BASELINE_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=REAL_BASELINE_EPOCHS)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--few-shot-k", type=int, default=4)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["zero_shot", "few_shot", "fine_tuned"],
        default=["zero_shot", "few_shot", "fine_tuned"],
    )
    parser.add_argument("--band-repeat", action="append", default=[])
    parser.add_argument("--band-repeat-preset")
    parser.add_argument("--train-split", default="synthetic_train.jsonl")
    parser.add_argument("--eval-split", default="synthetic_dev.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--inference-mode", choices=["zero_shot", "few_shot", "fine_tuned"], default="fine_tuned")
    parser.add_argument("--model-path")
    parser.add_argument("--max-new-tokens", type=int, default=REAL_BASELINE_MAX_NEW_TOKENS)
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help="Batch size passed to the remote run-inference / train-eval command. "
             "1 is the historical serial default; set higher to batch model.generate() "
             "calls and cut fine_tuned eval wall time.",
    )
    parser.add_argument("--failure-archive-limit", type=int, default=25)
    parser.add_argument("--step-budget", default="50M")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--remote-output", default="artifacts/eval_reports/runpod_train_eval")
    parser.add_argument("--local-output", default="artifacts/eval_reports/runpod_train_eval")
    parser.add_argument("--example-count", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--detach-remote", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--reuse-pod-workspace", action="store_true")
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--watch-collect", action="store_true")
    parser.add_argument("--auto-collect-poll-seconds", type=int, default=60)
    args = parser.parse_args()

    if args.watch_collect:
        raise SystemExit(_run_collect_watcher(args))

    if args.prepare_only and not (args.keep_pod or args.pod_id):
        raise SystemExit("--prepare-only requires --keep-pod or --pod-id so the prepared pod is retained")
    if args.prepare_only and (args.detach_remote or args.collect_only):
        raise SystemExit("--prepare-only cannot be combined with --detach-remote or --collect-only")
    if args.detach_remote and args.collect_only:
        raise SystemExit("--detach-remote and --collect-only are mutually exclusive")
    if args.reuse_pod_workspace and not args.pod_id:
        raise SystemExit("--reuse-pod-workspace requires --pod-id")
    if args.collect_only:
        _hydrate_args_from_metadata(args, force=True)
        if not args.pod_id:
            raise SystemExit("--collect-only requires --pod-id or a local metadata file with pod_id")

    ctl = RunpodCtl(repo_root=REPO_ROOT)
    ctl.ensure_ssh_key()

    pod = None
    if not args.collect_only:
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
    else:
        pod = ctl.get_pod(args.pod_id)
    pod_id = pod["id"]
    # Set cleanup policy BEFORE anything can fail. The finally block will delete the
    # pod unless the user opted out via --keep-pod/--pod-id/--detach-remote. The
    # collect-only path overrides this after a successful collect.
    delete_pod_on_exit = _initial_delete_pod_on_exit(args)
    try:
        ssh_info = ctl.wait_for_ssh(pod_id, timeout_seconds=600, poll_seconds=10)
        local_output_root = REPO_ROOT / args.local_output
        if args.collect_only:
            remote_status = _remote_job_status(ctl, ssh_info, remote_output=args.remote_output)
            state = remote_status.get("state") if isinstance(remote_status, dict) else None
            if remote_status.get("error"):
                raise SystemExit(remote_status["error"])
            if remote_status.get("active_process"):
                raise SystemExit(f"remote job on pod {pod_id} is still running; collect later")
            if state is not None and state.get("status") not in {"complete", "failed"}:
                raise SystemExit(f"remote job state is not terminal: {state}")
            remote_output_path = f"/workspace/ibmotron/{args.remote_output}"
            exists = ctl.ssh(ssh_info, f"test -e {remote_output_path}", check=False)
            if exists.returncode != 0:
                raise SystemExit(f"remote output missing: {remote_output_path}")
            _sync_remote_output(ctl, ssh_info, remote_output_path, local_output_root)
            finalized = _finalize_local_output(args, local_output_root)
            metadata = _job_metadata_payload(args, pod_id=pod_id, detached=True, status="complete")
            metadata["remote_state"] = state
            metadata["local_summary"] = str(local_output_root / "summary.json")
            _write_job_metadata(local_output_root, metadata)
            delete_pod_on_exit = not args.keep_pod
            print(
                json.dumps(
                    {
                        "pod_id": pod_id,
                        "local_output": str(local_output_root),
                        "remote_state": state,
                        "finalized": finalized is not None,
                    },
                    indent=2,
                )
            )
            return
        if args.prepare_only or not args.reuse_pod_workspace:
            base_archive = build_base_archive()
            ctl.scp_to(ssh_info, base_archive, "/workspace/ibmotron-base.tgz")
        if not args.collect_only:
            dataset_archive = build_dataset_archive(args)
            ctl.scp_to(ssh_info, dataset_archive, "/workspace/ibmotron-dataset.tgz")
        if args.reuse_pod_workspace:
            ready = ctl.ssh(ssh_info, f"test -f {READY_MARKER}", check=False)
            if ready.returncode != 0:
                raise SystemExit(f"remote workspace on pod {pod_id} is not prepared; missing {READY_MARKER}")
        remote_command = remote_prepare_command(args.model_name) if args.prepare_only else remote_train_command(
            args,
            args.remote_output,
            reuse_workspace=args.reuse_pod_workspace,
        )
        logs_dir = REPO_ROOT / "artifacts" / "logs" / "runpod"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_stem = f"{pod_id}.{args.name}"
        if args.detach_remote:
            launch = _launch_detached_remote_job(ctl, ssh_info, args=args, remote_command=remote_command)
            (logs_dir / f"{log_stem}.stdout.log").write_text(str(launch.get("stdout", "")), encoding="utf-8")
            (logs_dir / f"{log_stem}.stderr.log").write_text(str(launch.get("stderr", "")), encoding="utf-8")
            metadata = _job_metadata_payload(args, pod_id=pod_id, detached=True, status="launched")
            metadata["remote_pid"] = launch.get("remote_pid")
            metadata["remote_paths"] = launch.get("remote_paths")
            watcher_pid = _spawn_collect_watcher(args, local_output_root)
            metadata["watcher_pid"] = watcher_pid
            _write_job_metadata(local_output_root, metadata)
            if launch["returncode"] != 0:
                raise SystemExit(launch["returncode"])
            print(
                json.dumps(
                    {
                        "pod_id": pod_id,
                        "ssh": ssh_info,
                        "remote_pid": launch.get("remote_pid"),
                        "watcher_pid": watcher_pid,
                        "remote_paths": launch.get("remote_paths"),
                        "local_output": str(local_output_root),
                        "metadata_path": str(_metadata_path(local_output_root)),
                    },
                    indent=2,
                )
            )
            return
        proc = ctl.ssh(ssh_info, remote_command, check=False)
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
        _sync_remote_output(ctl, ssh_info, remote_output_path, local_output_root)
        _finalize_local_output(args, local_output_root)
        summary = {
            "pod_id": pod_id,
            "pod": pod,
            "ssh": ssh_info,
            "returncode": proc.returncode,
            "local_output": str(local_output_root),
        }
        print(json.dumps(summary, indent=2))
        if proc.returncode != 0:
            # The sync + finalize ran successfully above even though ssh returned
            # non-zero (e.g. the SSH session dropped mid-run but the remote job kept
            # going and we synced its output on a later pass). Still surface the
            # ssh returncode to the caller so CI sees the failure — delete_pod_on_exit
            # was set at the top of the try block, so the finally block below will
            # take the pod down regardless of this SystemExit.
            raise SystemExit(proc.returncode)
    finally:
        if delete_pod_on_exit:
            try:
                ctl.delete_pod(pod_id)
            except Exception:
                pass


if __name__ == "__main__":
    main()
