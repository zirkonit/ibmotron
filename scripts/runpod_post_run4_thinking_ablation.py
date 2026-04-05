from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibm650_it import REPO_ROOT
from ibm650_it.cloud import RunpodCtl
from ibm650_it.training.thinking_ablation import finalize_thinking_ablation_output


POD_ID = "kd3ufb9t40jpai"
RUN3_ROOT = REPO_ROOT / "artifacts/eval_reports/sweeps/subset_128_20_a40_20260405_1137/e5_lr0p0001"
RUN4_ROOT = REPO_ROOT / "artifacts/eval_reports/sweeps/subset_128_20_a40_20260405_1137/e5_lr0p0002"
REFERENCE_INDEX = REPO_ROOT / "artifacts/datasets/pilot_remote_128_20/splits/synthetic_dev.jsonl"
LOCAL_OUTPUT = REPO_ROOT / "artifacts/eval_reports/thinking_ablation/subset_128_20_best_after_run4"
REMOTE_OUTPUT = "artifacts/eval_reports/thinking_ablation/subset_128_20_best_after_run4"
LOG_PATH = REPO_ROOT / "artifacts/logs/runpod/post_run4_thinking_ablation.log"


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S %z')}] {message}\n"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(stamped)
    print(stamped, end="")
    sys.stdout.flush()


def wait_for_report(run_root: Path, *, timeout_seconds: int = 4 * 60 * 60) -> Path:
    report = run_root / "reports" / "fine_tuned.json"
    started = time.time()
    while not report.exists():
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"timed out waiting for {report}")
        time.sleep(20)
    return report


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score_report(report: dict) -> tuple[float, float, float, float]:
    return (
        float(report["functional_equivalence"]),
        float(report["exact_match"]),
        float(report["per_card_exact"]),
        float(report["assemblability"]),
    )


def pick_best_run() -> tuple[str, Path]:
    run3_report = load_report(wait_for_report(RUN3_ROOT))
    run4_report = load_report(wait_for_report(RUN4_ROOT))
    candidates = [
        ("run3_e5_lr1e-4", RUN3_ROOT, run3_report),
        ("run4_e5_lr2e-4", RUN4_ROOT, run4_report),
    ]
    best_name, best_root, _ = max(candidates, key=lambda item: score_report(item[2]))
    return best_name, best_root


def remote_ablation_command(remote_model_dir: str) -> str:
    return " && ".join(
        [
            "cd /workspace/ibmotron",
            ". .venv/bin/activate",
            "export HF_HOME=/workspace/.cache/huggingface",
            "mkdir -p \"$HF_HOME\"",
            (
                "python -m ibm650_it.cli thinking-ablation "
                f"--reference-index {REFERENCE_INDEX.relative_to(REPO_ROOT)} "
                f"--model {remote_model_dir} "
                f"--output {REMOTE_OUTPUT} "
                "--limit 20 "
                "--max-new-tokens 1024 "
                "--eval-mode skip "
                "--failure-archive-limit 25 "
                "--timeout-seconds 30"
            ),
        ]
    )


def main() -> None:
    log("Waiting for run 3 and run 4 fine_tuned reports.")
    best_name, best_root = pick_best_run()
    log(f"Selected best fine-tuned model: {best_name} at {best_root}.")

    ctl = RunpodCtl(repo_root=REPO_ROOT)
    ctl.ensure_ssh_key()
    ssh_info = ctl.wait_for_ssh(POD_ID, timeout_seconds=600, poll_seconds=10)
    ready = ctl.ssh(ssh_info, "test -f /workspace/ibmotron/.ibmotron_ready.json", check=False)
    if ready.returncode != 0:
        raise SystemExit(f"warm pod {POD_ID} is not prepared")

    remote_model_dir = str(best_root.relative_to(REPO_ROOT) / "model")
    log(f"Launching remote thinking ablation on pod {POD_ID} with model {remote_model_dir}.")
    proc = ctl.ssh(ssh_info, remote_ablation_command(remote_model_dir), check=False)
    log_stem = REPO_ROOT / "artifacts/logs/runpod/post_run4_thinking_ablation"
    log_stem.parent.mkdir(parents=True, exist_ok=True)
    (log_stem.with_suffix(".stdout.log")).write_text(proc.stdout, encoding="utf-8")
    (log_stem.with_suffix(".stderr.log")).write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    remote_output_path = f"/workspace/ibmotron/{REMOTE_OUTPUT}"
    exists = ctl.ssh(ssh_info, f"test -e {remote_output_path}", check=False)
    if exists.returncode != 0:
        raise SystemExit(f"remote thinking ablation output missing: {remote_output_path}")

    ctl.scp_from(ssh_info, remote_output_path, LOCAL_OUTPUT)
    subprocess.run(["./scripts/build_simh.sh"], cwd=REPO_ROOT, check=True)
    summary = finalize_thinking_ablation_output(
        reference_index=REFERENCE_INDEX,
        output_root=LOCAL_OUTPUT,
        model_dir=best_root / "model",
        repo_root=REPO_ROOT,
        failure_archive_limit=25,
        step_budget="50M",
        timeout_seconds=30,
    )
    summary["selected_run"] = {
        "name": best_name,
        "path": str(best_root),
    }
    (LOCAL_OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Thinking ablation complete. Output: {LOCAL_OUTPUT}")


if __name__ == "__main__":
    main()
