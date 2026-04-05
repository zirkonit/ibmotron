from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class SweepRun:
    epochs: int
    learning_rate: float
    pod_name: str
    output_path: str


def lr_slug(value: float) -> str:
    text = format(value, "g")
    return text.replace(".", "p").replace("-", "m")


def build_sweep_runs(
    *,
    sweep_name: str,
    output_root: str,
    epochs: list[int],
    learning_rates: list[float],
) -> list[SweepRun]:
    runs: list[SweepRun] = []
    for epoch_count in epochs:
        for learning_rate in learning_rates:
            run_slug = f"e{epoch_count}_lr{lr_slug(learning_rate)}"
            runs.append(
                SweepRun(
                    epochs=epoch_count,
                    learning_rate=learning_rate,
                    pod_name=f"{sweep_name}-{run_slug}",
                    output_path=f"{output_root}/{run_slug}",
                )
            )
    return runs


def extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    evaluations = summary.get("evaluations", {})
    extracted: dict[str, Any] = {}
    for mode in ("zero_shot", "few_shot", "fine_tuned"):
        report = evaluations.get(mode, {}).get("report")
        if report is None:
            continue
        extracted[mode] = {
            "exact_match": report.get("exact_match"),
            "assemblability": report.get("assemblability"),
            "functional_equivalence": report.get("functional_equivalence"),
            "per_card_exact": report.get("per_card_exact"),
            "normalized_edit_distance": report.get("normalized_edit_distance"),
            "failure_taxonomy": report.get("failure_taxonomy", {}),
        }
    return extracted


def evaluate_gate(summary: dict[str, Any]) -> dict[str, Any]:
    evaluations = summary.get("evaluations", {})
    few = evaluations.get("few_shot", {}).get("report", {})
    fine = evaluations.get("fine_tuned", {}).get("report", {})
    failure_taxonomy = fine.get("failure_taxonomy", {})
    dominant_failure = None
    if failure_taxonomy:
        dominant_failure = max(failure_taxonomy.items(), key=lambda item: item[1])[0]
    return {
        "fine_tuned_beats_few_shot_exact": (fine.get("exact_match") or 0.0) > (few.get("exact_match") or 0.0),
        "fine_tuned_beats_few_shot_assemblability": (fine.get("assemblability") or 0.0) > (few.get("assemblability") or 0.0),
        "returned_it_source_not_dominant": dominant_failure != "returned_it_source_instead_of_pit",
        "dominant_failure": dominant_failure,
        "passes_plan_gate": (
            (fine.get("exact_match") or 0.0) > (few.get("exact_match") or 0.0)
            and (fine.get("assemblability") or 0.0) > (few.get("assemblability") or 0.0)
            and dominant_failure != "returned_it_source_instead_of_pit"
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _load_summary(summary_path: Path) -> dict[str, Any]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", default=f"subset_128_20_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--run-mode", choices=["train-eval"], default="train-eval")
    parser.add_argument("--dataset-name", default="pilot_remote_128_20")
    parser.add_argument("--backend", default="transformers_qlora")
    parser.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    parser.add_argument("--gpu-id", default="NVIDIA A40")
    parser.add_argument("--cloud-type", default="SECURE")
    parser.add_argument("--image", default="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")
    parser.add_argument("--qlora-bits", type=int, default=0)
    parser.add_argument("--epochs", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 5e-5])
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--few-shot-k", type=int, default=4)
    parser.add_argument("--eval-split", default="synthetic_dev.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--failure-archive-limit", type=int, default=25)
    parser.add_argument("--step-budget", default="50M")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--output-root", default="artifacts/eval_reports/sweeps")
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sweep_root = Path(args.output_root) / args.sweep_name
    manifest_path = sweep_root / "manifest.json"
    runs = build_sweep_runs(
        sweep_name=args.sweep_name,
        output_root=str(sweep_root),
        epochs=args.epochs,
        learning_rates=args.learning_rates,
    )
    manifest: dict[str, Any] = {
        "sweep_name": args.sweep_name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_name": args.dataset_name,
        "grid": {
            "epochs": args.epochs,
            "learning_rates": args.learning_rates,
        },
        "runs": [],
    }
    _write_json(manifest_path, manifest)

    for sweep_run in runs:
        summary_path = REPO_ROOT / sweep_run.output_path / "summary.json"
        run_record: dict[str, Any] = {
            **asdict(sweep_run),
            "summary_path": str(summary_path),
        }
        if args.resume and summary_path.exists():
            summary = _load_summary(summary_path)
            run_record["status"] = "skipped_existing"
            run_record["metrics"] = extract_metrics(summary)
            run_record["gate"] = evaluate_gate(summary)
            manifest["runs"].append(run_record)
            _write_json(manifest_path, manifest)
            continue

        command = [
            sys.executable,
            "scripts/runpod_train_eval.py",
            "--name",
            sweep_run.pod_name,
            "--run-mode",
            args.run_mode,
            "--dataset-name",
            args.dataset_name,
            "--backend",
            args.backend,
            "--model-name",
            args.model_name,
            "--gpu-id",
            args.gpu_id,
            "--cloud-type",
            args.cloud_type,
            "--image",
            args.image,
            "--qlora-bits",
            str(args.qlora_bits),
            "--learning-rate",
            str(sweep_run.learning_rate),
            "--epochs",
            str(sweep_run.epochs),
            "--max-seq-length",
            str(args.max_seq_length),
            "--per-device-train-batch-size",
            str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--few-shot-k",
            str(args.few_shot_k),
            "--eval-split",
            args.eval_split,
            "--limit",
            str(args.limit),
            "--max-examples",
            str(args.max_examples),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--failure-archive-limit",
            str(args.failure_archive_limit),
            "--step-budget",
            args.step_budget,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-output",
            sweep_run.output_path,
            "--local-output",
            sweep_run.output_path,
        ]
        if args.keep_pod:
            command.append("--keep-pod")
        run_record["command"] = command
        if args.dry_run:
            run_record["status"] = "dry_run"
            manifest["runs"].append(run_record)
            _write_json(manifest_path, manifest)
            continue

        proc = _run_command(command)
        run_record["returncode"] = proc.returncode
        run_record["stdout_tail"] = proc.stdout.splitlines()[-20:]
        run_record["stderr_tail"] = proc.stderr.splitlines()[-20:]
        if summary_path.exists():
            summary = _load_summary(summary_path)
            run_record["status"] = "ok" if proc.returncode == 0 else "completed_with_nonzero_returncode"
            run_record["metrics"] = extract_metrics(summary)
            run_record["gate"] = evaluate_gate(summary)
        else:
            run_record["status"] = "failed_no_summary"
        manifest["runs"].append(run_record)
        _write_json(manifest_path, manifest)

        if proc.returncode != 0 and not args.continue_on_error:
            raise SystemExit(proc.returncode)

    passed = sum(1 for run in manifest["runs"] if run.get("gate", {}).get("passes_plan_gate"))
    manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["passing_runs"] = passed
    manifest["total_runs"] = len(manifest["runs"])
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
