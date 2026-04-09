from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibm650_it.cloud import RunpodCtl


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "eval_reports"
    / "sweeps"
    / "subset_128_20_a40_20260405_1137"
    / "e5_lr0p0002"
    / "summary.json"
)
MIN_SELECTION_ASSEMBLABILITY = 0.95


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


def _fine_tuned_report(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("evaluations", {}).get("fine_tuned", {}).get("report", {})


def _dominant_failure(report: dict[str, Any]) -> str | None:
    failure_taxonomy = report.get("failure_taxonomy", {})
    if not failure_taxonomy:
        return None
    return max(failure_taxonomy.items(), key=lambda item: item[1])[0]


def selection_score(summary: dict[str, Any]) -> dict[str, Any]:
    fine = _fine_tuned_report(summary)
    assemblability = float(fine.get("assemblability", 0.0) or 0.0)
    eligible = assemblability >= MIN_SELECTION_ASSEMBLABILITY
    return {
        "eligible": eligible,
        "assemblability_threshold": MIN_SELECTION_ASSEMBLABILITY,
        "score": (
            float(fine.get("functional_equivalence", 0.0) or 0.0),
            float(fine.get("exact_match", 0.0) or 0.0),
            float(fine.get("per_card_exact", 0.0) or 0.0),
        )
        if eligible
        else None,
    }


def compare_selection_scores(left: dict[str, Any] | None, right: dict[str, Any] | None) -> int:
    left_score = left.get("score") if left else None
    right_score = right.get("score") if right else None
    if left_score == right_score:
        return 0
    if left_score is None:
        return -1
    if right_score is None:
        return 1
    return 1 if tuple(left_score) > tuple(right_score) else -1


def evaluate_gate_a(summary: dict[str, Any]) -> dict[str, Any]:
    fine = _fine_tuned_report(summary)
    by_band = fine.get("by_band", {})
    b1 = by_band.get("B1", {})
    return {
        "overall_functional_at_least_70": float(fine.get("functional_equivalence", 0.0) or 0.0) >= 0.70,
        "overall_exact_at_least_50": float(fine.get("exact_match", 0.0) or 0.0) >= 0.50,
        "b1_functional_at_least_40": float(b1.get("functional_equivalence", 0.0) or 0.0) >= 0.40,
        "passes_gate_a": (
            float(fine.get("functional_equivalence", 0.0) or 0.0) >= 0.70
            and float(fine.get("exact_match", 0.0) or 0.0) >= 0.50
            and float(b1.get("functional_equivalence", 0.0) or 0.0) >= 0.40
        ),
    }


def compare_to_baseline(summary: dict[str, Any], baseline_summary: dict[str, Any]) -> dict[str, Any]:
    candidate = selection_score(summary)
    baseline = selection_score(baseline_summary)
    comparison = compare_selection_scores(candidate, baseline)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "beats_baseline": comparison > 0,
        "matches_baseline": comparison == 0,
        "loses_to_baseline": comparison < 0,
    }


def evaluate_gate(summary: dict[str, Any]) -> dict[str, Any]:
    evaluations = summary.get("evaluations", {})
    few = evaluations.get("few_shot", {}).get("report", {})
    fine = _fine_tuned_report(summary)
    dominant_failure = _dominant_failure(fine)
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


def _load_json_output(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


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
    parser.add_argument("--epochs", nargs="+", type=int, default=[5])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[3e-4, 4e-4])
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--few-shot-k", type=int, default=4)
    parser.add_argument("--train-split", default="synthetic_train.jsonl")
    parser.add_argument("--eval-split", default="synthetic_dev.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--failure-archive-limit", type=int, default=25)
    parser.add_argument("--step-budget", default="50M")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--output-root", default="artifacts/eval_reports/sweeps")
    parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY) if DEFAULT_BASELINE_SUMMARY.exists() else None)
    parser.add_argument("--reuse-single-pod", action="store_true")
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
        "reuse_single_pod": args.reuse_single_pod,
        "grid": {
            "epochs": args.epochs,
            "learning_rates": args.learning_rates,
        },
        "runs": [],
    }
    baseline_summary: dict[str, Any] | None = None
    if args.baseline_summary:
        baseline_path = Path(args.baseline_summary)
        manifest["baseline_summary"] = str(baseline_path)
        if baseline_path.exists():
            baseline_summary = _load_summary(baseline_path)
            manifest["baseline_metrics"] = extract_metrics(baseline_summary)
            manifest["baseline_selection"] = selection_score(baseline_summary)
            manifest["baseline_gate_a"] = evaluate_gate_a(baseline_summary)
    _write_json(manifest_path, manifest)

    warm_pod_id: str | None = None
    if args.reuse_single_pod:
        prepare_name = f"{args.sweep_name}-warm"
        prepare_command = [
            sys.executable,
            "scripts/runpod_train_eval.py",
            "--name",
            prepare_name,
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
            str(args.learning_rates[0]),
            "--epochs",
            str(args.epochs[0]),
            "--max-seq-length",
            str(args.max_seq_length),
            "--per-device-train-batch-size",
            str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--few-shot-k",
            str(args.few_shot_k),
            "--train-split",
            args.train_split,
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
            "--prepare-only",
            "--keep-pod",
        ]
        if args.dry_run:
            manifest["warm_pod"] = {"status": "dry_run", "command": prepare_command}
        else:
            proc = _run_command(prepare_command)
            parsed = _load_json_output(proc.stdout)
            manifest["warm_pod"] = {
                "status": "ok" if proc.returncode == 0 else "failed",
                "command": prepare_command,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout.splitlines()[-20:],
                "stderr_tail": proc.stderr.splitlines()[-20:],
            }
            if parsed is not None:
                manifest["warm_pod"].update(parsed)
                warm_pod_id = parsed.get("pod_id")
            _write_json(manifest_path, manifest)
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)
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
            run_record["gate_a"] = evaluate_gate_a(summary)
            run_record["selection"] = selection_score(summary)
            if baseline_summary is not None:
                run_record["vs_baseline"] = compare_to_baseline(summary, baseline_summary)
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
            "--train-split",
            args.train_split,
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
        if args.reuse_single_pod and warm_pod_id is not None:
            command.extend(["--pod-id", warm_pod_id, "--reuse-pod-workspace"])
        run_record["command"] = command
        if args.dry_run:
            run_record["status"] = "dry_run"
            manifest["runs"].append(run_record)
            _write_json(manifest_path, manifest)
            continue

        proc = _run_command(command)
        parsed = _load_json_output(proc.stdout)
        run_record["returncode"] = proc.returncode
        run_record["stdout_tail"] = proc.stdout.splitlines()[-20:]
        run_record["stderr_tail"] = proc.stderr.splitlines()[-20:]
        if parsed is not None:
            run_record["summary"] = parsed
            if parsed.get("pod_id"):
                run_record["pod_id"] = parsed["pod_id"]
        if summary_path.exists():
            summary = _load_summary(summary_path)
            run_record["status"] = "ok" if proc.returncode == 0 else "completed_with_nonzero_returncode"
            run_record["metrics"] = extract_metrics(summary)
            run_record["gate"] = evaluate_gate(summary)
            run_record["gate_a"] = evaluate_gate_a(summary)
            run_record["selection"] = selection_score(summary)
            if baseline_summary is not None:
                run_record["vs_baseline"] = compare_to_baseline(summary, baseline_summary)
        else:
            run_record["status"] = "failed_no_summary"
        manifest["runs"].append(run_record)
        _write_json(manifest_path, manifest)

        if proc.returncode != 0 and not args.continue_on_error:
            raise SystemExit(proc.returncode)

    passed = sum(1 for run in manifest["runs"] if run.get("gate", {}).get("passes_plan_gate"))
    completed_runs = [run for run in manifest["runs"] if run.get("selection", {}).get("eligible")]
    best_candidate = None
    for run in completed_runs:
        if best_candidate is None or compare_selection_scores(run.get("selection"), best_candidate.get("selection")) > 0:
            best_candidate = run

    manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["passing_runs"] = passed
    manifest["total_runs"] = len(manifest["runs"])
    if best_candidate is not None:
        manifest["best_candidate"] = {
            "pod_name": best_candidate["pod_name"],
            "output_path": best_candidate["output_path"],
            "selection": best_candidate["selection"],
            "gate_a": best_candidate.get("gate_a"),
        }
    if baseline_summary is not None:
        baseline_entry = {
            "source": "baseline",
            "summary_path": str(Path(args.baseline_summary)),
            "selection": selection_score(baseline_summary),
            "gate_a": evaluate_gate_a(baseline_summary),
        }
        if best_candidate is None or compare_selection_scores(best_candidate.get("selection"), baseline_entry["selection"]) <= 0:
            manifest["winner"] = baseline_entry
            manifest["lr_escalation_recommendation"] = "freeze_2e-4_baseline"
        else:
            manifest["winner"] = {
                "source": "sweep_run",
                "pod_name": best_candidate["pod_name"],
                "output_path": best_candidate["output_path"],
                "selection": best_candidate["selection"],
                "gate_a": best_candidate.get("gate_a"),
            }
            manifest["lr_escalation_recommendation"] = "confirm_new_winner"
    elif best_candidate is not None:
        manifest["winner"] = {
            "source": "sweep_run",
            "pod_name": best_candidate["pod_name"],
            "output_path": best_candidate["output_path"],
            "selection": best_candidate["selection"],
            "gate_a": best_candidate.get("gate_a"),
        }
    if args.reuse_single_pod and warm_pod_id and not args.keep_pod and not args.dry_run:
        ctl = RunpodCtl(repo_root=REPO_ROOT)
        manifest["warm_pod_cleanup"] = ctl.delete_pod(warm_pod_id)
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
