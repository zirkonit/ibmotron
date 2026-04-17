from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibm650_it import REPO_ROOT
from ibm650_it.cloud import RunpodCtl
from ibm650_it.dataset.io import load_jsonl, write_jsonl
from ibm650_it.dataset.sampling import stable_limit_records, stable_weighted_band_sample
from ibm650_it.training.hf_qlora import DEFAULT_TRANSFORMERS_QLORA_MODEL
from scripts.runpod_train_eval import READY_MARKER


DEFAULT_RUN_NAME = f"progressive_qwen_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
DEFAULT_SOURCE_DATASET = "stage_20k_repaired"
DEFAULT_EVAL_SPLIT_NAME = "progressive_eval.jsonl"
DEFAULT_FOCUS_WEIGHT = 4
DEFAULT_READY_TIMEOUT_SECONDS = 1800
DEFAULT_READY_POLL_SECONDS = 10


@dataclass(frozen=True, slots=True)
class ProgressiveStage:
    train_count: int
    eval_count: int
    focus_bands: tuple[str, ...]
    focus_weight: int = DEFAULT_FOCUS_WEIGHT

    @property
    def slug(self) -> str:
        focus_slug = "_".join(band.lower() for band in self.focus_bands)
        return f"s{self.train_count}_{focus_slug}_e{self.eval_count}"

    @property
    def focus_weights(self) -> dict[str, int]:
        return {band: self.focus_weight for band in self.focus_bands}


PROGRESSIVE_STAGES = [
    ProgressiveStage(train_count=100, eval_count=12, focus_bands=("B0",)),
    ProgressiveStage(train_count=300, eval_count=36, focus_bands=("B0", "B1")),
    ProgressiveStage(train_count=1000, eval_count=108, focus_bands=("B0", "B1", "B2")),
    ProgressiveStage(train_count=3000, eval_count=324, focus_bands=("B0", "B1", "B2", "B3")),
    ProgressiveStage(train_count=10000, eval_count=972, focus_bands=("B0", "B1", "B2", "B3", "B4")),
]


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _load_json_output(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _band_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[str(record.get("band", ""))] += 1
    return dict(sorted(counts.items()))


def _prediction_timing_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    fine_tuned = summary.get("evaluations", {}).get("fine_tuned", {})
    prediction_index = fine_tuned.get("prediction_index")
    if not prediction_index:
        return {}
    prediction_path = Path(str(prediction_index))
    if not prediction_path.exists():
        return {"prediction_index": str(prediction_path), "missing": True}

    generation_seconds: list[float] = []
    evaluation_seconds: list[float] = []
    total_seconds: list[float] = []
    for record in load_jsonl(prediction_path):
        timings = record.get("timings", {})
        generation = timings.get("generation_seconds")
        evaluation = timings.get("evaluation_seconds")
        total = timings.get("total_seconds")
        if generation is not None:
            generation_seconds.append(float(generation))
        if evaluation is not None:
            evaluation_seconds.append(float(evaluation))
        if total is not None:
            total_seconds.append(float(total))

    metrics: dict[str, Any] = {
        "prediction_index": str(prediction_path),
        "prediction_count": len(generation_seconds) or len(evaluation_seconds) or len(total_seconds),
    }
    if generation_seconds:
        metrics.update(
            {
                "generation_total_seconds": sum(generation_seconds),
                "generation_avg_seconds": sum(generation_seconds) / len(generation_seconds),
                "generation_median_seconds": median(generation_seconds),
            }
        )
    if evaluation_seconds:
        metrics.update(
            {
                "evaluation_total_seconds": sum(evaluation_seconds),
                "evaluation_avg_seconds": sum(evaluation_seconds) / len(evaluation_seconds),
                "evaluation_median_seconds": median(evaluation_seconds),
            }
        )
    if total_seconds:
        metrics.update(
            {
                "per_example_total_seconds": sum(total_seconds) / len(total_seconds),
                "per_example_total_median_seconds": median(total_seconds),
            }
        )
    return metrics


def _rebase_record_paths(record: dict[str, Any], *, source_root: Path) -> dict[str, Any]:
    def transform(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: transform(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [transform(inner) for inner in value]
        if isinstance(value, str):
            candidate = (source_root / value).resolve()
            if candidate.exists():
                return str(candidate.relative_to(REPO_ROOT))
        return value

    return transform(record)


def _prepare_stage_dataset(
    *,
    source_root: Path,
    output_root: Path,
    stage: ProgressiveStage,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> dict[str, Any]:
    weighted_train = stable_weighted_band_sample(
        train_records,
        stage.train_count,
        band_weights=stage.focus_weights,
        salt=f"progressive_qwen:{output_root}:train",
    )
    balanced_eval = stable_limit_records(
        eval_records,
        stage.eval_count,
        salt=f"progressive_qwen:{output_root}:eval",
    )
    rebased_train = [_rebase_record_paths(record, source_root=source_root) for record in weighted_train]
    rebased_eval = [_rebase_record_paths(record, source_root=source_root) for record in balanced_eval]

    splits_root = output_root / "splits"
    write_jsonl(splits_root / "synthetic_train.jsonl", rebased_train)
    write_jsonl(splits_root / DEFAULT_EVAL_SPLIT_NAME, rebased_eval)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "stage": asdict(stage),
        "train_count": len(rebased_train),
        "eval_count": len(rebased_eval),
        "train_band_counts": _band_counts(rebased_train),
        "eval_band_counts": _band_counts(rebased_eval),
        "eval_source_splits": ["synthetic_dev.jsonl", "synthetic_test.jsonl"],
        "focus_bands": list(stage.focus_bands),
        "focus_band_weights": stage.focus_weights,
        "train_split": "synthetic_train.jsonl",
        "eval_split": DEFAULT_EVAL_SPLIT_NAME,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _extract_stage_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    fine_tuned = summary.get("evaluations", {}).get("fine_tuned", {})
    report = fine_tuned.get("report", {})
    return {
        "count": report.get("count"),
        "exact_match": report.get("exact_match"),
        "assemblability": report.get("assemblability"),
        "functional_equivalence": report.get("functional_equivalence"),
        "failure_taxonomy": report.get("failure_taxonomy", {}),
        "by_band": {
            band: {
                "count": band_report.get("count"),
                "exact_match": band_report.get("exact_match"),
                "assemblability": band_report.get("assemblability"),
                "functional_equivalence": band_report.get("functional_equivalence"),
            }
            for band, band_report in sorted(report.get("by_band", {}).items())
        },
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _wait_for_ready_marker(
    *,
    ctl: RunpodCtl,
    pod_id: str,
    timeout_seconds: int = DEFAULT_READY_TIMEOUT_SECONDS,
    poll_seconds: int = DEFAULT_READY_POLL_SECONDS,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    ssh_info = ctl.wait_for_ssh(pod_id, timeout_seconds=timeout_seconds, poll_seconds=max(1, poll_seconds))
    while True:
        ready = ctl.ssh(ssh_info, f"test -f {READY_MARKER}", check=False)
        if ready.returncode == 0:
            return ssh_info
        if time.time() >= deadline:
            break
        time.sleep(max(0, poll_seconds))
        remaining = max(1, int(deadline - time.time()))
        ssh_info = ctl.wait_for_ssh(pod_id, timeout_seconds=remaining, poll_seconds=max(1, poll_seconds))
    raise TimeoutError(f"pod {pod_id} did not publish ready marker {READY_MARKER} within {timeout_seconds}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--source-dataset-name", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--model-name", default=DEFAULT_TRANSFORMERS_QLORA_MODEL)
    parser.add_argument("--gpu-id", default="NVIDIA A40")
    parser.add_argument("--cloud-type", default="SECURE")
    parser.add_argument("--image", default="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--qlora-bits", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--failure-archive-limit", type=int, default=25)
    parser.add_argument("--step-budget", default="50M")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--ready-timeout-seconds", type=int, default=DEFAULT_READY_TIMEOUT_SECONDS)
    parser.add_argument("--ready-poll-seconds", type=int, default=DEFAULT_READY_POLL_SECONDS)
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pod-id")
    args = parser.parse_args()

    source_root = REPO_ROOT / "artifacts" / "datasets" / args.source_dataset_name
    train_source = load_jsonl(source_root / "splits" / "synthetic_train.jsonl")
    eval_source = load_jsonl(source_root / "splits" / "synthetic_dev.jsonl") + load_jsonl(
        source_root / "splits" / "synthetic_test.jsonl"
    )

    dataset_bundle_root = REPO_ROOT / "artifacts" / "datasets" / args.run_name
    eval_bundle_root = REPO_ROOT / "artifacts" / "eval_reports" / args.run_name
    manifest_path = eval_bundle_root / "manifest.json"
    manifest: dict[str, Any] = {
        "run_name": args.run_name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_dataset_name": args.source_dataset_name,
        "source_dataset_root": str(source_root),
        "model_name": args.model_name,
        "stages": [],
        "runs": [],
    }

    stage_specs: list[dict[str, Any]] = []
    for stage in PROGRESSIVE_STAGES:
        dataset_root = dataset_bundle_root / stage.slug
        dataset_name = f"{args.run_name}/{stage.slug}"
        output_path = f"artifacts/eval_reports/{args.run_name}/{stage.slug}"
        dataset_summary = _prepare_stage_dataset(
            source_root=source_root,
            output_root=dataset_root,
            stage=stage,
            train_records=train_source,
            eval_records=eval_source,
        )
        stage_specs.append(
            {
                "stage": stage,
                "dataset_root": dataset_root,
                "dataset_name": dataset_name,
                "output_path": output_path,
                "summary_path": REPO_ROOT / output_path / "summary.json",
                "dataset_summary": dataset_summary,
            }
        )
        manifest["stages"].append(
            {
                "slug": stage.slug,
                "dataset_name": dataset_name,
                "dataset_root": str(dataset_root),
                "output_path": output_path,
                "dataset_summary": dataset_summary,
            }
        )
    _write_manifest(manifest_path, manifest)

    warm_pod_id = args.pod_id
    if warm_pod_id is None and not args.dry_run:
        first_stage = stage_specs[0]
        prepare_command = [
            sys.executable,
            "scripts/runpod_train_eval.py",
            "--name",
            f"{args.run_name}-warm",
            "--run-mode",
            "train-eval",
            "--dataset-name",
            str(first_stage["dataset_name"]),
            "--backend",
            "transformers_qlora",
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
            str(args.learning_rate),
            "--epochs",
            str(args.epochs),
            "--max-seq-length",
            str(args.max_seq_length),
            "--per-device-train-batch-size",
            str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--modes",
            "fine_tuned",
            "--train-split",
            "synthetic_train.jsonl",
            "--eval-split",
            DEFAULT_EVAL_SPLIT_NAME,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--inference-batch-size",
            str(args.inference_batch_size),
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
            _write_manifest(manifest_path, manifest)
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)

    if warm_pod_id and not args.dry_run:
        ctl = RunpodCtl(repo_root=REPO_ROOT)
        try:
            ssh_info = _wait_for_ready_marker(
                ctl=ctl,
                pod_id=str(warm_pod_id),
                timeout_seconds=args.ready_timeout_seconds,
                poll_seconds=args.ready_poll_seconds,
            )
        except TimeoutError as exc:
            manifest["warm_pod_ready_check"] = {
                "status": "failed",
                "pod_id": str(warm_pod_id),
                "ready_marker": READY_MARKER,
                "timeout_seconds": args.ready_timeout_seconds,
                "poll_seconds": args.ready_poll_seconds,
                "error": str(exc),
            }
            _write_manifest(manifest_path, manifest)
            raise SystemExit(str(exc)) from exc
        manifest["warm_pod_ready_check"] = {
            "status": "ok",
            "pod_id": str(warm_pod_id),
            "ready_marker": READY_MARKER,
            "timeout_seconds": args.ready_timeout_seconds,
            "poll_seconds": args.ready_poll_seconds,
            "ssh": ssh_info,
            "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _write_manifest(manifest_path, manifest)

    for spec in stage_specs:
        stage: ProgressiveStage = spec["stage"]
        summary_path: Path = spec["summary_path"]
        run_record: dict[str, Any] = {
            "stage": stage.slug,
            "train_count": stage.train_count,
            "eval_count": stage.eval_count,
            "focus_bands": list(stage.focus_bands),
            "focus_weight": stage.focus_weight,
            "focus_band_weights": stage.focus_weights,
            "dataset_name": spec["dataset_name"],
            "output_path": spec["output_path"],
            "summary_path": str(summary_path),
        }
        if args.resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run_record["status"] = "skipped_existing"
            run_record["metrics"] = _extract_stage_metrics(summary)
            run_record["performance"] = _prediction_timing_metrics(summary)
            manifest["runs"].append(run_record)
            _write_manifest(manifest_path, manifest)
            continue

        command = [
            sys.executable,
            "scripts/runpod_train_eval.py",
            "--name",
            f"{args.run_name}-{stage.slug}",
            "--run-mode",
            "train-eval",
            "--dataset-name",
            str(spec["dataset_name"]),
            "--backend",
            "transformers_qlora",
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
            str(args.learning_rate),
            "--epochs",
            str(args.epochs),
            "--max-seq-length",
            str(args.max_seq_length),
            "--per-device-train-batch-size",
            str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--modes",
            "fine_tuned",
            "--train-split",
            "synthetic_train.jsonl",
            "--eval-split",
            DEFAULT_EVAL_SPLIT_NAME,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--inference-batch-size",
            str(args.inference_batch_size),
            "--failure-archive-limit",
            str(args.failure_archive_limit),
            "--step-budget",
            args.step_budget,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-output",
            str(spec["output_path"]),
            "--local-output",
            str(spec["output_path"]),
        ]
        if warm_pod_id:
            command.extend(["--pod-id", str(warm_pod_id), "--reuse-pod-workspace"])
        if args.keep_pod:
            command.append("--keep-pod")

        run_record["command"] = command
        if args.dry_run:
            run_record["status"] = "dry_run"
            manifest["runs"].append(run_record)
            _write_manifest(manifest_path, manifest)
            continue

        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        started_perf = time.perf_counter()
        proc = _run_command(command)
        wall_seconds = time.perf_counter() - started_perf
        run_record["started_at"] = started_at
        run_record["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        run_record["wall_seconds"] = wall_seconds
        parsed = _load_json_output(proc.stdout)
        run_record["returncode"] = proc.returncode
        run_record["stdout_tail"] = proc.stdout.splitlines()[-20:]
        run_record["stderr_tail"] = proc.stderr.splitlines()[-20:]
        if parsed is not None:
            run_record["summary"] = parsed
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run_record["status"] = "ok" if proc.returncode == 0 else "completed_with_nonzero_returncode"
            run_record["metrics"] = _extract_stage_metrics(summary)
            run_record["performance"] = _prediction_timing_metrics(summary)
        else:
            run_record["status"] = "failed_no_summary"
        manifest["runs"].append(run_record)
        _write_manifest(manifest_path, manifest)

        if proc.returncode != 0 and not args.continue_on_error:
            break

    manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    if warm_pod_id and not args.keep_pod and not args.dry_run:
        ctl = RunpodCtl(repo_root=REPO_ROOT)
        manifest["warm_pod_cleanup"] = ctl.delete_pod(str(warm_pod_id))
    _write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
