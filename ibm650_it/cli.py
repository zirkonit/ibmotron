from __future__ import annotations

import argparse
import json
from pathlib import Path

from ibm650_it import REPO_ROOT
from ibm650_it.dashboard import serve_dashboard
from ibm650_it.dataset.build_records import build_record
from ibm650_it.dataset.corpus import build_pilot_corpus, build_stage_corpus, parse_band_counts
from ibm650_it.dataset.subset import slice_dataset
from ibm650_it.eval.archive import archive_failures
from ibm650_it.eval.b1_failure_review import build_b1_failure_review
from ibm650_it.eval.band_failure_review import build_band_failure_review
from ibm650_it.eval.finalize import finalize_overfit_output, finalize_train_eval_output, reevaluate_and_report_mode
from ibm650_it.eval.report import build_evaluation_report, compare_mode_reports
from ibm650_it.eval.research_report import write_research_report
from ibm650_it.generate.sample_program import generate_accepted_programs, generate_band_program
from ibm650_it.pit.normalize_pit import canonicalize_pit_file
from ibm650_it.simh.runner import SimhRunner
from ibm650_it.source.render_it_text import render_program
from ibm650_it.training.infer import run_inference
from ibm650_it.training.prepare_sft import parse_band_repeats, prepare_sft_examples
from ibm650_it.training.thinking_ablation import run_thinking_ablation
from ibm650_it.training.train_unsloth import TrainConfig, train_model


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _runner() -> SimhRunner:
    return SimhRunner(repo_root=REPO_ROOT)


def cmd_translate(args: argparse.Namespace) -> None:
    result = _runner().translate_only(Path(args.source), Path(args.output))
    _print_json(result.to_dict())


def cmd_split_reservations(args: argparse.Namespace) -> None:
    result = _runner().split_reservation_cards(Path(args.pit_raw), Path(args.output))
    _print_json(result.to_dict())


def cmd_build_phase2(args: argparse.Namespace) -> None:
    output = _runner().build_pit_phase2_input_p1(
        Path(args.reservation_cards),
        Path(args.translation_body),
        Path(args.output),
    )
    _print_json({"pit_phase2_input_p1": output})


def cmd_assemble(args: argparse.Namespace) -> None:
    result = _runner().assemble_pit(Path(args.pit_phase2_input_p1), Path(args.output), timeout_seconds=args.timeout_seconds)
    _print_json(result.to_dict())


def cmd_build_spit(args: argparse.Namespace) -> None:
    result = _runner().build_spit_p1(Path(args.soap_output), Path(args.output))
    _print_json(result.to_dict())


def cmd_run_spit(args: argparse.Namespace) -> None:
    result = _runner().run_spit(
        Path(args.spit_p1),
        Path(args.output),
        input_deck=Path(args.input) if args.input else None,
        step_budget=args.step_budget,
        timeout_seconds=args.timeout_seconds,
    )
    _print_json(result.to_dict())


def cmd_pipeline(args: argparse.Namespace) -> None:
    result = _runner().reference_pipeline(
        source_deck=Path(args.source),
        output_dir=Path(args.output),
        input_deck=Path(args.input) if args.input else None,
        step_budget=args.step_budget,
        timeout_seconds=args.timeout_seconds,
    )
    _print_json(result.to_dict())


def cmd_smoke_examples(args: argparse.Namespace) -> None:
    output_root = Path(args.output)
    runner = _runner()
    examples = [
        {
            "name": "it_example_1",
            "source": REPO_ROOT / "third_party/simh/I650/sw/it/it_example_1_src.txt",
            "input": REPO_ROOT / "third_party/simh/I650/sw/it/it_example_1_data.txt",
        },
        {
            "name": "it_example_2",
            "source": REPO_ROOT / "third_party/simh/I650/sw/it/it_example_2_src.txt",
            "input": None,
        },
    ]
    results = []
    for example in examples:
        results.append(
            runner.reference_pipeline(
                source_deck=example["source"],
                output_dir=output_root / example["name"],
                input_deck=example["input"],
            ).to_dict()
        )
    _print_json(results)


def cmd_generate_sample(args: argparse.Namespace) -> None:
    program = generate_band_program(args.band, seed=args.seed)
    Path(args.output).write_text(render_program(program), encoding="utf-8")


def cmd_generate_accepted(args: argparse.Namespace) -> None:
    summary = generate_accepted_programs(
        runner=_runner(),
        band=args.band,
        count=args.count,
        output_dir=Path(args.output),
        start_seed=args.start_seed,
        max_attempts=args.max_attempts,
    )
    _print_json(summary)


def cmd_prepare_sft(args: argparse.Namespace) -> None:
    records = prepare_sft_examples(
        dataset_index=Path(args.dataset_index),
        output_path=Path(args.output),
        limit=args.limit,
        band_repeats=parse_band_repeats(args.band_repeat),
    )
    _print_json({"records_written": records, "output": args.output})


def cmd_build_pilot_corpus(args: argparse.Namespace) -> None:
    summary = build_pilot_corpus(
        output_root=Path(args.output),
        band_counts=parse_band_counts(args.band_count, total_count=args.total_count),
        workers=args.workers,
        max_attempts_per_band=args.max_attempts_per_band,
        include_historical_golden=not args.no_historical_golden,
        resume=args.resume,
    )
    _print_json(summary)


def cmd_build_stage_corpus(args: argparse.Namespace) -> None:
    summary = build_stage_corpus(
        stage=args.stage,
        output_root=Path(args.output),
        workers=args.workers,
        max_attempts_per_band=args.max_attempts_per_band,
        include_historical_golden=args.include_historical_golden,
        resume=args.resume,
    )
    _print_json(summary)


def cmd_slice_dataset(args: argparse.Namespace) -> None:
    summary = slice_dataset(
        source_root=Path(args.source_root),
        output_root=Path(args.output),
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
        adversarial_limit=args.adversarial_limit,
        include_historical_golden=not args.no_historical_golden,
    )
    _print_json(summary)


def cmd_train_model(args: argparse.Namespace) -> None:
    summary = train_model(
        sft_path=Path(args.sft_jsonl),
        output_dir=Path(args.output),
        config=TrainConfig(
            backend=args.backend,
            model_name=args.model_name,
            qlora_bits=args.qlora_bits,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        ),
        resume_from=Path(args.resume_from) if args.resume_from else None,
        max_examples=args.max_examples,
    )
    _print_json(summary)


def cmd_run_inference(args: argparse.Namespace) -> None:
    summary = run_inference(
        reference_index=Path(args.reference_index),
        output_dir=Path(args.output),
        mode=args.mode,
        model_dir=Path(args.model) if args.model else None,
        support_sft=Path(args.support_sft) if args.support_sft else None,
        few_shot_k=args.few_shot_k,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        prompt_style=args.prompt_style,
        enable_thinking=args.enable_thinking,
        preserve_raw_completion=args.preserve_raw_completion,
        step_budget=args.step_budget,
        timeout_seconds=args.timeout_seconds,
        eval_mode=args.eval_mode,
        inference_batch_size=getattr(args, "inference_batch_size", 1),
    )
    _print_json(summary)


def cmd_thinking_ablation(args: argparse.Namespace) -> None:
    summary = run_thinking_ablation(
        reference_index=Path(args.reference_index),
        output_root=Path(args.output),
        model_dir=Path(args.model),
        repo_root=REPO_ROOT,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        failure_archive_limit=args.failure_archive_limit,
        step_budget=args.step_budget,
        timeout_seconds=args.timeout_seconds,
        eval_mode=args.eval_mode,
    )
    _print_json(summary)


def cmd_reevaluate_predictions(args: argparse.Namespace) -> None:
    summary = reevaluate_and_report_mode(
        reference_index=Path(args.reference_index),
        prediction_index=Path(args.prediction_index),
        prediction_output_dir=Path(args.output),
        report_path=Path(args.report_output) if args.report_output else Path(args.output) / "report.json",
        failure_output_dir=Path(args.failure_output) if args.failure_output else Path(args.output) / "failures",
        failure_archive_limit=args.failure_archive_limit,
        repo_root=REPO_ROOT,
        step_budget=args.step_budget,
        timeout_seconds=args.timeout_seconds,
    )
    _print_json(summary)


def cmd_review_b1_failures(args: argparse.Namespace) -> None:
    summary = build_b1_failure_review(
        reference_index=Path(args.reference_index),
        prediction_index=Path(args.prediction_index),
        output_root=Path(args.output),
    )
    _print_json(summary)


def cmd_review_band_failures(args: argparse.Namespace) -> None:
    summary = build_band_failure_review(
        reference_index=Path(args.reference_index),
        prediction_index=Path(args.prediction_index),
        output_root=Path(args.output),
        bands=args.band,
    )
    _print_json(summary)


def cmd_build_record(args: argparse.Namespace) -> None:
    runner = _runner()
    pipeline = runner.reference_pipeline(
        source_deck=Path(args.source),
        output_dir=Path(args.output),
        input_deck=Path(args.input) if args.input else None,
    )
    record = build_record(
        band=args.band,
        seed=args.seed,
        source_path=Path(args.source),
        pipeline=pipeline,
        repo_root=REPO_ROOT,
        input_deck=Path(args.input) if args.input else None,
    )
    record_path = Path(args.output) / "record.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    _print_json({"record_path": record_path})


def cmd_canonicalize_pit(args: argparse.Namespace) -> None:
    canonicalize_pit_file(Path(args.input), Path(args.output))
    _print_json({"output": args.output})


def cmd_eval_report(args: argparse.Namespace) -> None:
    report = build_evaluation_report(
        reference_index=Path(args.reference_index),
        prediction_index=Path(args.prediction_index),
    )
    _print_json(report)


def cmd_archive_failures(args: argparse.Namespace) -> None:
    manifest = archive_failures(
        reference_index=Path(args.reference_index),
        prediction_index=Path(args.prediction_index),
        output_dir=Path(args.output),
        limit=args.limit,
    )
    _print_json(manifest)


def cmd_compare_reports(args: argparse.Namespace) -> None:
    payload = {
        "zero_shot": json.loads(Path(args.zero_shot).read_text(encoding="utf-8")),
        "few_shot": json.loads(Path(args.few_shot).read_text(encoding="utf-8")),
        "fine_tuned": json.loads(Path(args.fine_tuned).read_text(encoding="utf-8")),
    }
    _print_json(compare_mode_reports(payload))


def cmd_write_research_report(args: argparse.Namespace) -> None:
    output = write_research_report(
        summary_path=Path(args.summary),
        output_path=Path(args.output),
    )
    _print_json({"output": str(output)})


def cmd_dashboard(args: argparse.Namespace) -> None:
    serve_dashboard(host=args.host, port=args.port, refresh_seconds=args.refresh_seconds)


def cmd_train_eval(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    train_index = dataset_root / "splits" / args.train_split
    eval_index = dataset_root / "splits" / args.eval_split
    sft_path = output_root / "sft" / "train.jsonl"
    records_written = prepare_sft_examples(
        dataset_index=train_index,
        output_path=sft_path,
        band_repeats=parse_band_repeats(args.band_repeat),
    )

    train_summary = train_model(
        sft_path=sft_path,
        output_dir=output_root / "model",
        config=TrainConfig(
            backend=args.backend,
            model_name=args.model_name,
            qlora_bits=args.qlora_bits,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        ),
        max_examples=args.max_examples,
    )

    results = {
        "records_written": records_written,
        "train": train_summary,
        "evaluations": {},
        "eval_mode": args.eval_mode,
    }
    for mode in ["zero_shot", "few_shot", "fine_tuned"]:
        inference_summary = run_inference(
            reference_index=eval_index,
            output_dir=output_root / "predictions" / mode,
            mode=mode,
            model_dir=output_root / "model" if mode == "fine_tuned" else None,
            support_sft=sft_path if mode == "few_shot" else None,
            few_shot_k=args.few_shot_k,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            step_budget=args.step_budget,
            timeout_seconds=args.timeout_seconds,
            eval_mode=args.eval_mode,
            inference_batch_size=getattr(args, "inference_batch_size", 1),
        )
        results["evaluations"][mode] = {
            "prediction_index": inference_summary["prediction_index"],
        }

    if args.eval_mode == "inline":
        results = finalize_train_eval_output(
            dataset_root=dataset_root,
            output_root=output_root,
            eval_split=args.eval_split,
            failure_archive_limit=args.failure_archive_limit,
            repo_root=REPO_ROOT,
            step_budget=args.step_budget,
            timeout_seconds=args.timeout_seconds,
        )
        results["records_written"] = records_written
        results["train"] = train_summary

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _print_json(results)


def cmd_overfit_sanity(args: argparse.Namespace) -> None:
    dataset_index = Path(args.dataset_index)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    sft_path = output_root / "sft" / "train.jsonl"
    records_written = prepare_sft_examples(
        dataset_index=dataset_index,
        output_path=sft_path,
        limit=args.example_count,
        band_repeats=parse_band_repeats(args.band_repeat),
    )
    train_summary = train_model(
        sft_path=sft_path,
        output_dir=output_root / "model",
        config=TrainConfig(
            backend=args.backend,
            model_name=args.model_name,
            qlora_bits=args.qlora_bits,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        ),
        max_examples=args.example_count,
    )
    inference_summary = run_inference(
        reference_index=dataset_index,
        output_dir=output_root / "predictions" / "fine_tuned",
        mode="fine_tuned",
        model_dir=output_root / "model",
        limit=args.example_count,
        max_new_tokens=args.max_new_tokens,
        step_budget=args.step_budget,
        timeout_seconds=args.timeout_seconds,
        eval_mode=args.eval_mode,
        inference_batch_size=getattr(args, "inference_batch_size", 1),
    )
    summary = {
        "records_written": records_written,
        "example_count": args.example_count,
        "train": train_summary,
        "fine_tuned": {"prediction_index": inference_summary["prediction_index"]},
        "eval_mode": args.eval_mode,
    }
    if args.eval_mode == "inline":
        summary = finalize_overfit_output(
            dataset_index=dataset_index,
            output_root=output_root,
            failure_archive_limit=args.failure_archive_limit,
            repo_root=REPO_ROOT,
            step_budget=args.step_budget,
            timeout_seconds=args.timeout_seconds,
        )
        summary["records_written"] = records_written
        summary["example_count"] = args.example_count
        summary["train"] = train_summary

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_json(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ibm650-it")
    subparsers = parser.add_subparsers(dest="command", required=True)

    translate = subparsers.add_parser("translate")
    translate.add_argument("--source", required=True)
    translate.add_argument("--output", required=True)
    translate.set_defaults(func=cmd_translate)

    split = subparsers.add_parser("split-reservations")
    split.add_argument("--pit-raw", required=True)
    split.add_argument("--output", required=True)
    split.set_defaults(func=cmd_split_reservations)

    phase2 = subparsers.add_parser("build-phase2-p1")
    phase2.add_argument("--reservation-cards", required=True)
    phase2.add_argument("--translation-body", required=True)
    phase2.add_argument("--output", required=True)
    phase2.set_defaults(func=cmd_build_phase2)

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--pit-phase2-input-p1", required=True)
    assemble.add_argument("--output", required=True)
    assemble.add_argument("--timeout-seconds", type=int, default=30)
    assemble.set_defaults(func=cmd_assemble)

    spit = subparsers.add_parser("build-spit")
    spit.add_argument("--soap-output", required=True)
    spit.add_argument("--output", required=True)
    spit.set_defaults(func=cmd_build_spit)

    run = subparsers.add_parser("run-spit")
    run.add_argument("--spit-p1", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--input")
    run.add_argument("--step-budget", default="50M")
    run.add_argument("--timeout-seconds", type=int, default=30)
    run.set_defaults(func=cmd_run_spit)

    pipeline = subparsers.add_parser("pipeline")
    pipeline.add_argument("--source", required=True)
    pipeline.add_argument("--output", required=True)
    pipeline.add_argument("--input")
    pipeline.add_argument("--step-budget", default="50M")
    pipeline.add_argument("--timeout-seconds", type=int, default=30)
    pipeline.set_defaults(func=cmd_pipeline)

    smoke = subparsers.add_parser("smoke-examples")
    smoke.add_argument("--output", required=True)
    smoke.set_defaults(func=cmd_smoke_examples)

    gen = subparsers.add_parser("generate-sample")
    gen.add_argument("--band", default="B0")
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--output", required=True)
    gen.set_defaults(func=cmd_generate_sample)

    accepted = subparsers.add_parser("generate-accepted")
    accepted.add_argument("--band", required=True)
    accepted.add_argument("--count", type=int, required=True)
    accepted.add_argument("--output", required=True)
    accepted.add_argument("--start-seed", type=int, default=1)
    accepted.add_argument("--max-attempts", type=int)
    accepted.set_defaults(func=cmd_generate_accepted)

    sft = subparsers.add_parser("prepare-sft")
    sft.add_argument("--dataset-index", required=True)
    sft.add_argument("--output", required=True)
    sft.add_argument("--limit", type=int)
    sft.add_argument("--band-repeat", action="append", default=[])
    sft.set_defaults(func=cmd_prepare_sft)

    pilot = subparsers.add_parser("build-pilot-corpus")
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--total-count", type=int, default=1000)
    pilot.add_argument("--band-count", action="append")
    pilot.add_argument("--workers", type=int, default=4)
    pilot.add_argument("--max-attempts-per-band", type=int)
    pilot.add_argument("--no-historical-golden", action="store_true")
    pilot.add_argument("--resume", action="store_true")
    pilot.set_defaults(func=cmd_build_pilot_corpus)

    stage = subparsers.add_parser("build-stage-corpus")
    stage.add_argument("--stage", choices=["2k", "5k", "10k"], required=True)
    stage.add_argument("--output", required=True)
    stage.add_argument("--workers", type=int, default=4)
    stage.add_argument("--max-attempts-per-band", type=int)
    stage.add_argument("--include-historical-golden", action="store_true")
    stage.add_argument("--resume", action="store_true")
    stage.set_defaults(func=cmd_build_stage_corpus)

    subset = subparsers.add_parser("slice-dataset")
    subset.add_argument("--source-root", required=True)
    subset.add_argument("--output", required=True)
    subset.add_argument("--train-limit", type=int)
    subset.add_argument("--dev-limit", type=int)
    subset.add_argument("--test-limit", type=int)
    subset.add_argument("--adversarial-limit", type=int)
    subset.add_argument("--no-historical-golden", action="store_true")
    subset.set_defaults(func=cmd_slice_dataset)

    train = subparsers.add_parser("train-model")
    train.add_argument("--sft-jsonl", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--backend", default="smoke")
    train.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    train.add_argument("--qlora-bits", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--max-seq-length", type=int, default=4096)
    train.add_argument("--per-device-train-batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--resume-from")
    train.add_argument("--max-examples", type=int)
    train.set_defaults(func=cmd_train_model)

    infer = subparsers.add_parser("run-inference")
    infer.add_argument("--reference-index", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--mode", choices=["zero_shot", "few_shot", "fine_tuned"], required=True)
    infer.add_argument("--model")
    infer.add_argument("--support-sft")
    infer.add_argument("--few-shot-k", type=int, default=4)
    infer.add_argument("--limit", type=int)
    infer.add_argument("--max-new-tokens", type=int, default=1536)
    infer.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help="Batch size for hf_session-backed greedy generation. 1 = serial (default), "
             ">1 batches prompts into a single model.generate() call for throughput.",
    )
    infer.add_argument("--prompt-style", choices=["plain", "chat"], default="plain")
    infer.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    infer.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    infer.set_defaults(enable_thinking=None)
    infer.add_argument("--preserve-raw-completion", action="store_true")
    infer.add_argument("--eval-mode", choices=["inline", "skip"], default="inline")
    infer.add_argument("--step-budget", default="50M")
    infer.add_argument("--timeout-seconds", type=int, default=30)
    infer.set_defaults(func=cmd_run_inference)

    thinking = subparsers.add_parser("thinking-ablation")
    thinking.add_argument("--reference-index", required=True)
    thinking.add_argument("--model", required=True)
    thinking.add_argument("--output", required=True)
    thinking.add_argument("--limit", type=int)
    thinking.add_argument("--max-new-tokens", type=int, default=1536)
    thinking.add_argument("--eval-mode", choices=["inline", "skip"], default="inline")
    thinking.add_argument("--failure-archive-limit", type=int, default=25)
    thinking.add_argument("--step-budget", default="50M")
    thinking.add_argument("--timeout-seconds", type=int, default=30)
    thinking.set_defaults(func=cmd_thinking_ablation)

    reevaluate = subparsers.add_parser("reevaluate-predictions")
    reevaluate.add_argument("--reference-index", required=True)
    reevaluate.add_argument("--prediction-index", required=True)
    reevaluate.add_argument("--output", required=True)
    reevaluate.add_argument("--report-output")
    reevaluate.add_argument("--failure-output")
    reevaluate.add_argument("--failure-archive-limit", type=int, default=25)
    reevaluate.add_argument("--step-budget", default="50M")
    reevaluate.add_argument("--timeout-seconds", type=int, default=30)
    reevaluate.set_defaults(func=cmd_reevaluate_predictions)

    review_b1 = subparsers.add_parser("review-b1-failures")
    review_b1.add_argument("--reference-index", required=True)
    review_b1.add_argument("--prediction-index", required=True)
    review_b1.add_argument("--output", required=True)
    review_b1.set_defaults(func=cmd_review_b1_failures)

    review_bands = subparsers.add_parser("review-band-failures")
    review_bands.add_argument("--reference-index", required=True)
    review_bands.add_argument("--prediction-index", required=True)
    review_bands.add_argument("--output", required=True)
    review_bands.add_argument("--band", action="append", required=True)
    review_bands.set_defaults(func=cmd_review_band_failures)

    record = subparsers.add_parser("build-record")
    record.add_argument("--source", required=True)
    record.add_argument("--output", required=True)
    record.add_argument("--band", default="historical_golden")
    record.add_argument("--seed", type=int, default=0)
    record.add_argument("--input")
    record.set_defaults(func=cmd_build_record)

    pit = subparsers.add_parser("canonicalize-pit")
    pit.add_argument("--input", required=True)
    pit.add_argument("--output", required=True)
    pit.set_defaults(func=cmd_canonicalize_pit)

    report = subparsers.add_parser("eval-report")
    report.add_argument("--reference-index", required=True)
    report.add_argument("--prediction-index", required=True)
    report.set_defaults(func=cmd_eval_report)

    archive = subparsers.add_parser("archive-failures")
    archive.add_argument("--reference-index", required=True)
    archive.add_argument("--prediction-index", required=True)
    archive.add_argument("--output", required=True)
    archive.add_argument("--limit", type=int)
    archive.set_defaults(func=cmd_archive_failures)

    compare = subparsers.add_parser("compare-reports")
    compare.add_argument("--zero-shot", required=True)
    compare.add_argument("--few-shot", required=True)
    compare.add_argument("--fine-tuned", required=True)
    compare.set_defaults(func=cmd_compare_reports)

    report_md = subparsers.add_parser("write-research-report")
    report_md.add_argument("--summary", required=True)
    report_md.add_argument("--output", required=True)
    report_md.set_defaults(func=cmd_write_research_report)

    dashboard = subparsers.add_parser("dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--refresh-seconds", type=int, default=10)
    dashboard.set_defaults(func=cmd_dashboard)

    train_eval = subparsers.add_parser("train-eval")
    train_eval.add_argument("--dataset-root", required=True)
    train_eval.add_argument("--output", required=True)
    train_eval.add_argument("--backend", default="transformers_qlora")
    train_eval.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    train_eval.add_argument("--qlora-bits", type=int, default=0)
    train_eval.add_argument("--learning-rate", type=float, default=2e-4)
    train_eval.add_argument("--epochs", type=int, default=5)
    train_eval.add_argument("--max-seq-length", type=int, default=4096)
    train_eval.add_argument("--per-device-train-batch-size", type=int, default=1)
    train_eval.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train_eval.add_argument("--train-split", default="synthetic_train.jsonl")
    train_eval.add_argument("--eval-split", default="synthetic_dev.jsonl")
    train_eval.add_argument("--few-shot-k", type=int, default=4)
    train_eval.add_argument("--band-repeat", action="append", default=[])
    train_eval.add_argument("--limit", type=int)
    train_eval.add_argument("--max-examples", type=int)
    train_eval.add_argument("--max-new-tokens", type=int, default=1536)
    train_eval.add_argument("--inference-batch-size", type=int, default=1)
    train_eval.add_argument("--eval-mode", choices=["inline", "skip"], default="inline")
    train_eval.add_argument("--failure-archive-limit", type=int, default=25)
    train_eval.add_argument("--step-budget", default="50M")
    train_eval.add_argument("--timeout-seconds", type=int, default=30)
    train_eval.set_defaults(func=cmd_train_eval)

    smoke_train = subparsers.add_parser("smoke-train-eval")
    smoke_train.add_argument("--dataset-root", required=True)
    smoke_train.add_argument("--output", required=True)
    smoke_train.add_argument("--train-split", default="synthetic_train.jsonl")
    smoke_train.add_argument("--eval-split", default="synthetic_dev.jsonl")
    smoke_train.add_argument("--few-shot-k", type=int, default=4)
    smoke_train.add_argument("--band-repeat", action="append", default=[])
    smoke_train.add_argument("--limit", type=int)
    smoke_train.add_argument("--max-examples", type=int)
    smoke_train.add_argument("--max-new-tokens", type=int, default=1536)
    smoke_train.add_argument("--inference-batch-size", type=int, default=1)
    smoke_train.add_argument("--eval-mode", choices=["inline", "skip"], default="inline")
    smoke_train.add_argument("--failure-archive-limit", type=int, default=25)
    smoke_train.add_argument("--step-budget", default="50M")
    smoke_train.add_argument("--timeout-seconds", type=int, default=30)
    smoke_train.set_defaults(
        func=cmd_train_eval,
        backend="smoke",
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        qlora_bits=4,
        learning_rate=1e-4,
        epochs=3,
        max_seq_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
    )

    overfit = subparsers.add_parser("overfit-sanity")
    overfit.add_argument("--dataset-index", required=True)
    overfit.add_argument("--output", required=True)
    overfit.add_argument("--example-count", type=int, default=16)
    overfit.add_argument("--backend", default="transformers_qlora")
    overfit.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    overfit.add_argument("--qlora-bits", type=int, default=0)
    overfit.add_argument("--learning-rate", type=float, default=2e-4)
    overfit.add_argument("--epochs", type=int, default=5)
    overfit.add_argument("--max-seq-length", type=int, default=4096)
    overfit.add_argument("--per-device-train-batch-size", type=int, default=1)
    overfit.add_argument("--gradient-accumulation-steps", type=int, default=8)
    overfit.add_argument("--band-repeat", action="append", default=[])
    overfit.add_argument("--max-new-tokens", type=int, default=1536)
    overfit.add_argument("--inference-batch-size", type=int, default=1)
    overfit.add_argument("--eval-mode", choices=["inline", "skip"], default="inline")
    overfit.add_argument("--failure-archive-limit", type=int, default=25)
    overfit.add_argument("--step-budget", default="50M")
    overfit.add_argument("--timeout-seconds", type=int, default=30)
    overfit.set_defaults(func=cmd_overfit_sanity)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
