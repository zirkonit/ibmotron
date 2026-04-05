from __future__ import annotations

import argparse
import json
from pathlib import Path

from ibm650_it import REPO_ROOT
from ibm650_it.dataset.build_records import build_record
from ibm650_it.dataset.corpus import build_pilot_corpus, parse_band_counts
from ibm650_it.eval.report import build_evaluation_report
from ibm650_it.generate.sample_program import generate_accepted_programs, generate_band_program
from ibm650_it.pit.normalize_pit import canonicalize_pit_file
from ibm650_it.simh.runner import SimhRunner
from ibm650_it.source.render_it_text import render_program
from ibm650_it.training.infer import run_inference
from ibm650_it.training.prepare_sft import prepare_sft_examples
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
    )
    _print_json({"records_written": records, "output": args.output})


def cmd_build_pilot_corpus(args: argparse.Namespace) -> None:
    summary = build_pilot_corpus(
        output_root=Path(args.output),
        band_counts=parse_band_counts(args.band_count, total_count=args.total_count),
        workers=args.workers,
        max_attempts_per_band=args.max_attempts_per_band,
        include_historical_golden=not args.no_historical_golden,
    )
    _print_json(summary)


def cmd_train_model(args: argparse.Namespace) -> None:
    summary = train_model(
        sft_path=Path(args.sft_jsonl),
        output_dir=Path(args.output),
        config=TrainConfig(backend=args.backend),
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
        step_budget=args.step_budget,
        timeout_seconds=args.timeout_seconds,
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


def cmd_smoke_train_eval(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    train_index = dataset_root / "splits" / args.train_split
    eval_index = dataset_root / "splits" / args.eval_split
    sft_path = output_root / "sft" / "train.jsonl"
    records_written = prepare_sft_examples(dataset_index=train_index, output_path=sft_path)

    train_summary = train_model(
        sft_path=sft_path,
        output_dir=output_root / "model",
        config=TrainConfig(backend="smoke"),
        max_examples=args.max_examples,
    )

    results = {
        "records_written": records_written,
        "train": train_summary,
        "evaluations": {},
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
            step_budget=args.step_budget,
            timeout_seconds=args.timeout_seconds,
        )
        report = build_evaluation_report(
            reference_index=eval_index,
            prediction_index=Path(inference_summary["prediction_index"]),
        )
        report_path = output_root / "reports" / f"{mode}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        results["evaluations"][mode] = {
            "prediction_index": inference_summary["prediction_index"],
            "report_path": str(report_path),
            "report": report,
        }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _print_json(results)


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
    sft.set_defaults(func=cmd_prepare_sft)

    pilot = subparsers.add_parser("build-pilot-corpus")
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--total-count", type=int, default=1000)
    pilot.add_argument("--band-count", action="append")
    pilot.add_argument("--workers", type=int, default=4)
    pilot.add_argument("--max-attempts-per-band", type=int)
    pilot.add_argument("--no-historical-golden", action="store_true")
    pilot.set_defaults(func=cmd_build_pilot_corpus)

    train = subparsers.add_parser("train-model")
    train.add_argument("--sft-jsonl", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--backend", default="smoke")
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
    infer.add_argument("--step-budget", default="50M")
    infer.add_argument("--timeout-seconds", type=int, default=30)
    infer.set_defaults(func=cmd_run_inference)

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

    smoke_train = subparsers.add_parser("smoke-train-eval")
    smoke_train.add_argument("--dataset-root", required=True)
    smoke_train.add_argument("--output", required=True)
    smoke_train.add_argument("--train-split", default="synthetic_train.jsonl")
    smoke_train.add_argument("--eval-split", default="synthetic_dev.jsonl")
    smoke_train.add_argument("--few-shot-k", type=int, default=4)
    smoke_train.add_argument("--limit", type=int)
    smoke_train.add_argument("--max-examples", type=int)
    smoke_train.add_argument("--step-budget", default="50M")
    smoke_train.add_argument("--timeout-seconds", type=int, default=30)
    smoke_train.set_defaults(func=cmd_smoke_train_eval)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
