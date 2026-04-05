import argparse
import json
from pathlib import Path

import pytest

from ibm650_it import REPO_ROOT
from ibm650_it.cli import cmd_overfit_sanity
from ibm650_it.dataset.corpus import build_pilot_corpus
from ibm650_it.dataset.io import load_jsonl, resolve_record_path
from ibm650_it.eval.reevaluate import reevaluate_prediction_records
from ibm650_it.eval.report import build_evaluation_report
from ibm650_it.generate.sample_program import generate_accepted_programs
from ibm650_it.simh.deckio import join_decks, read_deck_cards
from ibm650_it.simh.runner import SimhRunner
from ibm650_it.training.infer import run_inference
from ibm650_it.training.prepare_sft import prepare_sft_examples
from ibm650_it.training.train_unsloth import TrainConfig, train_model


SIMH_BINARY = REPO_ROOT / "third_party/simh/BIN/i650"


pytestmark = pytest.mark.skipif(not SIMH_BINARY.exists(), reason="SIMH i650 binary not built")


def test_translate_only_example2(tmp_path: Path) -> None:
    runner = SimhRunner(repo_root=REPO_ROOT)
    result = runner.translate_only(REPO_ROOT / "third_party/simh/I650/sw/it/it_example_2_src.txt", tmp_path / "translate")
    assert result.status == "ok"
    assert result.upper_acc == 0
    assert result.pit_raw is not None and result.pit_raw.exists()
    assert result.pit_raw_canonical is not None and result.pit_raw_canonical.exists()


def test_reservation_split_reconstructs_phase2_input(tmp_path: Path) -> None:
    runner = SimhRunner(repo_root=REPO_ROOT)
    translation = runner.translate_only(REPO_ROOT / "third_party/simh/I650/sw/it/it_example_2_src.txt", tmp_path / "translate")
    split = runner.split_reservation_cards(translation.pit_raw, tmp_path / "translate")  # type: ignore[arg-type]
    phase2 = runner.build_pit_phase2_input_p1(split.reservation_cards, split.translation_body, tmp_path / "assemble" / "pit_phase2_input_p1.dck")
    expected = join_decks(
        [
            split.reservation_cards,
            REPO_ROOT / "third_party/simh/I650/sw/it/it_reservation_p1.dck",
            split.translation_body,
        ],
        tmp_path / "expected.dck",
    )
    assert read_deck_cards(phase2) == read_deck_cards(expected)


def test_stage_pipeline_matches_shipped_example2_output(tmp_path: Path) -> None:
    runner = SimhRunner(repo_root=REPO_ROOT)
    pipeline = runner.reference_pipeline(
        source_deck=REPO_ROOT / "third_party/simh/I650/sw/it/it_example_2_src.txt",
        output_dir=tmp_path / "pipeline",
    )
    baseline = runner.run_shipped_run_it(
        source_deck=REPO_ROOT / "third_party/simh/I650/sw/it/it_example_2_src.txt",
        output_dir=tmp_path / "baseline",
    )
    assert read_deck_cards(pipeline.run.output_deck) == read_deck_cards(Path(baseline["output_deck"]))  # type: ignore[arg-type]


def test_stage_pipeline_matches_shipped_example1_output(tmp_path: Path) -> None:
    runner = SimhRunner(repo_root=REPO_ROOT)
    pipeline = runner.reference_pipeline(
        source_deck=REPO_ROOT / "third_party/simh/I650/sw/it/it_example_1_src.txt",
        input_deck=REPO_ROOT / "third_party/simh/I650/sw/it/it_example_1_data.txt",
        output_dir=tmp_path / "pipeline",
    )
    baseline = runner.run_shipped_run_it(
        source_deck=REPO_ROOT / "third_party/simh/I650/sw/it/it_example_1_src.txt",
        input_deck=REPO_ROOT / "third_party/simh/I650/sw/it/it_example_1_data.txt",
        output_dir=tmp_path / "baseline",
    )
    assert read_deck_cards(pipeline.run.output_deck) == read_deck_cards(Path(baseline["output_deck"]))  # type: ignore[arg-type]


def test_generate_accepted_b0_and_b1(tmp_path: Path) -> None:
    runner = SimhRunner(repo_root=REPO_ROOT)
    b0 = generate_accepted_programs(
        runner=runner,
        band="B0",
        count=1,
        output_dir=tmp_path / "b0",
        start_seed=1,
    )
    b1 = generate_accepted_programs(
        runner=runner,
        band="B1",
        count=1,
        output_dir=tmp_path / "b1",
        start_seed=100,
    )
    assert b0["accepted"] == 1
    assert b1["accepted"] == 1
    assert Path(b0["index_path"]).exists()
    assert Path(b1["index_path"]).exists()


def test_build_small_pilot_corpus(tmp_path: Path) -> None:
    summary = build_pilot_corpus(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "pilot",
        band_counts={"B0": 2, "B1": 2},
        workers=2,
        include_historical_golden=False,
    )
    assert summary["accepted_total"] == 4
    assert summary["accepted_by_band"]["B0"] == 2
    assert summary["accepted_by_band"]["B1"] == 2
    assert Path(summary["index_path"]).exists()
    assert Path(summary["rejected_path"]).exists()
    records = load_jsonl(Path(summary["index_path"]))
    assert len(records) == 4
    assert len({record["hashes"]["alpha_hash"] for record in records}) == 4
    assert not (tmp_path / "pilot" / "staging").exists()
    for record in records:
        assert not str(record["source"]["it_text_v1"]).startswith("/")
        assert record["provenance"]["simh_commit_or_checksum"]
    for split_path in summary["split_outputs"].values():
        assert Path(split_path).exists()


def test_resume_pilot_corpus(tmp_path: Path) -> None:
    initial = build_pilot_corpus(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "pilot",
        band_counts={"B0": 1, "B1": 1},
        workers=2,
        include_historical_golden=False,
    )
    resumed = build_pilot_corpus(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "pilot",
        band_counts={"B0": 2, "B1": 2},
        workers=2,
        include_historical_golden=False,
        resume=True,
    )
    assert initial["accepted_total"] == 2
    assert resumed["accepted_total"] == 4
    records = load_jsonl(Path(resumed["index_path"]))
    assert len(records) == 4
    assert len({record["hashes"]["alpha_hash"] for record in records}) == 4


def test_smoke_training_eval_pipeline(tmp_path: Path) -> None:
    pilot = build_pilot_corpus(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "pilot",
        band_counts={"B0": 2, "B1": 2},
        workers=2,
        include_historical_golden=False,
    )
    index_path = Path(pilot["index_path"])
    sft_path = tmp_path / "training" / "train.jsonl"
    assert prepare_sft_examples(dataset_index=index_path, output_path=sft_path) == 4

    train_summary = train_model(
        sft_path=sft_path,
        output_dir=tmp_path / "training" / "model",
        config=TrainConfig(backend="smoke"),
    )
    assert train_summary["example_count"] == 4

    zero = run_inference(
        reference_index=index_path,
        output_dir=tmp_path / "eval" / "zero",
        mode="zero_shot",
        limit=1,
    )
    few = run_inference(
        reference_index=index_path,
        output_dir=tmp_path / "eval" / "few",
        mode="few_shot",
        support_sft=sft_path,
        few_shot_k=1,
        limit=1,
    )
    fine = run_inference(
        reference_index=index_path,
        output_dir=tmp_path / "eval" / "fine",
        mode="fine_tuned",
        model_dir=tmp_path / "training" / "model",
        limit=1,
    )

    zero_report = build_evaluation_report(reference_index=index_path, prediction_index=Path(zero["prediction_index"]))
    few_report = build_evaluation_report(reference_index=index_path, prediction_index=Path(few["prediction_index"]))
    fine_report = build_evaluation_report(reference_index=index_path, prediction_index=Path(fine["prediction_index"]))

    assert zero_report["count"] == 1
    assert zero_report["exact_match"] == 0.0
    assert zero_report["assemblability"] == 0.0
    assert zero_report["failure_taxonomy"]["malformed_pit_card"] == 1
    assert "by_statement_count" in zero_report
    assert "by_feature" in zero_report
    assert "by_indexed_usage" in zero_report

    assert few_report["count"] == 1
    assert fine_report["count"] == 1
    assert fine_report["exact_match"] == 1.0
    assert fine_report["assemblability"] == 1.0
    assert fine_report["functional_equivalence"] == 1.0


def test_smoke_overfit_sanity_pipeline(tmp_path: Path) -> None:
    pilot = build_pilot_corpus(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "pilot",
        band_counts={"B0": 2, "B1": 2},
        workers=2,
        include_historical_golden=False,
    )
    index_path = Path(pilot["index_path"])
    output_root = tmp_path / "overfit"

    cmd_overfit_sanity(
        argparse.Namespace(
            dataset_index=str(index_path),
            output=str(output_root),
            example_count=2,
            backend="smoke",
            model_name="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            qlora_bits=4,
            learning_rate=1e-4,
            epochs=3,
            max_seq_length=4096,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_new_tokens=1024,
            eval_mode="inline",
            failure_archive_limit=10,
            step_budget="50M",
            timeout_seconds=30,
        )
    )

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    report = summary["fine_tuned"]["report"]
    assert summary["records_written"] == 2
    assert report["count"] == 2
    assert report["exact_match"] == 1.0
    assert report["assemblability"] == 1.0
    assert report["functional_equivalence"] == 1.0


def test_local_reevaluation_of_exact_match_prediction_produces_successful_assembly(tmp_path: Path) -> None:
    pilot = build_pilot_corpus(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "pilot",
        band_counts={"B0": 2, "B1": 2},
        workers=2,
        include_historical_golden=False,
    )
    index_path = Path(pilot["index_path"])
    sft_path = tmp_path / "training" / "train.jsonl"
    prepare_sft_examples(dataset_index=index_path, output_path=sft_path)
    train_model(
        sft_path=sft_path,
        output_dir=tmp_path / "training" / "model",
        config=TrainConfig(backend="smoke"),
    )
    inference = run_inference(
        reference_index=index_path,
        output_dir=tmp_path / "eval" / "fine",
        mode="fine_tuned",
        model_dir=tmp_path / "training" / "model",
        limit=1,
        eval_mode="skip",
    )
    reevaluate = reevaluate_prediction_records(
        reference_index=index_path,
        prediction_index=Path(inference["prediction_index"]),
        output_dir=tmp_path / "eval" / "fine",
    )
    report = build_evaluation_report(
        reference_index=index_path,
        prediction_index=Path(reevaluate["prediction_index"]),
    )

    assert report["count"] == 1
    assert report["exact_match"] == 1.0
    assert report["assemblability"] == 1.0
    assert report["functional_equivalence"] == 1.0


def test_local_reevaluation_flags_tail_truncation_as_misexecution(tmp_path: Path) -> None:
    pilot = build_pilot_corpus(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "pilot",
        band_counts={"B0": 2, "B1": 2},
        workers=2,
        include_historical_golden=False,
    )
    index_path = Path(pilot["index_path"])
    sft_path = tmp_path / "training" / "train.jsonl"
    prepare_sft_examples(dataset_index=index_path, output_path=sft_path)
    train_model(
        sft_path=sft_path,
        output_dir=tmp_path / "training" / "model",
        config=TrainConfig(backend="smoke"),
    )
    inference = run_inference(
        reference_index=index_path,
        output_dir=tmp_path / "eval" / "fine",
        mode="fine_tuned",
        model_dir=tmp_path / "training" / "model",
        limit=1,
        eval_mode="skip",
    )
    prediction_index = Path(inference["prediction_index"])
    record = load_jsonl(prediction_index)[0]
    candidate_path = resolve_record_path(str(record["pit_raw_canonical"]), prediction_index.parent)
    cards = candidate_path.read_text(encoding="latin-1").splitlines()
    candidate_path.write_text("\n".join(cards[:-1]) + "\n", encoding="latin-1")

    reevaluate = reevaluate_prediction_records(
        reference_index=index_path,
        prediction_index=prediction_index,
        output_dir=tmp_path / "eval" / "fine",
    )
    reevaluated_record = load_jsonl(Path(reevaluate["prediction_index"]))[0]

    assert reevaluated_record["metrics"]["exact_match"] is False
    assert reevaluated_record["assemblable"] is True
    assert reevaluated_record["failure_type"] == "assembles_but_misexecutes"
