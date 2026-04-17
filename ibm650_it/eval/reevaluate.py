from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ibm650_it import REPO_ROOT
from ibm650_it.dataset.io import load_jsonl, relativize_record_paths, resolve_record_base, resolve_record_path, write_jsonl
from ibm650_it.eval.exact_match import compare_pit_files
from ibm650_it.eval.failure_taxonomy import classify_failure, should_attempt_assembly
from ibm650_it.eval.functional import compare_run_outputs
from ibm650_it.simh.deckio import split_tail_cards
from ibm650_it.simh.runner import SimhRunner, output_deck_has_cards


def _error_payload(exc: BaseException) -> dict[str, str]:
    message = str(exc).strip() or repr(exc)
    return {
        "error_type": type(exc).__name__,
        "error_message": message,
    }


def _reference_output_path(reference: dict[str, Any], reference_base: Path) -> Path | None:
    output_deck = reference.get("reference", {}).get("run", {}).get("output_deck")
    if not output_deck:
        return None
    return resolve_record_path(str(output_deck), reference_base)


def _reference_has_successful_run(reference: dict[str, Any], reference_base: Path) -> bool:
    run = reference.get("reference", {}).get("run", {})
    reference_output = _reference_output_path(reference, reference_base)
    return bool(run.get("status") == "ok" and output_deck_has_cards(reference_output))


def _invariant_payload(code: str, message: str) -> dict[str, str]:
    return {
        "type": code,
        "message": message,
    }


def _prediction_dir(prediction: dict[str, Any], prediction_base: Path, output_dir: Path) -> Path:
    pit_ref = str(prediction.get("pit_raw_canonical", ""))
    if pit_ref:
        pit_path = resolve_record_path(pit_ref, prediction_base)
        if pit_path.exists():
            return pit_path.parent
    return output_dir / str(prediction["id"])


def _base_prediction_record(prediction: dict[str, Any]) -> dict[str, Any]:
    record = json.loads(json.dumps(prediction))
    record.setdefault("assemble", {})
    record.setdefault("run", {})
    record.setdefault("timings", {})
    return record


def reevaluate_prediction_records(
    *,
    reference_index: Path,
    prediction_index: Path,
    output_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
    step_budget: str = "50M",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    prediction_base = prediction_index.parent
    output_dir = output_dir or prediction_base
    output_dir.mkdir(parents=True, exist_ok=True)

    references = {str(record["id"]): record for record in load_jsonl(reference_index)}
    reference_base = resolve_record_base(reference_index)
    predictions = load_jsonl(prediction_index)
    runner = SimhRunner(repo_root=repo_root)

    reevaluated_records: list[dict[str, Any]] = []
    mode = None
    for prediction in predictions:
        evaluation_started = time.perf_counter()
        prediction_record = _base_prediction_record(prediction)
        mode = mode or str(prediction_record.get("mode", ""))
        reference = references.get(str(prediction_record["id"]))
        if reference is None:
            reevaluated_records.append(relativize_record_paths(prediction_record, output_dir))
            continue

        prediction_dir = _prediction_dir(prediction_record, prediction_base, output_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        pit_path = resolve_record_path(str(prediction_record.get("pit_raw_canonical", "")), prediction_base)
        prompt_ref = prediction_record.get("prompt_path")
        if prompt_ref:
            prediction_record["prompt_path"] = str(resolve_record_path(str(prompt_ref), prediction_base))
        prediction_record["pit_raw_canonical"] = str(pit_path)

        candidate_cards = pit_path.read_text(encoding="latin-1").splitlines() if pit_path.exists() else []
        reference_pit = resolve_record_path(str(reference["reference"]["translate"]["pit_raw_canonical"]), reference_base)
        stored_metrics = prediction_record.get("metrics", {})
        if pit_path.exists():
            exact_metrics = compare_pit_files(reference_pit, pit_path)
        else:
            exact_metrics = {
                "exact_match": bool(stored_metrics.get("exact_match", False)),
                "per_card_exact": float(stored_metrics.get("per_card_exact", 0.0)),
                "normalized_edit_distance": float(stored_metrics.get("normalized_edit_distance", 1.0)),
            }

        assemble_status = "not_evaluated"
        run_status = "not_run"
        assemblable = False
        functional = False
        evaluator_invariant: dict[str, str] | None = None
        assemble_payload: dict[str, Any] = {}
        run_payload: dict[str, Any] = {}

        if not pit_path.exists():
            assemble_status = "candidate_missing"
            assemble_payload.update(
                {
                    "error_type": "FileNotFoundError",
                    "error_message": f"candidate PIT not found: {pit_path}",
                }
            )
        elif should_attempt_assembly(candidate_cards, exact_match=bool(exact_metrics["exact_match"])):
            try:
                translation_body, reservation_cards = split_tail_cards(
                    pit_path,
                    10,
                    prediction_dir / "translation_body.dck",
                    prediction_dir / "reservation_cards.dck",
                )
                assemble_input = runner.build_pit_phase2_input_p1(
                    reservation_cards,
                    translation_body,
                    prediction_dir / "assemble" / "pit_phase2_input_p1.dck",
                )
                assemble = runner.assemble_pit(
                    assemble_input,
                    prediction_dir / "assemble",
                    timeout_seconds=timeout_seconds,
                )
                assemble_status = assemble.status
                assemblable = assemble.status == "ok" and assemble.soap_output is not None
                assemble_payload.update(
                    {
                        "pit_phase2_input_p1": str(assemble.pit_phase2_input_p1),
                        "soap_output": str(assemble.soap_output) if assemble.soap_output is not None else None,
                        "console_log": str(assemble.console_log),
                        "stdout_log": str(assemble.stdout_log),
                        "print_log": str(assemble.print_log),
                    }
                )
            except Exception as exc:
                assemble_status = "assemble_error"
                assemble_payload.update(_error_payload(exc))
        else:
            assemble_status = "precheck_skipped"

        if bool(exact_metrics["exact_match"]) and not assemblable:
            evaluator_invariant = _invariant_payload(
                "exact_match_failed_to_assemble",
                "Exact PIT match did not assemble during local reevaluation.",
            )

        if assemblable and assemble_payload.get("soap_output"):
            try:
                spit = runner.build_spit_p1(
                    Path(str(assemble_payload["soap_output"])),
                    prediction_dir / "run" / "spit_p1.dck",
                )
                input_deck_ref = reference["reference"]["run"].get("input_deck")
                input_deck = resolve_record_path(str(input_deck_ref), reference_base) if input_deck_ref else None
                run = runner.run_spit(
                    spit.spit_p1,
                    prediction_dir / "run",
                    input_deck=input_deck,
                    step_budget=step_budget,
                    timeout_seconds=timeout_seconds,
                )
                run_status = run.status
                reference_output = _reference_output_path(reference, reference_base)
                functional = (
                    reference_output is not None
                    and output_deck_has_cards(reference_output)
                    and run.status == "ok"
                    and output_deck_has_cards(run.output_deck)
                    and compare_run_outputs(reference_output, run.output_deck)
                )
                run_payload.update(
                    {
                        "spit_p1": str(run.spit_p1),
                        "output_deck": str(run.output_deck) if run.output_deck is not None else None,
                        "console_log": str(run.console_log),
                        "stdout_log": str(run.stdout_log),
                        "print_log": str(run.print_log),
                    }
                )
            except Exception as exc:
                run_status = "run_error"
                run_payload.update(_error_payload(exc))

        if evaluator_invariant is None and bool(exact_metrics["exact_match"]) and _reference_has_successful_run(reference, reference_base) and not functional:
            evaluator_invariant = _invariant_payload(
                "exact_match_failed_functional_check",
                "Exact PIT match did not complete a matching local functional run.",
            )

        failure_type = classify_failure(
            candidate_cards=candidate_cards,
            exact_match=bool(exact_metrics["exact_match"]),
            assemblable=assemblable,
            functional=functional,
            assemble_status=assemble_status,
            evaluator_invariant=evaluator_invariant["type"] if evaluator_invariant else None,
        )

        prediction_record["metrics"] = exact_metrics
        prediction_record["assemble"] = {
            "status": assemble_status,
            **assemble_payload,
        }
        prediction_record["run"] = {
            "status": run_status,
            **run_payload,
        }
        prediction_record["assemblable"] = assemblable
        prediction_record["functional"] = functional
        prediction_record["failure_type"] = failure_type
        if evaluator_invariant is not None:
            prediction_record["evaluator_invariant"] = evaluator_invariant
        elif "evaluator_invariant" in prediction_record:
            del prediction_record["evaluator_invariant"]

        timings = prediction_record.setdefault("timings", {})
        generation_seconds = float(timings.get("generation_seconds", 0.0))
        evaluation_seconds = time.perf_counter() - evaluation_started
        timings["generation_seconds"] = generation_seconds
        timings["evaluation_seconds"] = evaluation_seconds
        timings["total_seconds"] = generation_seconds + evaluation_seconds

        reevaluated_records.append(relativize_record_paths(prediction_record, output_dir))

    prediction_output = write_jsonl(output_dir / "predictions.jsonl", reevaluated_records)
    summary = {
        "mode": mode,
        "count": len(reevaluated_records),
        "prediction_index": str(prediction_output),
        "eval_mode": "local_cpu_reevaluate",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
