from __future__ import annotations

from scripts import runpod_sweep


def test_build_sweep_runs_builds_cartesian_grid() -> None:
    runs = runpod_sweep.build_sweep_runs(
        sweep_name="subset_128_20",
        output_root="artifacts/eval_reports/sweeps/subset_128_20",
        epochs=[3, 5],
        learning_rates=[1e-4, 5e-5],
    )

    assert [run.epochs for run in runs] == [3, 3, 5, 5]
    assert [run.learning_rate for run in runs] == [1e-4, 5e-5, 1e-4, 5e-5]
    assert runs[0].pod_name == "subset_128_20-e3_lr0p0001"
    assert runs[-1].output_path == "artifacts/eval_reports/sweeps/subset_128_20/e5_lr5em05"


def test_extract_metrics_gate_and_selection_read_summary_shape() -> None:
    summary = {
        "evaluations": {
            "few_shot": {
                "report": {
                    "exact_match": 0.1,
                    "assemblability": 0.5,
                    "functional_equivalence": 0.0,
                    "per_card_exact": 0.2,
                    "normalized_edit_distance": 0.8,
                    "failure_taxonomy": {"assembles_but_misexecutes": 10},
                }
            },
            "fine_tuned": {
                "report": {
                    "exact_match": 0.2,
                    "assemblability": 1.0,
                    "functional_equivalence": 0.1,
                    "per_card_exact": 0.7,
                    "normalized_edit_distance": 0.2,
                    "failure_taxonomy": {"assembles_but_misexecutes": 8, "returned_it_source_instead_of_pit": 1},
                    "by_band": {"B1": {"functional_equivalence": 0.45}},
                }
            },
        }
    }

    metrics = runpod_sweep.extract_metrics(summary)
    gate = runpod_sweep.evaluate_gate(summary)
    selection = runpod_sweep.selection_score(summary)
    gate_a = runpod_sweep.evaluate_gate_a(summary)

    assert metrics["fine_tuned"]["per_card_exact"] == 0.7
    assert gate["fine_tuned_beats_few_shot_exact"] is True
    assert gate["fine_tuned_beats_few_shot_assemblability"] is True
    assert gate["returned_it_source_not_dominant"] is True
    assert gate["passes_plan_gate"] is True
    assert selection["eligible"] is True
    assert selection["score"] == (0.1, 0.2, 0.7)
    assert gate_a["b1_functional_at_least_40"] is True
    assert gate_a["passes_gate_a"] is False


def test_load_json_output_accepts_summary_stdout() -> None:
    payload = runpod_sweep._load_json_output('{"pod_id":"abc123","returncode":0}')

    assert payload == {"pod_id": "abc123", "returncode": 0}


def test_compare_to_baseline_prefers_higher_functional_then_exact_then_card_score() -> None:
    baseline = {
        "evaluations": {
            "fine_tuned": {
                "report": {
                    "exact_match": 0.4,
                    "assemblability": 1.0,
                    "functional_equivalence": 0.6,
                    "per_card_exact": 0.86,
                    "by_band": {"B1": {"functional_equivalence": 0.27}},
                }
            }
        }
    }
    candidate = {
        "evaluations": {
            "fine_tuned": {
                "report": {
                    "exact_match": 0.45,
                    "assemblability": 1.0,
                    "functional_equivalence": 0.65,
                    "per_card_exact": 0.84,
                    "by_band": {"B1": {"functional_equivalence": 0.39}},
                }
            }
        }
    }

    comparison = runpod_sweep.compare_to_baseline(candidate, baseline)

    assert comparison["beats_baseline"] is True
    assert comparison["candidate"]["score"] == (0.65, 0.45, 0.84)
