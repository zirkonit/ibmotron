from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from ibm650_it.dataset.io import load_jsonl, resolve_record_base, resolve_record_path
from ibm650_it.eval.exact_match import compare_pit_files


def build_evaluation_report(*, reference_index: Path, prediction_index: Path) -> dict[str, object]:
    references = load_jsonl(reference_index)
    predictions = {str(entry["id"]): entry for entry in load_jsonl(prediction_index)}
    reference_base = resolve_record_base(reference_index)
    prediction_base = prediction_index.parent

    metrics = []
    failures: Counter[str] = Counter()
    by_band: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    mode = None

    for reference in references:
        prediction = predictions.get(str(reference["id"]))
        if prediction is None:
            continue
        mode = mode or prediction.get("mode")
        exact_metrics = compare_pit_files(
            resolve_record_path(str(reference["reference"]["translate"]["pit_raw_canonical"]), reference_base),
            resolve_record_path(str(prediction["pit_raw_canonical"]), prediction_base),
        )
        assemblable = bool(prediction.get("assemblable", False))
        functional = bool(prediction.get("functional", False))
        failure_type = str(prediction.get("failure_type", "unclassified"))
        failures[failure_type] += 1
        bucket = {
            "exact_match": bool(exact_metrics["exact_match"]),
            "per_card_exact": float(exact_metrics["per_card_exact"]),
            "normalized_edit_distance": float(exact_metrics["normalized_edit_distance"]),
            "assemblable": assemblable,
            "functional": functional,
        }
        metrics.append(bucket)
        by_band[str(reference["band"])].append(bucket)

    total = len(metrics) or 1
    report = {
        "mode": mode,
        "count": len(metrics),
        "exact_match": sum(1 for metric in metrics if metric["exact_match"]) / total,
        "per_card_exact": sum(float(metric["per_card_exact"]) for metric in metrics) / total,
        "normalized_edit_distance": sum(float(metric["normalized_edit_distance"]) for metric in metrics) / total,
        "assemblability": sum(1 for metric in metrics if metric["assemblable"]) / total,
        "functional_equivalence": sum(1 for metric in metrics if metric["functional"]) / total,
        "failure_taxonomy": dict(failures),
        "by_band": {},
    }
    for band, band_metrics in by_band.items():
        band_total = len(band_metrics) or 1
        report["by_band"][band] = {
            "count": len(band_metrics),
            "exact_match": sum(1 for metric in band_metrics if metric["exact_match"]) / band_total,
            "assemblability": sum(1 for metric in band_metrics if metric["assemblable"]) / band_total,
            "functional_equivalence": sum(1 for metric in band_metrics if metric["functional"]) / band_total,
        }
    return report
