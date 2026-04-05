from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ibm650_it.dataset.io import load_jsonl, resolve_record_base, resolve_record_path
from ibm650_it.eval.exact_match import compare_pit_files


def _average_metric(records: list[dict[str, float | bool]], key: str) -> float:
    total = len(records) or 1
    if key in {"exact_match", "assemblable", "functional"}:
        return sum(1 for record in records if record[key]) / total
    return sum(float(record[key]) for record in records) / total


def _statement_count(record: dict[str, Any], reference_base: Path) -> int:
    source_text = resolve_record_path(str(record["source"]["it_text_v1"]), reference_base).read_text(encoding="utf-8")
    return sum(1 for line in source_text.splitlines() if line.strip() and not line.startswith("+"))


def _statement_count_bucket(count: int) -> str:
    if count <= 3:
        return "1_3"
    if count <= 8:
        return "4_8"
    if count <= 20:
        return "9_20"
    return "21_plus"


def _expr_depth(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    if "lhs" in node and "rhs" in node:
        return 1 + max(_expr_depth(node["lhs"]), _expr_depth(node["rhs"]))
    if "expr" in node:
        return 1 + _expr_depth(node["expr"])
    if "index" in node:
        return _expr_depth(node["index"])
    return 0


def _expr_depth_bucket(record: dict[str, Any], reference_base: Path) -> str:
    ast_ref = str(record["generator"].get("ast_json", ""))
    if not ast_ref:
        return "unknown"
    ast_path = resolve_record_path(ast_ref, reference_base)
    if not ast_path.exists():
        return "unknown"
    payload = json.loads(ast_path.read_text(encoding="utf-8"))
    depth = 0
    for statement in payload.get("statements", []):
        depth = max(depth, _expr_depth(statement))
    if depth <= 1:
        return "1"
    if depth == 2:
        return "2"
    if depth == 3:
        return "3"
    return "4_plus"


def _has_indexed_usage(record: dict[str, Any], reference_base: Path) -> bool:
    ast_ref = str(record["generator"].get("ast_json", ""))
    if not ast_ref:
        return False
    ast_path = resolve_record_path(ast_ref, reference_base)
    if not ast_path.exists():
        return False
    payload = json.loads(ast_path.read_text(encoding="utf-8"))

    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            if "cls" in node and "index" in node:
                index = node["index"]
                if not (isinstance(index, dict) and set(index.keys()) == {"value"}):
                    return True
            return any(walk(value) for value in node.values())
        if isinstance(node, list):
            return any(walk(item) for item in node)
        return False

    return walk(payload)


def _has_loop(record: dict[str, Any], reference_base: Path) -> bool:
    features = set(record["generator"].get("features", []))
    if "iterate" in features:
        return True
    ast_ref = str(record["generator"].get("ast_json", ""))
    if not ast_ref:
        return False
    ast_path = resolve_record_path(ast_ref, reference_base)
    if not ast_path.exists():
        return False
    payload = json.loads(ast_path.read_text(encoding="utf-8"))
    return any("end_stmt" in statement and "loop_var" in statement for statement in payload.get("statements", []))


def _bucket_report(records: list[dict[str, float | bool]]) -> dict[str, float | int]:
    return {
        "count": len(records),
        "exact_match": _average_metric(records, "exact_match"),
        "assemblability": _average_metric(records, "assemblable"),
        "functional_equivalence": _average_metric(records, "functional"),
        "per_card_exact": _average_metric(records, "per_card_exact"),
        "normalized_edit_distance": _average_metric(records, "normalized_edit_distance"),
    }


def build_evaluation_report(*, reference_index: Path, prediction_index: Path) -> dict[str, object]:
    references = load_jsonl(reference_index)
    predictions = {str(entry["id"]): entry for entry in load_jsonl(prediction_index)}
    reference_base = resolve_record_base(reference_index)
    prediction_base = prediction_index.parent

    metrics = []
    failures: Counter[str] = Counter()
    by_band: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    by_statement_count: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    by_expression_depth: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    by_loop_presence: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    by_indexed_usage: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    by_feature: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    mode = None

    for reference in references:
        prediction = predictions.get(str(reference["id"]))
        if prediction is None:
            continue
        mode = mode or prediction.get("mode")
        exact_metrics: dict[str, float | bool]
        reference_pit = resolve_record_path(str(reference["reference"]["translate"]["pit_raw_canonical"]), reference_base)
        candidate_pit = resolve_record_path(str(prediction["pit_raw_canonical"]), prediction_base)
        if candidate_pit.exists():
            exact_metrics = compare_pit_files(reference_pit, candidate_pit)
        else:
            stored_metrics = prediction.get("metrics", {})
            exact_metrics = {
                "exact_match": bool(stored_metrics.get("exact_match", False)),
                "per_card_exact": float(stored_metrics.get("per_card_exact", 0.0)),
                "normalized_edit_distance": float(stored_metrics.get("normalized_edit_distance", 1.0)),
            }
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
        by_statement_count[_statement_count_bucket(_statement_count(reference, reference_base))].append(bucket)
        by_expression_depth[_expr_depth_bucket(reference, reference_base)].append(bucket)
        by_loop_presence["loop" if _has_loop(reference, reference_base) else "no_loop"].append(bucket)
        by_indexed_usage["indexed" if _has_indexed_usage(reference, reference_base) else "scalar_only"].append(bucket)
        features = set(reference["generator"].get("features", []))
        if not features:
            features = {"no_feature_tags"}
        for feature in features:
            by_feature[str(feature)].append(bucket)

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
        "by_band": {key: _bucket_report(value) for key, value in sorted(by_band.items())},
        "by_statement_count": {key: _bucket_report(value) for key, value in sorted(by_statement_count.items())},
        "by_expression_depth": {key: _bucket_report(value) for key, value in sorted(by_expression_depth.items())},
        "by_loop_presence": {key: _bucket_report(value) for key, value in sorted(by_loop_presence.items())},
        "by_indexed_usage": {key: _bucket_report(value) for key, value in sorted(by_indexed_usage.items())},
        "by_feature": {key: _bucket_report(value) for key, value in sorted(by_feature.items())},
    }
    return report


def compare_mode_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = reports.get("zero_shot")
    if baseline is None:
        return {}
    metrics = ["exact_match", "assemblability", "functional_equivalence", "per_card_exact", "normalized_edit_distance"]
    comparison: dict[str, Any] = {}
    for mode, report in reports.items():
        comparison[mode] = {
            metric: {
                "value": report.get(metric),
                "delta_vs_zero_shot": None if mode == "zero_shot" else float(report.get(metric, 0.0)) - float(baseline.get(metric, 0.0)),
            }
            for metric in metrics
        }
    return comparison
