from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ibm650_it.dataset.io import load_jsonl, resolve_record_base, resolve_record_path, write_jsonl


REVIEW_CATEGORY_ORDER = [
    "symbolic_tail_or_reservation_drift",
    "wrong_constant_card",
    "wrong_transfer_target",
    "assembles_but_misexecutes_small_diff",
    "malformed_pit",
    "functional_success_exact_failure",
]

REPRESENTATIVE_PLAN = [
    ("symbolic_tail_or_reservation_drift", 3),
    ("assembles_but_misexecutes_small_diff", 3),
    ("functional_success_exact_failure", 2),
]


def _read_cards(path: Path) -> list[str]:
    return path.read_text(encoding="latin-1").splitlines()


def _line_label(card: str) -> str:
    stripped = card.strip().lower()
    if not stripped:
        return ""
    return stripped.split()[0]


def _diff_indices(reference_cards: list[str], candidate_cards: list[str]) -> list[int]:
    diff_indices: list[int] = []
    max_len = max(len(reference_cards), len(candidate_cards))
    for index in range(max_len):
        reference_card = reference_cards[index] if index < len(reference_cards) else None
        candidate_card = candidate_cards[index] if index < len(candidate_cards) else None
        if reference_card != candidate_card:
            diff_indices.append(index)
    return diff_indices


def _classify_review_category(
    *,
    prediction: dict[str, Any],
    reference_cards: list[str],
    candidate_cards: list[str],
) -> str:
    if prediction.get("functional") and not prediction.get("metrics", {}).get("exact_match"):
        return "functional_success_exact_failure"
    if not prediction.get("assemblable", False):
        return "malformed_pit"

    diff_indices = _diff_indices(reference_cards, candidate_cards)
    if not diff_indices:
        return "functional_success_exact_failure"

    labels: list[str] = []
    transfer_like = False
    tail_like = False
    max_len = max(len(reference_cards), len(candidate_cards))
    for index in diff_indices:
        reference_card = reference_cards[index] if index < len(reference_cards) else ""
        candidate_card = candidate_cards[index] if index < len(candidate_cards) else ""
        label = _line_label(reference_card) or _line_label(candidate_card)
        labels.append(label)
        line_text = f"{reference_card}\n{candidate_card}".lower()
        transfer_like = transfer_like or any(token in line_text for token in ["tra", "tze", "tnz", "nze"])
        tail_like = tail_like or index >= max_len - 10 or label.startswith("a")

    if transfer_like:
        return "wrong_transfer_target"
    if labels and len(diff_indices) <= 2 and all(label.startswith("a") for label in labels):
        return "wrong_constant_card"
    if tail_like:
        return "symbolic_tail_or_reservation_drift"
    return "assembles_but_misexecutes_small_diff"


def _copy_if_exists(source: Path | None, destination: Path) -> None:
    if source is None or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _case_sort_key(case: dict[str, Any]) -> tuple[float, int, float, str]:
    return (
        float(case["metrics"]["normalized_edit_distance"]),
        int(case["diff_count"]),
        1.0 - float(case["metrics"]["per_card_exact"]),
        str(case["id"]),
    )


def _archive_case(
    *,
    case: dict[str, Any],
    archive_root: Path,
    ordinal: int,
    reference_index: Path,
    prediction_index: Path,
) -> dict[str, Any]:
    case_dir = archive_root / f"{ordinal:04d}_{case['id']}"
    case_dir.mkdir(parents=True, exist_ok=True)
    reference_base = resolve_record_base(reference_index)
    prediction_base = prediction_index.parent

    reference = case["reference_record"]
    prediction = case["prediction_record"]
    source_path = resolve_record_path(str(reference["source"]["it_text_v1"]), reference_base)
    reference_pit = resolve_record_path(str(reference["reference"]["translate"]["pit_raw_canonical"]), reference_base)
    candidate_pit = resolve_record_path(str(prediction["pit_raw_canonical"]), prediction_base)

    _copy_if_exists(source_path, case_dir / "source.it")
    _copy_if_exists(reference_pit, case_dir / "reference_pit.dck")
    _copy_if_exists(candidate_pit, case_dir / "candidate_pit.dck")
    for target, name in [
        (prediction.get("assemble", {}).get("console_log"), "assemble_console.log"),
        (prediction.get("assemble", {}).get("stdout_log"), "assemble_stdout.log"),
        (prediction.get("assemble", {}).get("print_log"), "assemble_print.log"),
        (prediction.get("run", {}).get("console_log"), "run_console.log"),
        (prediction.get("run", {}).get("stdout_log"), "run_stdout.log"),
        (prediction.get("run", {}).get("print_log"), "run_print.log"),
    ]:
        if target:
            _copy_if_exists(resolve_record_path(str(target), prediction_base), case_dir / name)

    summary = {
        "id": case["id"],
        "band": case["band"],
        "review_category": case["review_category"],
        "failure_type": case["failure_type"],
        "metrics": case["metrics"],
        "diff_count": case["diff_count"],
        "first_diff_index": case["first_diff_index"],
        "functional": case["functional"],
        "assemblable": case["assemblable"],
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_b1_failure_review(
    *,
    reference_index: Path,
    prediction_index: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reference_records = {str(record["id"]): record for record in load_jsonl(reference_index)}
    predictions = load_jsonl(prediction_index)
    reference_base = resolve_record_base(reference_index)
    prediction_base = prediction_index.parent

    cases: list[dict[str, Any]] = []
    for prediction in predictions:
        if prediction.get("band") != "B1":
            continue
        if prediction.get("metrics", {}).get("exact_match"):
            continue
        reference = reference_records.get(str(prediction["id"]))
        if reference is None:
            continue
        reference_pit = resolve_record_path(str(reference["reference"]["translate"]["pit_raw_canonical"]), reference_base)
        candidate_pit = resolve_record_path(str(prediction["pit_raw_canonical"]), prediction_base)
        if not candidate_pit.exists():
            continue
        reference_cards = _read_cards(reference_pit)
        candidate_cards = _read_cards(candidate_pit)
        diff_indices = _diff_indices(reference_cards, candidate_cards)
        review_category = _classify_review_category(
            prediction=prediction,
            reference_cards=reference_cards,
            candidate_cards=candidate_cards,
        )
        cases.append(
            {
                "id": prediction["id"],
                "band": prediction["band"],
                "review_category": review_category,
                "failure_type": prediction.get("failure_type"),
                "metrics": prediction.get("metrics", {}),
                "assemblable": prediction.get("assemblable", False),
                "functional": prediction.get("functional", False),
                "diff_count": len(diff_indices),
                "first_diff_index": diff_indices[0] if diff_indices else None,
                "prediction_record": prediction,
                "reference_record": reference,
            }
        )

    category_counts = {category: 0 for category in REVIEW_CATEGORY_ORDER}
    for case in cases:
        category_counts[case["review_category"]] += 1

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category, target_count in REPRESENTATIVE_PLAN:
        category_cases = sorted(
            [case for case in cases if case["review_category"] == category],
            key=_case_sort_key,
            reverse=True,
        )
        for case in category_cases[:target_count]:
            selected.append(case)
            selected_ids.add(str(case["id"]))

    remaining = sorted(
        [case for case in cases if str(case["id"]) not in selected_ids],
        key=_case_sort_key,
        reverse=True,
    )
    for case in remaining[: max(0, 10 - len(selected))]:
        selected.append(case)
        selected_ids.add(str(case["id"]))

    selected = sorted(selected, key=_case_sort_key, reverse=True)[:10]
    cases_output = [
        {
            key: value
            for key, value in case.items()
            if key not in {"prediction_record", "reference_record"}
        }
        for case in sorted(cases, key=_case_sort_key, reverse=True)
    ]
    write_jsonl(output_root / "cases.jsonl", cases_output)

    archive_root = output_root / "selected_failures"
    if archive_root.exists():
        shutil.rmtree(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    archived = [
        _archive_case(
            case=case,
            archive_root=archive_root,
            ordinal=index,
            reference_index=reference_index,
            prediction_index=prediction_index,
        )
        for index, case in enumerate(selected, start=1)
    ]

    summary = {
        "reference_index": str(reference_index),
        "prediction_index": str(prediction_index),
        "b1_non_exact_count": len(cases),
        "category_counts": category_counts,
        "selected_count": len(archived),
        "selected_categories": [entry["review_category"] for entry in archived],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# B1 Failure Review",
        "",
        f"- B1 non-exact cases: {len(cases)}",
        f"- Selected representative failures: {len(archived)}",
        "",
        "## Category Counts",
    ]
    for category in REVIEW_CATEGORY_ORDER:
        lines.append(f"- `{category}`: {category_counts[category]}")
    lines.append("")
    lines.append("## Selected Cases")
    for entry in archived:
        lines.append(
            f"- `{entry['id']}`: `{entry['review_category']}` · diff_count={entry['diff_count']} · "
            f"first_diff_index={entry['first_diff_index']} · functional={entry['functional']}"
        )
    (output_root / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
