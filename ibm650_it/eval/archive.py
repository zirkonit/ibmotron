from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ibm650_it.dataset.io import load_jsonl, resolve_record_base, resolve_record_path


def archive_failures(
    *,
    reference_index: Path,
    prediction_index: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    references = {str(record["id"]): record for record in load_jsonl(reference_index)}
    predictions = load_jsonl(prediction_index)
    reference_base = resolve_record_base(reference_index)
    prediction_base = prediction_index.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    archived = 0
    by_failure_type: dict[str, int] = {}
    for prediction in predictions:
        if limit is not None and archived >= limit:
            break
        if prediction.get("failure_type") in {None, "exact_match"}:
            continue
        reference = references.get(str(prediction["id"]))
        if reference is None:
            continue
        failure_type = str(prediction["failure_type"])
        by_failure_type[failure_type] = by_failure_type.get(failure_type, 0) + 1
        case_dir = output_dir / f"{archived + 1:04d}_{prediction['id']}"
        case_dir.mkdir(parents=True, exist_ok=True)

        source_path = resolve_record_path(str(reference["source"]["it_text_v1"]), reference_base)
        reference_pit = resolve_record_path(str(reference["reference"]["translate"]["pit_raw_canonical"]), reference_base)
        candidate_pit = resolve_record_path(str(prediction["pit_raw_canonical"]), prediction_base)

        (case_dir / "source.it").write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        (case_dir / "reference_pit.dck").write_bytes(reference_pit.read_bytes())
        candidate_missing = not candidate_pit.exists()
        if not candidate_missing:
            (case_dir / "candidate_pit.dck").write_bytes(candidate_pit.read_bytes())

        assemble = prediction.get("assemble", {})
        run = prediction.get("run", {})
        for target, filename in [
            (assemble.get("console_log"), "assemble_console.log"),
            (assemble.get("stdout_log"), "assemble_stdout.log"),
            (assemble.get("print_log"), "assemble_print.log"),
            (run.get("console_log"), "run_console.log"),
            (run.get("stdout_log"), "run_stdout.log"),
            (run.get("print_log"), "run_print.log"),
        ]:
            if target:
                path = resolve_record_path(str(target), prediction_base)
                if path.exists():
                    (case_dir / filename).write_bytes(path.read_bytes())

        summary = {
            "id": prediction["id"],
            "band": reference["band"],
            "failure_type": failure_type,
            "metrics": prediction.get("metrics", {}),
            "assemblable": prediction.get("assemblable", False),
            "functional": prediction.get("functional", False),
            "candidate_missing": candidate_missing,
            "retrieval": prediction.get("retrieval", {}),
            "assemble": assemble,
            "run": run,
            "evaluator_invariant": prediction.get("evaluator_invariant"),
        }
        (case_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        archived += 1

    manifest = {
        "count": archived,
        "by_failure_type": by_failure_type,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
