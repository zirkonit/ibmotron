from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ibm650_it import REPO_ROOT
from ibm650_it.dataset.build_records import build_record
from ibm650_it.dataset.dedupe import dedupe_by_hash
from ibm650_it.dataset.io import load_jsonl, relativize_record_paths, write_jsonl
from ibm650_it.dataset.provenance import build_provenance
from ibm650_it.dataset.split import split_by_alpha_hash
from ibm650_it.generate.sample_program import generate_band_program, infer_features
from ibm650_it.simh.runner import SimhRunner
from ibm650_it.source.render_it_card80 import render_simh_source_deck
from ibm650_it.source.render_it_text import render_program


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _build_candidate(
    *,
    repo_root: Path,
    band: str,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    runner = SimhRunner(repo_root=repo_root)
    sample_dir = output_root / "staging" / band / f"seed_{seed:06d}"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    program = generate_band_program(band, seed=seed)
    source_path = sample_dir / "source.it"
    source_deck_path = sample_dir / "source.simh.txt"
    ast_path = sample_dir / "ast.json"
    bounds_path = sample_dir / "bounds.json"
    source_path.write_text(render_program(program), encoding="utf-8")
    source_deck_path.write_text(render_simh_source_deck(program), encoding="utf-8")
    ast_path.write_text(json.dumps(asdict(program), indent=2), encoding="utf-8")
    bounds_path.write_text(json.dumps(asdict(program.header), indent=2), encoding="utf-8")
    pipeline = runner.reference_pipeline(source_deck=source_deck_path, output_dir=sample_dir / "pipeline")
    record = build_record(
        band=band,
        seed=seed,
        source_path=source_path,
        pipeline=pipeline,
        header=asdict(program.header),
        ast_path=ast_path,
        bounds_path=bounds_path,
        features=infer_features(program),
        repo_root=repo_root,
    )
    return {
        "band": band,
        "seed": seed,
        "sample_dir": sample_dir,
        "record": record,
    }


def parse_band_counts(entries: list[str] | None, *, total_count: int = 1000) -> dict[str, int]:
    if not entries:
        half = total_count // 2
        return {"B0": half, "B1": total_count - half}
    counts: dict[str, int] = {}
    for entry in entries:
        band, raw_count = entry.split(":", 1)
        counts[band] = int(raw_count)
    return counts


def _rewrite_record_path_prefix(record: dict[str, Any], old_root: Path, new_root: Path) -> dict[str, Any]:
    old_prefix = str(old_root)
    new_prefix = str(new_root)

    def transform(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: transform(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [transform(inner) for inner in value]
        if isinstance(value, str) and value.startswith(old_prefix):
            suffix = os.path.relpath(value, old_prefix)
            return str(new_root / suffix)
        return value

    return transform(record)


def build_splits(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "historical_golden": [],
        "synthetic_train": [],
        "synthetic_dev": [],
        "synthetic_test": [],
        "adversarial_test": [],
    }
    for record in records:
        if record["band"] == "historical_golden":
            buckets["historical_golden"].append(record)
        else:
            buckets[split_by_alpha_hash(record)].append(record)
    return buckets


def write_split_outputs(records: list[dict[str, Any]], output_root: Path) -> dict[str, str]:
    buckets = build_splits(records)
    outputs: dict[str, str] = {}
    for split_name, split_records in buckets.items():
        path = output_root / "splits" / f"{split_name}.jsonl"
        write_jsonl(path, split_records)
        outputs[split_name] = str(path)
    return outputs


def add_historical_golden_records(
    *,
    repo_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    runner = SimhRunner(repo_root=repo_root)
    golden_specs = [
        {
            "name": "it_example_1",
            "source": repo_root / "third_party/simh/I650/sw/it/it_example_1_src.txt",
            "input": repo_root / "third_party/simh/I650/sw/it/it_example_1_data.txt",
        },
        {
            "name": "it_example_2",
            "source": repo_root / "third_party/simh/I650/sw/it/it_example_2_src.txt",
            "input": None,
        },
    ]
    records: list[dict[str, Any]] = []
    for spec in golden_specs:
        sample_dir = output_root / "historical_golden" / spec["name"]
        pipeline = runner.reference_pipeline(
            source_deck=spec["source"],
            input_deck=spec["input"],
            output_dir=sample_dir / "pipeline",
        )
        record = build_record(
            band="historical_golden",
            seed=0,
            source_path=spec["source"],
            pipeline=pipeline,
            header={},
            repo_root=repo_root,
            input_deck=spec["input"],
        )
        record = relativize_record_paths(record, output_root)
        (sample_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
    return records


def build_pilot_corpus(
    *,
    repo_root: Path = REPO_ROOT,
    output_root: Path,
    band_counts: dict[str, int],
    workers: int = 4,
    max_attempts_per_band: int | None = None,
    include_historical_golden: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(repo_root)
    accepted: list[dict[str, Any]] = []
    historical_golden: list[dict[str, Any]] = []
    seen_alpha: set[str] = set()
    accepted_counts: Counter[str] = Counter()
    attempts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    rejections: list[dict[str, Any]] = []
    next_seed = {band: 1 for band in band_counts}
    max_attempts_per_band = max_attempts_per_band or 20 * max(band_counts.values(), default=1)

    if resume:
        for record in load_jsonl(output_root / "index.jsonl"):
            band = str(record["band"])
            if band == "historical_golden":
                historical_golden.append(record)
                continue
            accepted.append(record)
            seen_alpha.add(str(record["hashes"]["alpha_hash"]))
            accepted_counts[band] += 1
            next_seed[band] = max(next_seed.get(band, 1), int(record["seed"]) + 1)
        rejections.extend(load_jsonl(output_root / "rejections.jsonl"))
        summary_path = output_root / "summary.json"
        if summary_path.exists():
            previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            attempts.update(previous_summary.get("attempts_by_band", {}))
            rejection_counts.update(previous_summary.get("rejection_counts", {}))

    def needs_more() -> list[str]:
        return [
            band
            for band, target in band_counts.items()
            if accepted_counts[band] < target and attempts[band] < max_attempts_per_band
        ]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures: dict[Future[dict[str, Any]], tuple[str, int]] = {}
        while True:
            while len(futures) < max(1, workers):
                pending_bands = needs_more()
                if not pending_bands:
                    break
                band = min(pending_bands, key=lambda name: (attempts[name], accepted_counts[name]))
                seed = next_seed[band]
                next_seed[band] += 1
                attempts[band] += 1
                future = executor.submit(
                    _build_candidate,
                    repo_root=repo_root,
                    band=band,
                    seed=seed,
                    output_root=output_root,
                )
                futures[future] = (band, seed)
            if not futures:
                break
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                band, seed = futures.pop(future)
                try:
                    candidate = future.result()
                except Exception as exc:
                    rejection_counts[f"{band}:error:{type(exc).__name__}"] += 1
                    rejections.append({"band": band, "seed": seed, "status": "error", "reason": type(exc).__name__})
                    _remove_tree(output_root / "staging" / band / f"seed_{seed:06d}")
                    continue
                record = candidate["record"]
                sample_dir = Path(candidate["sample_dir"])
                if accepted_counts[band] >= band_counts[band]:
                    rejection_counts[f"{band}:quota_reached"] += 1
                    rejections.append({"band": band, "seed": seed, "status": "quota_reached"})
                    _remove_tree(sample_dir)
                    continue
                alpha_hash = record["hashes"]["alpha_hash"]
                if alpha_hash in seen_alpha:
                    rejection_counts[f"{band}:duplicate_alpha_hash"] += 1
                    rejections.append({"band": band, "seed": seed, "status": "duplicate_alpha_hash"})
                    _remove_tree(sample_dir)
                    continue
                seen_alpha.add(alpha_hash)
                accepted_counts[band] += 1
                final_dir = output_root / "accepted" / band / f"{accepted_counts[band]:04d}_{seed:06d}"
                final_dir.parent.mkdir(parents=True, exist_ok=True)
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                shutil.move(str(sample_dir), str(final_dir))
                record = _rewrite_record_path_prefix(record, sample_dir, final_dir)
                record = relativize_record_paths(record, output_root)
                (final_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                accepted.append(record)

    deduped = dedupe_by_hash(accepted, key="alpha_hash")
    dedupe_removed = len(accepted) - len(deduped)
    accepted = deduped
    _remove_tree(output_root / "staging")
    if include_historical_golden:
        if historical_golden:
            accepted.extend(historical_golden)
        else:
            accepted.extend(add_historical_golden_records(repo_root=repo_root, output_root=output_root))

    index_path = write_jsonl(output_root / "index.jsonl", accepted)
    split_outputs = write_split_outputs(accepted, output_root)
    rejected_path = write_jsonl(output_root / "rejections.jsonl", rejections)
    summary = {
        "accepted_total": len(accepted),
        "accepted_by_band": dict(accepted_counts),
        "attempts_by_band": dict(attempts),
        "rejection_counts": dict(rejection_counts),
        "dedupe_removed": dedupe_removed,
        "target_band_counts": band_counts,
        "provenance": provenance,
        "index_path": str(index_path),
        "rejected_path": str(rejected_path),
        "split_outputs": split_outputs,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    missing = {band: target - accepted_counts[band] for band, target in band_counts.items() if accepted_counts[band] < target}
    if missing:
        raise RuntimeError(f"pilot corpus did not reach requested counts: {missing}")
    return summary
