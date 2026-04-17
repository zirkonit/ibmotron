from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ibm650_it import REPO_ROOT
from ibm650_it.dataset.provenance import build_provenance
from ibm650_it.simh.deckio import deck_hash
from ibm650_it.simh.runner import PipelineResult
from ibm650_it.source.normalize_it import normalize_it_text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def alpha_normalize_source(source_text: str) -> str:
    lines = []
    stmt_map: dict[str, str] = {}
    var_maps: dict[str, dict[str, str]] = {"i": {}, "y": {}, "c": {}}
    var_next: dict[str, int] = {"i": 1, "y": 1, "c": 1}
    next_stmt = 1
    for line in source_text.splitlines():
        if line.startswith("+"):
            lines.append(line)
            continue
        parts = line.split()
        if parts:
            stmt_token = parts[0]
            match = re.fullmatch(r"(\d{1,4})\+?", stmt_token)
            if match:
                stmt = match.group(1)
                stmt_map.setdefault(stmt, f"{next_stmt:04d}")
                if stmt_map[stmt] == f"{next_stmt:04d}":
                    next_stmt += 1
                parts[0] = f"{stmt_map[stmt]}+"
        normalized_line = " ".join(parts)

        def repl(match: re.Match[str]) -> str:
            cls = match.group(1)
            token = match.group(0)
            if token not in var_maps[cls]:
                var_maps[cls][token] = f"{cls}{var_next[cls]}"
                var_next[cls] += 1
            return var_maps[cls][token]

        normalized_line = re.sub(r"\b([iyc])(\d+)\b", repl, normalized_line)
        lines.append(normalized_line)
    return "\n".join(lines) + "\n"


def shape_normalize_source(source_text: str) -> str:
    lines = []
    for line in source_text.splitlines():
        if line.startswith("+"):
            lines.append("+ header")
            continue
        parts = line.split()
        if parts:
            if re.fullmatch(r"\d{1,4}\+?", parts[0]):
                parts[0] = "stmt+"
        normalized_line = " ".join(parts)
        normalized_line = re.sub(r"\b[iyc]\d+\b", "var", normalized_line)
        normalized_line = re.sub(r"\b\d+j\b", "float", normalized_line)
        normalized_line = re.sub(r"\b\d+\b", "int", normalized_line)
        lines.append(normalized_line)
    return "\n".join(lines) + "\n"


def stable_record_id(*, surface_hash: str, alpha_hash: str, pit_hash: str, band: str, seed: int) -> str:
    payload = "\n".join([band, str(seed), surface_hash, alpha_hash, pit_hash]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_record(
    *,
    band: str,
    seed: int,
    source_path: Path,
    pipeline: PipelineResult,
    header: dict[str, object] | None = None,
    ast_path: Path | None = None,
    bounds_path: Path | None = None,
    features: list[str] | None = None,
    repo_root: Path = REPO_ROOT,
    input_deck: Path | None = None,
) -> dict[str, object]:
    source_text = source_path.read_text(encoding="latin-1")
    try:
        canonical_source = normalize_it_text(source_text)
    except Exception:
        canonical_source = source_text
    surface_hash = _sha256(canonical_source)
    alpha_hash = _sha256(alpha_normalize_source(canonical_source))
    shape_hash = _sha256(shape_normalize_source(canonical_source))
    pit_hash = deck_hash(pipeline.translate.pit_raw_canonical or pipeline.translate.pit_raw)  # type: ignore[arg-type]
    provenance = build_provenance(repo_root)
    record_id = stable_record_id(
        surface_hash=surface_hash,
        alpha_hash=alpha_hash,
        pit_hash=pit_hash,
        band=band,
        seed=seed,
    )
    return {
        "id": record_id,
        "band": band,
        "seed": seed,
        "runtime_package": "P1",
        "source_format": "it_text_v1",
        "source": {
            "it_text_v1": str(source_path),
            "header": header or {},
        },
        "hashes": {
            "surface_hash": surface_hash,
            "alpha_hash": alpha_hash,
            "shape_hash": shape_hash,
            "pit_hash": pit_hash,
        },
        "reference": {
            "translate": {
                "status": pipeline.translate.status,
                "upper_acc": pipeline.translate.upper_acc,
                "pit_raw": str(pipeline.translate.pit_raw),
                "pit_raw_canonical": str(pipeline.translate.pit_raw_canonical),
                "reservation_cards": str(pipeline.split.reservation_cards),
                "translation_body": str(pipeline.split.translation_body),
                "console_log": str(pipeline.translate.console_log),
                "stdout_log": str(pipeline.translate.stdout_log),
                "print_log": str(pipeline.translate.print_log),
            },
            "assemble": {
                "status": pipeline.assemble.status,
                "pit_phase2_input_p1": str(pipeline.assemble.pit_phase2_input_p1),
                "soap_output": str(pipeline.assemble.soap_output),
                "console_log": str(pipeline.assemble.console_log),
                "stdout_log": str(pipeline.assemble.stdout_log),
                "print_log": str(pipeline.assemble.print_log),
            },
            "run": {
                "status": pipeline.run.status,
                "input_deck": str(input_deck) if input_deck is not None else None,
                "spit_p1": str(pipeline.spit.spit_p1),
                "output_deck": str(pipeline.run.output_deck) if pipeline.run.output_deck is not None else None,
                "console_log": str(pipeline.run.console_log),
                "stdout_log": str(pipeline.run.stdout_log),
                "print_log": str(pipeline.run.print_log),
            },
        },
        "generator": {
            "ast_json": str(ast_path) if ast_path is not None else "",
            "bounds_json": str(bounds_path) if bounds_path is not None else "",
            "features": features or [],
        },
        "provenance": {
            **provenance,
        },
    }
