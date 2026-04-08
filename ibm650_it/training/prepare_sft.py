from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ibm650_it.dataset.build_records import alpha_normalize_source
from ibm650_it.dataset.io import resolve_record_base, resolve_record_path
from ibm650_it.training.prompt_templates import build_prompt, wrap_pit_completion


def parse_band_repeats(values: list[str] | None) -> dict[str, int]:
    repeats: dict[str, int] = {}
    for value in values or []:
        band, sep, raw_repeat = value.partition("=")
        if sep != "=":
            raise ValueError(f"invalid band repeat {value!r}; expected BAND=N")
        parsed_band = band.strip().upper()
        repeat = int(raw_repeat.strip())
        if repeat < 1:
            raise ValueError(f"band repeat must be >= 1 for {parsed_band}")
        repeats[parsed_band] = repeat
    return repeats


def prepare_sft_examples(
    *,
    dataset_index: Path,
    output_path: Path,
    limit: int | None = None,
    band_repeats: dict[str, int] | None = None,
) -> int:
    records = [json.loads(line) for line in dataset_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        records = records[:limit]
    base_dir = resolve_record_base(dataset_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            source_ref = record["source"]["it_text_v1"]
            target_ref = record["reference"]["translate"]["pit_raw_canonical"]
            source_text = resolve_record_path(source_ref, base_dir).read_text(encoding="utf-8")
            target_text = resolve_record_path(target_ref, base_dir).read_text(encoding="latin-1")
            repeat = max(1, int((band_repeats or {}).get(str(record["band"]).upper(), 1)))
            for repeat_index in range(repeat):
                payload: dict[str, Any] = {
                    "prompt": build_prompt(source_text),
                    "completion": wrap_pit_completion(target_text),
                    "target_text": target_text,
                    "id": record["id"],
                    "source_text": source_text,
                    "source_alpha": alpha_normalize_source(source_text),
                    "band": record["band"],
                    "repeat_index": repeat_index,
                }
                handle.write(json.dumps(payload) + "\n")
                count += 1
    return count
