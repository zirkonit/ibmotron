from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ibm650_it.training.hf_qlora import train_hf_qlora
from ibm650_it.training.smoke_model import train_smoke_model


@dataclass(slots=True)
class TrainConfig:
    backend: str = "smoke"
    model_name: str = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    qlora_bits: int = 4
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    epochs: int = 3
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.0


def write_train_config(output_path: Path, config: TrainConfig | None = None) -> Path:
    payload = asdict(config or TrainConfig())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def train_model(
    *,
    sft_path: Path,
    output_dir: Path,
    config: TrainConfig | None = None,
    resume_from: Path | None = None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    resolved = config or TrainConfig()
    write_train_config(output_dir / "train_config.json", resolved)
    if resolved.backend == "transformers_qlora":
        summary = train_hf_qlora(
            sft_path=sft_path,
            output_dir=output_dir,
            config=resolved,
            resume_from=resume_from,
            max_examples=max_examples,
        )
        summary["config_path"] = str(output_dir / "train_config.json")
        (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    if resolved.backend != "smoke":
        raise NotImplementedError(f"training backend is not implemented yet: {resolved.backend}")
    summary = train_smoke_model(
        sft_path=sft_path,
        output_dir=output_dir,
        resume_from=resume_from,
        max_examples=max_examples,
    )
    summary["config_path"] = str(output_dir / "train_config.json")
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
