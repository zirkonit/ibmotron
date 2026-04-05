from __future__ import annotations

import json
from pathlib import Path
from typing import Any


NEMOTRON_LORA_TARGET_MODULES = [
    # The Mamba `in_proj` / `out_proj` weights are consumed by custom kernels;
    # PEFT wrappers on those modules break the expected tensor shapes.
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
]


def build_supervised_rows(
    *,
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> list[dict[str, list[int]]]:
    rows: list[dict[str, list[int]]] = []
    eos_text = tokenizer.eos_token or ""

    for record in records:
        prompt_ids = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(record["completion"] + eos_text, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[:max_seq_length]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_seq_length]
        if not any(label != -100 for label in labels):
            continue
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    return rows


def train_hf_qlora(
    *,
    sft_path: Path,
    output_dir: Path,
    config: Any,
    resume_from: Path | None = None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "transformers_qlora backend requires torch, transformers, datasets, peft, accelerate, and bitsandbytes"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in sft_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_examples is not None:
        records = records[:max_examples]
    if not records:
        raise ValueError(f"no training records found in {sft_path}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows = build_supervised_rows(
        records=records,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )
    if not rows:
        raise ValueError("no trainable examples remain after tokenization and truncation")
    tokenized = Dataset.from_list(rows)
    quantized = config.qlora_bits == 4
    bnb_config = None
    if quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto" if quantized else None,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(model)
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=NEMOTRON_LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora)

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        optim="paged_adamw_8bit" if quantized else "adamw_torch",
        weight_decay=config.weight_decay,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
        ),
    )
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from is not None else None)

    adapter_dir = output_dir / "adapter"
    tokenizer_dir = output_dir / "tokenizer"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))

    manifest = {
        "backend": "transformers_qlora",
        "format": "hf_qlora_v1",
        "model_name": config.model_name,
        "qlora_bits": config.qlora_bits,
        "adapter_dir": str(adapter_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "example_count": len(records),
        "sft_path": str(sft_path),
        "target_modules": NEMOTRON_LORA_TARGET_MODULES,
    }
    model_path = output_dir / "model.json"
    model_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "backend": "transformers_qlora",
        "qlora_bits": config.qlora_bits,
        "example_count": len(records),
        "model_path": str(model_path),
        "adapter_dir": str(adapter_dir),
        "tokenizer_dir": str(tokenizer_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
