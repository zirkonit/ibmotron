from ibm650_it.training.hf_qlora import (
    DEFAULT_TRANSFORMERS_QLORA_MODEL,
    NEMOTRON_LORA_TARGET_MODULES,
    QWEN_LORA_TARGET_MODULES,
    build_supervised_rows,
    resolve_lora_target_modules,
    trainer_processing_kwargs,
)


class DummyTokenizer:
    eos_token = "|"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


def test_build_supervised_rows_masks_prompt_tokens() -> None:
    rows = build_supervised_rows(
        records=[
            {
                "prompt": "abc",
                "completion": "xy",
            }
        ],
        tokenizer=DummyTokenizer(),
        max_seq_length=32,
    )

    assert len(rows) == 1
    row = rows[0]
    prompt_len = 3
    completion_ids = [ord("x"), ord("y"), ord("|")]

    assert row["input_ids"] == [ord("a"), ord("b"), ord("c"), *completion_ids]
    assert row["labels"] == [-100, -100, -100, *completion_ids]
    assert row["attention_mask"] == [1] * len(row["input_ids"])


def test_build_supervised_rows_drops_examples_without_completion_tokens_after_truncation() -> None:
    rows = build_supervised_rows(
        records=[
            {
                "prompt": "abcdef",
                "completion": "xy",
            }
        ],
        tokenizer=DummyTokenizer(),
        max_seq_length=3,
    )

    assert rows == []


def test_nemotron_target_modules_avoid_mamba_kernel_wrapped_projections() -> None:
    assert "in_proj" not in NEMOTRON_LORA_TARGET_MODULES
    assert "out_proj" not in NEMOTRON_LORA_TARGET_MODULES


def test_qwen_target_modules_cover_attention_and_mlp_projections() -> None:
    assert QWEN_LORA_TARGET_MODULES == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_resolve_lora_target_modules_prefers_qwen_defaults() -> None:
    class DummyQwenConfig:
        model_type = "qwen3"

    assert resolve_lora_target_modules(
        model_name=DEFAULT_TRANSFORMERS_QLORA_MODEL,
        model_config=DummyQwenConfig(),
    ) == QWEN_LORA_TARGET_MODULES


def test_resolve_lora_target_modules_keeps_nemotron_mamba_safe_list() -> None:
    class DummyNemotronConfig:
        model_type = "nemotron_h"

    assert resolve_lora_target_modules(
        model_name="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        model_config=DummyNemotronConfig(),
    ) == NEMOTRON_LORA_TARGET_MODULES


def test_trainer_processing_kwargs_prefers_processing_class_when_available() -> None:
    class DummyTrainer:
        def __init__(self, *, model: object, processing_class: object | None = None) -> None:
            del model, processing_class

    tokenizer = object()

    assert trainer_processing_kwargs(
        trainer_cls=DummyTrainer,
        tokenizer=tokenizer,
    ) == {"processing_class": tokenizer}


def test_trainer_processing_kwargs_falls_back_to_tokenizer_for_older_transformers() -> None:
    class DummyTrainer:
        def __init__(self, *, model: object, tokenizer: object | None = None) -> None:
            del model, tokenizer

    tokenizer = object()

    assert trainer_processing_kwargs(
        trainer_cls=DummyTrainer,
        tokenizer=tokenizer,
    ) == {"tokenizer": tokenizer}


def test_trainer_processing_kwargs_returns_empty_when_neither_argument_exists() -> None:
    class DummyTrainer:
        def __init__(self, *, model: object) -> None:
            del model

    assert trainer_processing_kwargs(
        trainer_cls=DummyTrainer,
        tokenizer=object(),
    ) == {}
