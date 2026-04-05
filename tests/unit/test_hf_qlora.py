from ibm650_it.training.hf_qlora import NEMOTRON_LORA_TARGET_MODULES, build_supervised_rows


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
