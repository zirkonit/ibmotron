from pathlib import Path

from ibm650_it.eval.failure_taxonomy import classify_failure
from ibm650_it.training.prepare_sft import parse_band_repeats, prepare_sft_examples, resolve_band_repeats
from ibm650_it.training.prompt_templates import (
    build_chat_messages,
    build_few_shot_chat_messages,
    build_few_shot_prompt,
    build_prompt,
    ensure_pit_wrapped,
    wrap_pit_completion,
)


def test_build_prompt_leaves_assistant_ready_for_wrapped_pit() -> None:
    prompt = build_prompt("+ 0 1 0 3 1730\n0001+ y1 z 2j f\n0002+ t y1 f\n0003+ h ff\n")

    assert prompt.endswith("Assistant:\n")
    assert "<PIT>\n" not in prompt.split("Assistant:\n", 1)[1]


def test_wrap_and_ensure_pit_completion_are_idempotent() -> None:
    raw = "card-1\ncard-2\n"
    wrapped = wrap_pit_completion(raw)

    assert wrapped == "<PIT>\ncard-1\ncard-2\n</PIT>"
    assert ensure_pit_wrapped(raw) == wrapped
    assert ensure_pit_wrapped(wrapped) == wrapped


def test_build_few_shot_prompt_includes_single_wrapped_example_completion() -> None:
    prompt = build_few_shot_prompt(
        "+ 0 1 0 3 1730\n0001+ y1 z 2j f\n0002+ t y1 f\n0003+ h ff\n",
        [
            {
                "source_text": "+ 0 1 0 3 1730\n0001+ y1 z 1j f\n0002+ t y1 f\n0003+ h ff\n",
                "completion": "<PIT>\ncard-1\ncard-2\n</PIT>",
            }
        ],
    )

    assert prompt.count("Assistant:\n<PIT>\ncard-1\ncard-2\n</PIT>") == 1
    assert prompt.endswith("Assistant:\n")


def test_chat_message_builders_wrap_it_and_pit_cleanly() -> None:
    messages = build_chat_messages("+ 0 1 0 3 1730\n0001+ y1 z 2j f\n0002+ t y1 f\n0003+ h ff\n")
    few_shot = build_few_shot_chat_messages(
        "+ 0 1 0 3 1730\n0001+ y1 z 2j f\n0002+ t y1 f\n0003+ h ff\n",
        [
            {
                "source_text": "+ 0 1 0 3 1730\n0001+ y1 z 1j f\n0002+ t y1 f\n0003+ h ff\n",
                "completion": "card-1\ncard-2\n",
            }
        ],
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["content"].startswith("<IT>\n")
    assert messages[1]["content"].endswith("\n</IT>")
    assert few_shot[2]["role"] == "assistant"
    assert few_shot[2]["content"] == "<PIT>\ncard-1\ncard-2\n</PIT>"


def test_prepare_sft_examples_wraps_targets(tmp_path: Path) -> None:
    target = tmp_path / "target.dck"
    target.write_text("card-1\ncard-2\n", encoding="latin-1")
    source = tmp_path / "source.it"
    source.write_text("+ 0 1 0 3 1730\n0001+ y1 z 2j f\n0002+ t y1 f\n0003+ h ff\n", encoding="utf-8")
    index = tmp_path / "index.jsonl"
    index.write_text(
        (
            '{"id":"sample","band":"B0","source":{"it_text_v1":"source.it"},'
            '"reference":{"translate":{"pit_raw_canonical":"target.dck"}}}\n'
        ),
        encoding="utf-8",
    )
    output = tmp_path / "train.jsonl"

    count = prepare_sft_examples(dataset_index=index, output_path=output)

    assert count == 1
    row = output.read_text(encoding="utf-8")
    assert '"completion": "<PIT>\\ncard-1\\ncard-2\\n</PIT>"' in row
    assert '"target_text": "card-1\\ncard-2\\n"' in row


def test_prepare_sft_examples_can_oversample_selected_bands(tmp_path: Path) -> None:
    target = tmp_path / "target.dck"
    target.write_text("card-1\n", encoding="latin-1")
    source = tmp_path / "source.it"
    source.write_text("+ 0 1 0 3 1730\n0001+ y1 z 2j f\n0002+ t y1 f\n0003+ h ff\n", encoding="utf-8")
    index = tmp_path / "index.jsonl"
    index.write_text(
        "\n".join(
            [
                '{"id":"b2","band":"B2","source":{"it_text_v1":"source.it"},'
                '"reference":{"translate":{"pit_raw_canonical":"target.dck"}}}',
                '{"id":"b0","band":"B0","source":{"it_text_v1":"source.it"},'
                '"reference":{"translate":{"pit_raw_canonical":"target.dck"}}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "train.jsonl"

    count = prepare_sft_examples(
        dataset_index=index,
        output_path=output,
        band_repeats={"B2": 3},
    )

    rows = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert count == 4
    assert len(rows) == 4
    assert sum('"id": "b2"' in row for row in rows) == 3
    assert sum('"id": "b0"' in row for row in rows) == 1
    assert '"repeat_index": 2' in rows[2]


def test_prepare_sft_examples_limit_is_band_balanced_on_ordered_index(tmp_path: Path) -> None:
    target = tmp_path / "target.dck"
    target.write_text("card-1\n", encoding="latin-1")
    source = tmp_path / "source.it"
    source.write_text("+ 0 1 0 3 1730\n0001+ y1 z 2j f\n0002+ t y1 f\n0003+ h ff\n", encoding="utf-8")
    index = tmp_path / "index.jsonl"
    index.write_text(
        "\n".join(
            [
                '{"id":"b0_a","band":"B0","source":{"it_text_v1":"source.it"},'
                '"reference":{"translate":{"pit_raw_canonical":"target.dck"}}}',
                '{"id":"b0_b","band":"B0","source":{"it_text_v1":"source.it"},'
                '"reference":{"translate":{"pit_raw_canonical":"target.dck"}}}',
                '{"id":"b1_a","band":"B1","source":{"it_text_v1":"source.it"},'
                '"reference":{"translate":{"pit_raw_canonical":"target.dck"}}}',
                '{"id":"b1_b","band":"B1","source":{"it_text_v1":"source.it"},'
                '"reference":{"translate":{"pit_raw_canonical":"target.dck"}}}',
                '{"id":"b2_a","band":"B2","source":{"it_text_v1":"source.it"},'
                '"reference":{"translate":{"pit_raw_canonical":"target.dck"}}}',
                '{"id":"b2_b","band":"B2","source":{"it_text_v1":"source.it"},'
                '"reference":{"translate":{"pit_raw_canonical":"target.dck"}}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "train.jsonl"

    count = prepare_sft_examples(
        dataset_index=index,
        output_path=output,
        limit=3,
    )

    rows = output.read_text(encoding="utf-8").splitlines()
    assert count == 3
    assert sum('"band": "B0"' in row for row in rows) == 1
    assert sum('"band": "B1"' in row for row in rows) == 1
    assert sum('"band": "B2"' in row for row in rows) == 1


def test_parse_band_repeats_parses_cli_values() -> None:
    assert parse_band_repeats(["B2=2", "b3=3"]) == {"B2": 2, "B3": 3}


def test_resolve_band_repeats_merges_preset_and_cli_overrides() -> None:
    assert resolve_band_repeats(["B4=5"], preset="b45_focus") == {"B3": 2, "B4": 5, "B5": 4}


def test_failure_taxonomy_distinguishes_it_source_echo_from_generic_malformed_pit() -> None:
    failure = classify_failure(
        candidate_cards=[
            "0001+ y1 z 2j f",
            "0002+ t y1 f",
            "0003+ h ff",
        ],
        exact_match=False,
        assemblable=False,
        functional=False,
        assemble_status="assemble_error",
    )

    assert failure == "returned_it_source_instead_of_pit"
