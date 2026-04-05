from pathlib import Path

from ibm650_it.eval.failure_taxonomy import classify_failure
from ibm650_it.training.prepare_sft import prepare_sft_examples
from ibm650_it.training.prompt_templates import build_few_shot_prompt, build_prompt, ensure_pit_wrapped, wrap_pit_completion


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
