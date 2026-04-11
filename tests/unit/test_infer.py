import json
from pathlib import Path

from ibm650_it.dataset.io import load_jsonl
from ibm650_it.eval.failure_taxonomy import should_attempt_assembly
from ibm650_it.training.infer import (
    HfGenerationSession,
    PreflightTokenBudgetReport,
    StopOnTokenSequence,
    _generate_with_hf_model,
    _hf_inference_runtime,
    _log_preflight_report,
    extract_thinking_trace,
    normalize_completion_text,
    preflight_token_budget,
    run_inference,
)


class DummyCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class DummyTorch:
    bfloat16 = "bf16"
    float32 = "fp32"

    def __init__(self, *, cuda_available: bool) -> None:
        self.cuda = DummyCuda(cuda_available)

    def device(self, name: str) -> str:
        return name


class DummyBatch(dict):
    def to(self, device: str) -> "DummyBatch":
        self["device"] = device
        return self


class DummyIds(list):
    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self[0]))

    @property
    def ndim(self) -> int:
        return 2

    def to(self, device: str) -> "DummyIds":
        self.device = device
        return self


class DummyTokenizerForGenerate:
    pad_token_id = 1
    eos_token_id = 2

    def __call__(self, text: str, return_tensors: str) -> DummyBatch:
        assert text == "prompt"
        assert return_tensors == "pt"
        return DummyBatch({"input_ids": DummyIds([[10, 11]])})

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return ",".join(str(token) for token in tokens)


class DummyModel:
    def generate(self, **kwargs: object) -> list[list[int]]:
        assert kwargs["input_ids"] == [[10, 11]]
        assert kwargs["max_new_tokens"] == 7
        assert kwargs["use_cache"] is True
        return [[10, 11, 21, 22, 23]]


def test_hf_inference_runtime_uses_single_cuda_device_without_auto_sharding() -> None:
    runtime = _hf_inference_runtime(DummyTorch(cuda_available=True))

    assert runtime == {
        "device": "cuda:0",
        "torch_dtype": "bf16",
        "device_map": {"": 0},
    }


def test_hf_inference_runtime_falls_back_to_cpu_float32() -> None:
    runtime = _hf_inference_runtime(DummyTorch(cuda_available=False))

    assert runtime == {
        "device": "cpu",
        "torch_dtype": "fp32",
        "device_map": None,
    }


def test_generate_with_hf_model_uses_session_model_and_strips_prompt_tokens() -> None:
    session = HfGenerationSession(
        tokenizer=DummyTokenizerForGenerate(),
        model=DummyModel(),
        device="cuda:0",
        stop_token_sequences=[[3, 4], [5, 6]],
    )

    completion = _generate_with_hf_model(
        prompt="prompt",
        session=session,
        max_new_tokens=7,
    )

    assert completion == "21,22,23"


def test_generate_with_hf_model_accepts_pre_tokenized_input_ids() -> None:
    session = HfGenerationSession(
        tokenizer=DummyTokenizerForGenerate(),
        model=DummyModel(),
        device="cuda:0",
        stop_token_sequences=[],
    )

    completion = _generate_with_hf_model(
        prompt=None,
        prompt_input_ids=DummyIds([[10, 11]]),
        session=session,
        max_new_tokens=7,
    )

    assert completion == "21,22,23"


def test_extract_thinking_trace_returns_prefix_before_pit() -> None:
    raw_completion = "I will reason first.\nStill thinking.\n<PIT>\ncard-1\n</PIT>"

    assert extract_thinking_trace(raw_completion) == "I will reason first.\nStill thinking."


def test_stop_on_token_sequence_accepts_multiple_variants() -> None:
    criteria = StopOnTokenSequence([[7, 8], [10, 11, 12]])

    class FakeRow(list):
        def tolist(self) -> list[int]:
            return list(self)

    class FakeIds:
        shape = (1, 5)

        def __getitem__(self, idx: int):
            if idx == 0:
                return FakeRow([9, 10, 11, 12, 13])
            raise IndexError

    assert criteria(FakeIds(), None) is False

    class MatchingIds:
        shape = (1, 4)

        def __getitem__(self, idx: int):
            if idx == 0:
                return FakeRow([1, 7, 8, 9])
            raise IndexError

    assert criteria(MatchingIds(), None) is False

    class MatchingTailIds:
        shape = (1, 4)

        def __getitem__(self, idx: int):
            if idx == 0:
                return FakeRow([0, 7, 8, 7])
            raise IndexError

    assert criteria(MatchingTailIds(), None) is False

    class MatchingLongTailIds:
        shape = (1, 4)

        def __getitem__(self, idx: int):
            if idx == 0:
                return FakeRow([9, 10, 11, 12])
            raise IndexError

    assert criteria(MatchingLongTailIds(), None) is True


class _FakeBatchRow(list):
    def tolist(self) -> list[int]:
        return list(self)


class _FakeBatchIds:
    """Shape-aware 2-D tensor stub for StopOnTokenSequence batch tests."""

    def __init__(self, rows: list[list[int]]):
        self._rows = [_FakeBatchRow(row) for row in rows]
        self.shape = (len(rows), max(len(row) for row in rows) if rows else 0)

    def __getitem__(self, idx: int) -> _FakeBatchRow:
        return self._rows[idx]


def test_stop_on_token_sequence_waits_for_all_batch_rows_to_hit_stop() -> None:
    criteria = StopOnTokenSequence([[10, 11, 12]])

    # Row 0 has the stop tail, row 1 does not. The batched criterion should NOT
    # stop yet — otherwise we'd prematurely cut row 1's generation off and the
    # batched path would produce broken predictions for any sequence that isn't
    # the fastest in the batch.
    mixed = _FakeBatchIds([[9, 10, 11, 12], [0, 1, 2, 3]])
    assert criteria(mixed, None) is False

    # All rows hit the stop tail → stop.
    all_hit = _FakeBatchIds([[9, 10, 11, 12], [0, 10, 11, 12]])
    assert criteria(all_hit, None) is True


def test_should_attempt_assembly_skips_obvious_it_echo() -> None:
    assert should_attempt_assembly(
        [
            "0001+ y1 z 1j f",
            "0002+ c1 z y1 s 7j f",
            "0003+ y2 z c1 s 8j f",
            "0004+ t y2 f",
            "0005+ h ff",
        ],
        exact_match=False,
    ) is False


_FIRST_CARD_INDENT = " " * 42


def test_normalize_completion_text_preserves_first_card_indent_when_closed() -> None:
    body = (
        f"{_FIRST_CARD_INDENT}s0001 00 0000 laaaa\n"
        f"{_FIRST_CARD_INDENT}laaaarala0007       i1 z 3\n"
        f"{_FIRST_CARD_INDENT}3         i0002  0002"
    )
    completion = f"<PIT>\n{body}\n</PIT>"

    normalized = normalize_completion_text(completion)

    assert normalized == body
    assert normalized.startswith(_FIRST_CARD_INDENT + "s0001")


def test_normalize_completion_text_preserves_first_card_indent_on_truncated_completion() -> None:
    # Model ran out of new_tokens before emitting </PIT>. The prior implementation
    # called .strip() on the leftover, which ate the 42-space indent on the very
    # first dictionary card and broke SOAP column alignment at runtime.
    body_truncated = (
        f"{_FIRST_CARD_INDENT}s0001 00 0000 laaaa\n"
        f"{_FIRST_CARD_INDENT}laaaarala0007       i1 z 3\n"
        f"{_FIRST_CARD_INDENT}3         i0002"  # cut mid-card, no closing </PIT>
    )
    completion = f"<PIT>\n{body_truncated}"

    normalized = normalize_completion_text(completion)

    assert normalized.startswith(_FIRST_CARD_INDENT + "s0001 00 0000 laaaa")
    assert not normalized.startswith("s0001"), "leading indent must not be stripped on truncated output"


def test_run_inference_limit_is_band_balanced_on_ordered_reference_index(tmp_path: Path) -> None:
    source = tmp_path / "source.it"
    source.write_text("+ 0 1 0 3 1730\n0001+ y1 z 2j f\n0002+ t y1 f\n0003+ h ff\n", encoding="utf-8")
    target = tmp_path / "target.dck"
    target.write_text("card-1\n", encoding="latin-1")
    index = tmp_path / "index.jsonl"
    records = []
    for band in ["B0", "B1", "B2"]:
        for suffix in ["a", "b"]:
            records.append(
                {
                    "id": f"{band.lower()}_{suffix}",
                    "band": band,
                    "source": {"it_text_v1": "source.it"},
                    "reference": {"translate": {"pit_raw_canonical": "target.dck"}},
                    "generator": {"features": ["punch"]},
                }
            )
    index.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    summary = run_inference(
        reference_index=index,
        output_dir=tmp_path / "eval",
        mode="zero_shot",
        limit=3,
        eval_mode="skip",
    )

    predictions = load_jsonl(Path(summary["prediction_index"]))
    assert len(predictions) == 3
    assert {str(prediction["band"]) for prediction in predictions} == {"B0", "B1", "B2"}


def test_normalize_completion_text_handles_truncated_completion_with_multiple_leading_newlines() -> None:
    body_truncated = (
        f"{_FIRST_CARD_INDENT}s0001 00 0000 laaaa\n"
        f"{_FIRST_CARD_INDENT}laaaarala0007       i1 z 3"
    )
    completion = f"<PIT>\n\n{body_truncated}"

    normalized = normalize_completion_text(completion)

    assert normalized.startswith(_FIRST_CARD_INDENT + "s0001 00 0000 laaaa")


def test_preflight_token_budget_reports_ok_when_all_under_cap() -> None:
    report = preflight_token_budget(
        reference_tokens=[("b0-1", 480), ("b0-2", 692), ("b1-1", 1005)],
        max_new_tokens=1536,
    )
    assert report.ok is True
    assert report.over_budget == []
    assert report.largest_ref_tokens == 1005
    assert report.sample_size == 3


def test_preflight_token_budget_flags_over_budget_cases() -> None:
    # Reproduces the 20260406 stage_2k distribution: B0 fits, B2/B3 exceed 1024
    report = preflight_token_budget(
        reference_tokens=[
            ("b0", 600),
            ("b1-small", 970),
            ("b1-big", 1030),
            ("b2", 1050),
            ("b3", 1154),
        ],
        max_new_tokens=1024,
    )
    assert report.ok is False
    assert report.over_budget_count == 3
    assert report.largest_ref_tokens == 1154
    assert {rid for rid, _ in report.over_budget} == {"b1-big", "b2", "b3"}


def test_preflight_token_budget_empty_sample_is_ok() -> None:
    report = preflight_token_budget(reference_tokens=[], max_new_tokens=1536)
    assert report.ok is True
    assert report.sample_size == 0
    assert report.largest_ref_tokens == 0


def test_log_preflight_report_ok_path(capsys) -> None:
    report = PreflightTokenBudgetReport(
        max_new_tokens=1536,
        sample_size=10,
        over_budget=[],
        largest_ref_tokens=1200,
    )
    _log_preflight_report(report)
    err = capsys.readouterr().err
    assert "ok" in err
    assert "1536" in err
    assert "1200" in err


def test_log_preflight_report_warn_path_names_offenders(capsys) -> None:
    report = PreflightTokenBudgetReport(
        max_new_tokens=1024,
        sample_size=5,
        over_budget=[("b2-a", 1050), ("b2-b", 1049), ("b3-a", 1154)],
        largest_ref_tokens=1154,
    )
    _log_preflight_report(report)
    err = capsys.readouterr().err
    assert "WARN" in err
    assert "3/5" in err
    assert "b2-a:1050" in err
    assert "1154" in err
    assert "truncated" in err


def test_should_attempt_assembly_keeps_symbolic_like_outputs() -> None:
    assert should_attempt_assembly(
        [
            "s0001 00 0000 laaaa 0000 0000",
            "s0002 69 1995 xbbbb 0000 0000",
            "s0003 65 1996 xcccc 0000 0000",
            "s0004 24 1997 xdddd 0000 0000",
            "s0005 69 1998 xeeee 0000 0000",
            "s0006 24 1999 xffff 0000 0000",
            "s0007 65 1991 xgggg 0000 0000",
            "s0008 69 1992 xhhhh 0000 0000",
            "s0009 24 1993 xiiii 0000 0000",
            "s0010 69 1994 xjjjj 0000 0000",
        ],
        exact_match=False,
    ) is True
