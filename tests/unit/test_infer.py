from ibm650_it.eval.failure_taxonomy import should_attempt_assembly
from ibm650_it.training.infer import (
    HfGenerationSession,
    StopOnTokenSequence,
    _generate_with_hf_model,
    _hf_inference_runtime,
    extract_thinking_trace,
    normalize_completion_text,
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
    assert "<PIT>" not in normalized


def test_normalize_completion_text_handles_truncated_completion_with_multiple_leading_newlines() -> None:
    body_truncated = (
        f"{_FIRST_CARD_INDENT}s0001 00 0000 laaaa\n"
        f"{_FIRST_CARD_INDENT}laaaarala0007       i1 z 3"
    )
    completion = f"<PIT>\n\n{body_truncated}"

    normalized = normalize_completion_text(completion)

    assert normalized.startswith(_FIRST_CARD_INDENT + "s0001 00 0000 laaaa")


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
