from ibm650_it.training.infer import HfGenerationSession, _generate_with_hf_model, _hf_inference_runtime


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
        assert kwargs["device"] == "cuda:0"
        assert kwargs["max_new_tokens"] == 7
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
        stop_token_ids=[3, 4],
    )

    completion = _generate_with_hf_model(
        prompt="prompt",
        session=session,
        max_new_tokens=7,
    )

    assert completion == "21,22,23"
