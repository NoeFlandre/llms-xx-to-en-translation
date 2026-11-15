"""Generation backends driven by fake engines: the vLLM request path and the encoder-decoder batch path."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from xx_to_en.backends import (
    NLLB_TARGET,
    GenerationResult,
    PromptInput,
    TransformersBackend,
    VllmBackend,
)
from xx_to_en.config import ModelConfig

EOS = 5  # the fake sequence-to-sequence model "generates" the input followed by this token


def _model() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "name": "toy-model",
            "hf_repo": "org/toy-model",
            "family": "toy",
            "params_b": 1.0,
            "active_params_b": 1.0,
            "license": "apache-2.0",
            "access": "open",
            "backend": "vllm",
            "prompt": "generic_chat",
            "thinking": "not_applicable",
            "dtype": "bfloat16",
            "tensor_parallel": 1,
            "gpu_memory_gb_min": 8,
            "trust_remote_code": False,
            "official_languages": ["fra_Latn"],
        }
    )


class _FakeSamplingParams:
    def __init__(self, *, temperature: float, max_tokens: int, stop: list[str] | None) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop


class _FakeVllmTokenizer:
    """One id per word (the word's length); the BOS id 1 is prepended when requested."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [len(word) for word in text.split()]
        return [1, *ids] if add_special_tokens else ids


class _FakeCompletion:
    def __init__(self, text: str, token_ids: list[int], finish_reason: str | None) -> None:
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason


class _FakeRequestOutput:
    def __init__(self, completion: _FakeCompletion) -> None:
        self.outputs = [completion]


class _FakeLLM:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[Any, Any, bool]] = []

    def get_tokenizer(self) -> _FakeVllmTokenizer:
        return _FakeVllmTokenizer()

    def generate(self, prompts: Any, params: Any, use_tqdm: bool) -> list[_FakeRequestOutput]:
        self.calls.append((prompts, params, use_tqdm))
        return [
            _FakeRequestOutput(_FakeCompletion("<think>plan</think>Bonjour", [11, 12], None)),
            _FakeRequestOutput(_FakeCompletion("raw answer", [13], "length")),
        ]


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-in for the ``vllm`` package, which is only installed on GPU nodes."""
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams)
    )


def test_vllm_generate_tokenizes_prompts_and_builds_results(fake_vllm: None):
    backend = VllmBackend(_model())
    inputs = [PromptInput("le chat", add_special_tokens=True), PromptInput("un chien", stop=("\n",))]
    results = backend.generate(inputs, [16, 8])

    prompts, params, use_tqdm = backend._llm.calls[0]
    assert prompts == [{"prompt_token_ids": [1, 2, 4]}, {"prompt_token_ids": [2, 5]}]
    assert use_tqdm is False
    assert [(p.temperature, p.max_tokens, p.stop) for p in params] == [(0.0, 16, None), (0.0, 8, ["\n"])]
    assert results == [
        GenerationResult(
            text="Bonjour", output_tokens=2, raw_text="<think>plan</think>Bonjour", finish_reason=""
        ),
        GenerationResult(text="raw answer", output_tokens=1, raw_text="raw answer", finish_reason="length"),
    ]


def test_vllm_generate_requires_one_budget_per_prompt(fake_vllm: None):
    backend = VllmBackend(_model())
    with pytest.raises(ValueError, match="shorter than"):
        backend.generate([PromptInput("a"), PromptInput("b")], [8])


class _Padded(dict):
    def to(self, device: str) -> _Padded:
        return self


class _FakeSeq2SeqTokenizer:
    """Ids are word lengths; decoding writes ``w<id>`` per token and drops EOS when asked to skip specials."""

    pad_token_id = 0

    def __init__(self) -> None:
        self.src_lang: str | None = None

    def __call__(self, text: str, return_tensors: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[[len(word) for word in text.split()]])

    def pad(self, features: dict[str, list[list[int]]], return_tensors: str) -> _Padded:
        rows = features["input_ids"]
        width = max(len(row) for row in rows)
        return _Padded({"input_ids": [row + [self.pad_token_id] * (width - len(row)) for row in rows]})

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == NLLB_TARGET
        return 9

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        words = [f"w{token}" for token in token_ids if not (skip_special_tokens and token == EOS)]
        return " " + " ".join(words) + " "


class _FakeSeq2Seq:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, input_ids: list[list[int]], **kwargs: Any) -> list[list[int]]:
        self.calls.append({"input_ids": input_ids, **kwargs})
        return [[*row, EOS] for row in input_ids]


def _transformers_backend(*, batch_size: int = 8, force_target: bool = False) -> TransformersBackend:
    """A backend with fake engines, skipping the real constructor that loads a model onto CUDA."""
    backend = object.__new__(TransformersBackend)
    backend._tokenizer = _FakeSeq2SeqTokenizer()
    backend._model = _FakeSeq2Seq()
    backend._torch = SimpleNamespace(no_grad=nullcontext)
    backend._batch_size = batch_size
    backend._force_target = force_target
    return backend


def test_transformers_batch_decodes_rows_and_drops_padding():
    backend = _transformers_backend()
    results = backend.generate([PromptInput("a"), PromptInput("bb cc")], [3, 3])

    assert results == [
        GenerationResult(text="w1", output_tokens=2, raw_text=" w1 w5 ", finish_reason="stop"),
        GenerationResult(text="w2 w2", output_tokens=3, raw_text=" w2 w2 w5 ", finish_reason="length"),
    ]


def test_transformers_generation_kwargs_force_target_only_when_asked():
    plain = _transformers_backend()
    plain.generate([PromptInput("a")], [4])
    assert plain._model.calls[0]["max_new_tokens"] == 4
    assert "forced_bos_token_id" not in plain._model.calls[0]
    assert plain._model.calls[0]["do_sample"] is False
    assert plain._model.calls[0]["num_beams"] == 1

    forced = _transformers_backend(force_target=True)
    forced.generate([PromptInput("a")], [4])
    assert forced._model.calls[0]["forced_bos_token_id"] == 9


def test_transformers_prompt_src_lang_is_set_on_the_tokenizer():
    backend = _transformers_backend()
    backend.generate([PromptInput("a", src_lang="fra_Latn"), PromptInput("bb")], [3, 3])
    assert backend._tokenizer.src_lang == "fra_Latn"


def test_transformers_generate_batches_and_uses_the_largest_budget_per_batch():
    backend = _transformers_backend(batch_size=2)
    inputs = [PromptInput("a"), PromptInput("bb"), PromptInput("ccc")]
    results = backend.generate(inputs, [4, 9, 2])

    assert [call["max_new_tokens"] for call in backend._model.calls] == [9, 2]
    assert [result.finish_reason for result in results] == ["stop", "stop", "length"]
