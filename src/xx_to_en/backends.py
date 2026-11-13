"""Generation backends. Greedy decoding only.

Two implementations, both importing their heavy dependencies lazily so that this module imports
on a laptop without the ``gpu`` extra:

- ``VllmBackend`` for decoder-only chat models (served by vLLM).
- ``TransformersBackend`` for encoder-decoder models (NLLB-200, MADLAD-400), run with ``transformers``.

Prompt text is tokenised here, so the backend controls special tokens exactly once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from xx_to_en.config import ModelConfig


@dataclass(frozen=True)
class PromptInput:
    """One prompt. ``src_lang`` is only used by encoder-decoder models that take language tags."""

    text: str
    src_lang: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    output_tokens: int


class Backend(Protocol):
    def render_chat(self, messages: Sequence[dict[str, str]], chat_kwargs: dict[str, Any]) -> str: ...

    def count_tokens(self, text: str) -> int: ...

    def generate(
        self, inputs: Sequence[PromptInput], max_new_tokens: Sequence[int]
    ) -> list[GenerationResult]: ...


class VllmBackend:
    """Decoder-only chat models served by vLLM with greedy decoding."""

    def __init__(self, model: ModelConfig, *, seed: int = 0, max_model_len: int = 4096) -> None:
        from vllm import LLM  # ty: ignore[import-not-found]

        self._llm = LLM(
            model=model.hf_repo,
            dtype=model.dtype,
            tensor_parallel_size=model.tensor_parallel,
            trust_remote_code=model.trust_remote_code,
            seed=seed,
            max_model_len=max_model_len,
        )
        self._tokenizer = self._llm.get_tokenizer()

    def render_chat(self, messages: Sequence[dict[str, str]], chat_kwargs: dict[str, Any]) -> str:
        return str(
            self._tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True, **chat_kwargs
            )
        )

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def generate(
        self, inputs: Sequence[PromptInput], max_new_tokens: Sequence[int]
    ) -> list[GenerationResult]:
        from vllm import SamplingParams  # ty: ignore[import-not-found]

        prompts = [
            {"prompt_token_ids": self._tokenizer.encode(item.text, add_special_tokens=False)}
            for item in inputs
        ]
        params = [SamplingParams(temperature=0.0, max_tokens=budget) for budget in max_new_tokens]
        outputs = self._llm.generate(prompts, params, use_tqdm=False)
        results = []
        for output in outputs:
            completion = output.outputs[0]
            results.append(
                GenerationResult(text=completion.text.strip(), output_tokens=len(completion.token_ids))
            )
        return results


class TransformersBackend:
    """Encoder-decoder models (NLLB, MADLAD) with greedy decoding."""

    def __init__(
        self, model: ModelConfig, *, seed: int = 0, batch_size: int = 32, target_lang: str = "eng_Latn"
    ) -> None:
        import torch  # ty: ignore[import-not-found]
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # ty: ignore[import-not-found]

        torch.manual_seed(seed)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model.hf_repo, trust_remote_code=model.trust_remote_code
        )
        dtype = getattr(torch, model.dtype)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model.hf_repo, torch_dtype=dtype, trust_remote_code=model.trust_remote_code
        ).to("cuda")
        self._model.eval()
        self._batch_size = batch_size
        self._target_lang = target_lang

    def render_chat(self, messages: Sequence[dict[str, str]], chat_kwargs: dict[str, Any]) -> str:
        raise NotImplementedError("encoder-decoder models do not use chat templates")

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def generate(
        self, inputs: Sequence[PromptInput], max_new_tokens: Sequence[int]
    ) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        for start in range(0, len(inputs), self._batch_size):
            batch = list(inputs[start : start + self._batch_size])
            budget = max(max_new_tokens[start : start + self._batch_size])
            results.extend(self._generate_batch(batch, budget))
        return results

    def _generate_batch(self, batch: list[PromptInput], budget: int) -> list[GenerationResult]:
        forced = self._tokenizer.convert_tokens_to_ids(self._target_lang)
        encoded = []
        for item in batch:
            if item.src_lang is not None:
                self._tokenizer.src_lang = item.src_lang
            encoded.append(self._tokenizer(item.text, return_tensors="pt").input_ids[0])
        padded = self._tokenizer.pad({"input_ids": encoded}, return_tensors="pt").to("cuda")
        with self._torch.no_grad():
            output = self._model.generate(
                **padded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=budget,
                forced_bos_token_id=forced,
            )
        results = []
        for row in output:
            token_ids = [int(token) for token in row if int(token) != self._tokenizer.pad_token_id]
            text = self._tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            results.append(GenerationResult(text=text, output_tokens=len(token_ids)))
        return results
