"""Generation backends. Greedy decoding only.

Two implementations, both importing their heavy dependencies lazily so that this module imports
on a laptop without the ``gpu`` extra:

- ``VllmBackend`` for decoder-only models (served by vLLM).
- ``TransformersBackend`` for encoder-decoder models (NLLB-200, MADLAD-400), run with ``transformers``.

Each prompt says whether the tokenizer should add special tokens (PLaMo needs its BOS) and which
stop strings end generation. Chat prompts are already rendered and must not get a second BOS.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from xx_to_en.config import ModelConfig

# Markers that close a reasoning block. Anything before the last marker is discarded.
THINKING_END_MARKERS = ("<channel|>", "</think>")
NLLB_TARGET = "eng_Latn"


@dataclass(frozen=True)
class PromptInput:
    """One prompt.

    ``src_lang`` is used only by encoder-decoder models that take language tags (NLLB).
    ``add_special_tokens`` must be True for raw prompts whose format relies on the tokenizer's BOS.
    ``stop`` are strings that end generation (vLLM only).
    """

    text: str
    src_lang: str | None = None
    add_special_tokens: bool = False
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    text: str
    output_tokens: int
    raw_text: str = ""
    finish_reason: str = ""


class Backend(Protocol):
    def render_chat(self, messages: Sequence[dict[str, Any]], chat_kwargs: dict[str, Any]) -> str: ...

    def count_tokens(self, text: str) -> int: ...

    def generate(
        self, inputs: Sequence[PromptInput], max_new_tokens: Sequence[int]
    ) -> list[GenerationResult]: ...


def clean_output(text: str) -> str:
    """Drop reasoning blocks that some chat models emit even when thinking is disabled."""
    for marker in THINKING_END_MARKERS:
        if marker in text:
            text = text.rsplit(marker, 1)[1]
    return text.strip()


class VllmBackend:
    """Decoder-only models served by vLLM with greedy decoding."""

    def __init__(
        self, model: ModelConfig, *, seed: int = 0, max_model_len: int = 4096, dtype: str | None = None
    ) -> None:
        from vllm import LLM  # ty: ignore[import-not-found]

        self._llm = LLM(
            model=model.hf_repo,
            dtype=dtype or model.dtype,
            tensor_parallel_size=model.tensor_parallel,
            trust_remote_code=model.trust_remote_code,
            seed=seed,
            max_model_len=max_model_len,
        )
        self._tokenizer = self._llm.get_tokenizer()

    def render_chat(self, messages: Sequence[dict[str, Any]], chat_kwargs: dict[str, Any]) -> str:
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
            {
                "prompt_token_ids": self._tokenizer.encode(
                    item.text, add_special_tokens=item.add_special_tokens
                )
            }
            for item in inputs
        ]
        params = [
            SamplingParams(temperature=0.0, max_tokens=budget, stop=list(item.stop) or None)
            for item, budget in zip(inputs, max_new_tokens, strict=True)
        ]
        outputs = self._llm.generate(prompts, params, use_tqdm=False)
        results = []
        for output in outputs:
            completion = output.outputs[0]
            results.append(
                GenerationResult(
                    text=clean_output(completion.text),
                    output_tokens=len(completion.token_ids),
                    raw_text=completion.text,
                    finish_reason=completion.finish_reason or "",
                )
            )
        return results


class TransformersBackend:
    """Encoder-decoder models (NLLB, MADLAD) with greedy decoding.

    NLLB needs the target language forced as the first decoder token; MADLAD takes its target from a
    text prefix instead, so ``force_target_lang`` is False for it.
    """

    def __init__(
        self,
        model: ModelConfig,
        *,
        seed: int = 0,
        batch_size: int = 32,
        force_target_lang: bool = False,
        dtype: str | None = None,
    ) -> None:
        import torch  # ty: ignore[import-not-found]
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # ty: ignore[import-not-found]

        torch.manual_seed(seed)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model.hf_repo, trust_remote_code=model.trust_remote_code
        )
        dtype_obj = getattr(torch, dtype or model.dtype)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model.hf_repo, torch_dtype=dtype_obj, trust_remote_code=model.trust_remote_code
        ).to("cuda")
        self._model.eval()
        self._batch_size = batch_size
        self._force_target = force_target_lang

    def render_chat(self, messages: Sequence[dict[str, Any]], chat_kwargs: dict[str, Any]) -> str:
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
        encoded = []
        for item in batch:
            if item.src_lang is not None:
                self._tokenizer.src_lang = item.src_lang
            encoded.append(self._tokenizer(item.text, return_tensors="pt").input_ids[0])
        padded = self._tokenizer.pad({"input_ids": encoded}, return_tensors="pt").to("cuda")
        kwargs: dict[str, Any] = {"do_sample": False, "num_beams": 1, "max_new_tokens": budget}
        if self._force_target:
            kwargs["forced_bos_token_id"] = self._tokenizer.convert_tokens_to_ids(NLLB_TARGET)
        with self._torch.no_grad():
            output = self._model.generate(**padded, **kwargs)
        results = []
        for row in output:
            token_ids = [int(token) for token in row if int(token) != self._tokenizer.pad_token_id]
            text = self._tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            raw = self._tokenizer.decode(token_ids, skip_special_tokens=False)
            finish = "length" if len(token_ids) >= budget else "stop"
            results.append(
                GenerationResult(text=text, output_tokens=len(token_ids), raw_text=raw, finish_reason=finish)
            )
        return results
