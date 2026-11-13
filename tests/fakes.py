"""Deterministic stand-ins for generation backends, used by runner and prompt tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xx_to_en.backends import GenerationResult, PromptInput


class FakeBackend:
    """Returns the prompt's last paragraph (the source) in upper case, cut to the token budget.

    Records every call so tests can check how many generations were actually requested, and keeps
    the chat messages of the last ``render_chat`` call so prompt builders can be checked.
    """

    def __init__(self) -> None:
        self.generate_calls: list[int] = []
        self.last_messages: list[dict[str, Any]] = []

    def render_chat(self, messages: Sequence[dict[str, Any]], chat_kwargs: dict[str, Any]) -> str:
        self.last_messages = [dict(message) for message in messages]
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(
        self, inputs: Sequence[PromptInput], max_new_tokens: Sequence[int]
    ) -> list[GenerationResult]:
        self.generate_calls.append(len(inputs))
        results = []
        for item, budget in zip(inputs, max_new_tokens, strict=True):
            source = item.text.split("\n\n")[-1]
            words = source.upper().split()[:budget]
            results.append(GenerationResult(text=" ".join(words), output_tokens=len(words)))
        return results
