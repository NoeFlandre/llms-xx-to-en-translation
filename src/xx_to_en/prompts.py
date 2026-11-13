"""Prompt templates, keyed by the ``prompt`` field of each model config.

Bump ``PROMPT_VERSION`` whenever any template changes: it is part of every cache key.
Each template is either a chat message list rendered by the model's own chat template, or a
raw string for models with a documented plain-text format.
"""

from __future__ import annotations

from collections.abc import Callable

from xx_to_en.backends import Backend, PromptInput
from xx_to_en.benchmarks import Segment

PROMPT_VERSION = "1"

INSTRUCTION = "Translate the following text into English. Output only the translation.\n\n"

Builder = Callable[[Segment, Backend], PromptInput]


def _generic_chat(segment: Segment, backend: Backend) -> PromptInput:
    messages = [{"role": "user", "content": INSTRUCTION + segment.source}]
    return PromptInput(text=backend.render_chat(messages, {}))


BUILDERS: dict[str, Builder] = {
    "generic_chat": _generic_chat,
}


def template_names() -> frozenset[str]:
    return frozenset(BUILDERS)


def make_prompt(template: str, segment: Segment, backend: Backend) -> PromptInput:
    """Render the prompt for one segment with the named template."""
    try:
        builder = BUILDERS[template]
    except KeyError as error:
        raise ValueError(f"no prompt template named {template!r}") from error
    return builder(segment, backend)
