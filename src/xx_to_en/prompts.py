"""Prompt templates, keyed by the ``prompt`` field of each model config.

Bump ``PROMPT_VERSION`` whenever any template changes: it is part of every cache key.

Sources (see ``configs/models/*.yaml`` and the model cards):
- ``translategemma``: the model's own chat template (``chat_template.jinja``): user content is a
  list with ``source_lang_code``, ``target_lang_code`` and ``text``; the template renders the prompt.
  Corpus languages without an ISO 639-1 code (no name in that template) get ``generic_chat`` instead.
- ``plamo3_translate``: the PFML3 format printed in the PLaMo 3 Translate card.
- ``hy_mt2``: the "Default Translation" instruction from the Hy-MT2 card; markdown was removed.
- ``nllb_tag``: source language tag on the tokenizer; target forced to eng_Latn by the backend.
- ``madlad_tag``: ``<2xx>`` target prefix from the MADLAD card.
- ``generic_chat``: no official translation prompt exists for these models; a plain instruction
  is rendered through the model's own chat template.
- ``hy_mt15``: the instruction from the HY-MT1.5 model card, sent as one user message.
- ``qwen_nothink``: ``generic_chat`` with ``enable_thinking=False``: the Qwen template then emits an
  empty think block, without it the model reasons in the output and runs out of tokens.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from xx_to_en.backends import Backend, PromptInput
from xx_to_en.benchmarks import Segment

PROMPT_VERSION = "5"

GENERIC_INSTRUCTION = "Translate the following text into English. Output only the translation.\n\n"
HY_MT2_INSTRUCTION = (
    "Translate the following text into English. Note that you should only output the translated "
    "result without any additional explanation:\n\n"
)
PLAMO_TEMPLATE = (
    "<|plamo:tag|>dataset<|plamo:msg|>translation"
    "<|plamo:tag|>input<|plamo:msg|>{text}"
    "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
)
PLAMO_STOP = ("<|plamo:tag|>", "<|plamo:bos|>")
MADLAD_TARGET_TAG = "<2en> "
HY_MT15_INSTRUCTION = "Translate the following segment into English, without additional explanation.\n\n"

IsoLookup = Callable[[str], str | None]
Builder = Callable[[Segment, Backend, IsoLookup], PromptInput]


def _generic_chat(segment: Segment, backend: Backend, iso: IsoLookup) -> PromptInput:
    messages = [{"role": "user", "content": GENERIC_INSTRUCTION + segment.source}]
    return PromptInput(text=backend.render_chat(messages, {}))


def _qwen_nothink(segment: Segment, backend: Backend, iso: IsoLookup) -> PromptInput:
    """Generic instruction with Qwen's thinking turned off; otherwise the model writes its reasoning out."""
    messages = [{"role": "user", "content": GENERIC_INSTRUCTION + segment.source}]
    return PromptInput(text=backend.render_chat(messages, {"enable_thinking": False}))


def _hy_mt15(segment: Segment, backend: Backend, iso: IsoLookup) -> PromptInput:
    messages = [{"role": "user", "content": HY_MT15_INSTRUCTION + segment.source}]
    return PromptInput(text=backend.render_chat(messages, {}))


def _translategemma_generic(segment: Segment) -> PromptInput:
    """The generic instruction in TranslateGemma's own turn format, for languages its template does not name.

    Its chat template only accepts the structured content (with language codes), so this turn is
    written out directly; it is the template's output for a user turn with the generic instruction.
    """
    turn = GENERIC_INSTRUCTION + segment.source
    text = f"<bos><start_of_turn>user\n{turn}<end_of_turn>\n<start_of_turn>model\n"
    return PromptInput(text=text)


def _translategemma(segment: Segment, backend: Backend, iso: IsoLookup) -> PromptInput:
    source_code = iso(segment.corpus_lang)
    if not source_code:
        return _translategemma_generic(segment)
    content = [
        {
            "type": "text",
            "source_lang_code": source_code,
            "target_lang_code": "en",
            "text": segment.source,
        }
    ]
    return PromptInput(text=backend.render_chat([{"role": "user", "content": content}], {}))


def _plamo3(segment: Segment, backend: Backend, iso: IsoLookup) -> PromptInput:
    return PromptInput(
        text=PLAMO_TEMPLATE.format(text=segment.source),
        add_special_tokens=True,
        stop=PLAMO_STOP,
    )


def _hy_mt2(segment: Segment, backend: Backend, iso: IsoLookup) -> PromptInput:
    messages = [{"role": "user", "content": HY_MT2_INSTRUCTION + segment.source}]
    return PromptInput(text=backend.render_chat(messages, {}))


def _nllb(segment: Segment, backend: Backend, iso: IsoLookup) -> PromptInput:
    return PromptInput(text=segment.source, src_lang=segment.corpus_lang)


def _madlad(segment: Segment, backend: Backend, iso: IsoLookup) -> PromptInput:
    return PromptInput(text=MADLAD_TARGET_TAG + segment.source)


BUILDERS: dict[str, Builder] = {
    "generic_chat": _generic_chat,
    "qwen_nothink": _qwen_nothink,
    "hy_mt15": _hy_mt15,
    "translategemma": _translategemma,
    "plamo3_translate": _plamo3,
    "hy_mt2": _hy_mt2,
    "nllb_tag": _nllb,
    "madlad_tag": _madlad,
}


def template_names() -> frozenset[str]:
    return frozenset(BUILDERS)


def make_prompt(
    template: str,
    segment: Segment,
    backend: Backend,
    iso: IsoLookup | Mapping[str, str] | None = None,
) -> PromptInput:
    """Render the prompt for one segment. ``iso`` maps corpus codes to ISO 639-1 codes."""
    try:
        builder = BUILDERS[template]
    except KeyError as error:
        raise ValueError(f"no prompt template named {template!r}") from error
    if iso is None:
        lookup: IsoLookup = _no_iso
    elif isinstance(iso, Mapping):
        lookup = iso.get
    else:
        lookup = iso
    return builder(segment, backend, lookup)


def _no_iso(_: str) -> None:
    return None
