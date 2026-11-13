"""Corpus language codes and their benchmark-specific equivalents."""

from __future__ import annotations

import re
from collections.abc import Mapping

CORPUS_LANGUAGE = re.compile(r"[a-z]{3}_[A-Z][a-z]{3}")
ENGLISH = "eng_Latn"


def is_corpus_language(config_name: str) -> bool:
    """True for per-language corpus configs such as ``fra_Latn``.

    Source-type configs (``wikipedia``, ``website``) and auxiliary tables are excluded.
    """
    return CORPUS_LANGUAGE.fullmatch(config_name) is not None


def corpus_languages(config_rows: Mapping[str, int]) -> dict[str, int]:
    """Non-English language configs with their row counts."""
    return {name: rows for name, rows in config_rows.items() if is_corpus_language(name) and name != ENGLISH}


class LanguageMap:
    """Maps corpus codes to benchmark codes.

    Defaults: identity for FLORES+, NTREX and WMT-style codes; ISO 639-3 prefix for Tatoeba.
    An explicit empty string means the benchmark has no equivalent variant.
    """

    def __init__(self, mapping: Mapping[str, Mapping[str, str]]) -> None:
        self._mapping = {code: dict(entry) for code, entry in mapping.items()}

    def benchmark_code(self, corpus_code: str, benchmark: str) -> str:
        explicit = self._mapping.get(corpus_code, {})
        if benchmark in explicit:
            return explicit[benchmark]
        if benchmark.startswith("tatoeba"):
            return corpus_code.split("_", 1)[0]
        return corpus_code

    def iso639_1(self, corpus_code: str) -> str | None:
        """ISO 639-1 code for a corpus language, or None when the language has none."""
        return self._mapping.get(corpus_code, {}).get("iso639_1") or None
