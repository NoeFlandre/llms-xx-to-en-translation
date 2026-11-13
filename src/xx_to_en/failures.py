"""Failure checks on model outputs: empty, repetition loops, length outliers, copies, wrong language."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

WORD = re.compile(r"\w+")
ENGLISH_TARGET = "eng_Latn"

LanguageIdentifier = Callable[[str], str]


@dataclass(frozen=True)
class FailureFlags:
    empty: bool
    repetition_loop: bool
    length_ratio_outlier: bool
    untranslated_copy: bool
    wrong_language: bool | None  # None when no language identifier was supplied

    @property
    def any(self) -> bool:
        return any(
            (self.empty, self.repetition_loop, self.length_ratio_outlier, self.untranslated_copy)
        ) or bool(self.wrong_language)


def words(text: str) -> list[str]:
    """Lower-cased, NFKC-normalised word tokens."""
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def has_repetition_loop(text: str, *, max_period: int = 5, min_repeats: int = 4) -> bool:
    """True if some run of 1..max_period words repeats at least min_repeats times in a row."""
    tokens = words(text)
    for period in range(1, max_period + 1):
        needed = period * (min_repeats - 1)
        matches = 0
        for index in range(period, len(tokens)):
            matches = matches + 1 if tokens[index] == tokens[index - period] else 0
            if matches >= needed:
                return True
    return False


def is_length_outlier(source: str, hypothesis: str, *, low: float = 0.33, high: float = 3.0) -> bool:
    """True if the output word count is outside [low, high] times the source word count."""
    source_words = len(words(source))
    if source_words == 0:
        return False
    ratio = len(words(hypothesis)) / source_words
    return ratio < low or ratio > high


def is_untranslated_copy(source: str, hypothesis: str) -> bool:
    """True if the output equals the source after case and punctuation are removed."""
    source_words = words(source)
    return bool(source_words) and source_words == words(hypothesis)


def is_wrong_language(
    hypothesis: str, identify: LanguageIdentifier | None, target: str = ENGLISH_TARGET
) -> bool | None:
    """True if the identifier labels a non-empty output with a language other than ``target``."""
    if identify is None:
        return None
    if not hypothesis.strip():
        return False
    return identify(hypothesis) != target


def detect(
    source: str,
    hypothesis: str,
    identify: LanguageIdentifier | None = None,
    target: str = ENGLISH_TARGET,
) -> FailureFlags:
    """Run every check on one source/output pair."""
    return FailureFlags(
        empty=not hypothesis.strip(),
        repetition_loop=has_repetition_loop(hypothesis),
        length_ratio_outlier=is_length_outlier(source, hypothesis),
        untranslated_copy=is_untranslated_copy(source, hypothesis),
        wrong_language=is_wrong_language(hypothesis, identify, target),
    )
