"""Reference-based metrics. chrF++ is implemented; COMET-22 and MetricX-24 are not yet wired in."""

from __future__ import annotations

from collections.abc import Sequence

from sacrebleu.metrics import CHRF

_CHRF_PLUS_PLUS = CHRF(word_order=2)


def _check_lengths(hypotheses: Sequence[str], references: Sequence[str]) -> None:
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses but {len(references)} references")


def corpus_chrf_pp(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Corpus-level chrF++ on a 0-100 scale."""
    _check_lengths(hypotheses, references)
    if not hypotheses:
        raise ValueError("no segments to score")
    return float(_CHRF_PLUS_PLUS.corpus_score(list(hypotheses), [list(references)]).score)


def segment_chrf_pp(hypotheses: Sequence[str], references: Sequence[str]) -> list[float]:
    """Per-segment chrF++ on a 0-100 scale, for paired bootstrap."""
    _check_lengths(hypotheses, references)
    return [
        float(_CHRF_PLUS_PLUS.sentence_score(hypothesis, [reference]).score)
        for hypothesis, reference in zip(hypotheses, references, strict=True)
    ]
