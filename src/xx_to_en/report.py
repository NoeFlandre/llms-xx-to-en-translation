"""Aggregation: volume-weighted overall score and per-language winners."""

from __future__ import annotations

from collections.abc import Mapping


def _positive_weights(scores: Mapping[str, float], volumes: Mapping[str, int]) -> dict[str, int]:
    return {language: volumes[language] for language in scores if volumes.get(language, 0) > 0}


def volume_weighted_mean(scores: Mapping[str, float], volumes: Mapping[str, int]) -> float:
    """Mean of per-language scores weighted by corpus volume.

    Languages without a positive volume are ignored.
    """
    weights = _positive_weights(scores, volumes)
    if not weights:
        raise ValueError("no language has both a score and a positive volume")
    total = sum(weights.values())
    return sum(scores[language] * weight for language, weight in weights.items()) / total


def _scores_for(table: Mapping[str, Mapping[str, float]], language: str) -> dict[str, float]:
    return {model: scores[language] for model, scores in table.items() if language in scores}


def _best_model(candidates: Mapping[str, float]) -> str:
    return max(sorted(candidates), key=candidates.__getitem__)


def per_language_winner(table: Mapping[str, Mapping[str, float]]) -> dict[str, str]:
    """Best model per language. ``table`` maps model -> language -> score; ties go to the earlier name."""
    languages = sorted({language for scores in table.values() for language in scores})
    return {language: _best_model(_scores_for(table, language)) for language in languages}
