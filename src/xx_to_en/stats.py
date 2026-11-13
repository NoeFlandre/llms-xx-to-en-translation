"""Bootstrap confidence intervals for segment-level scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAX_CELLS_PER_CHUNK = 2_000_000


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float


def _validate(confidence: float, size: int) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if size == 0:
        raise ValueError("at least one score is required")


def _resampled_means(
    data: NDArray[np.float64], n_resamples: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Bootstrap means, generated in chunks to bound memory."""
    means = np.empty(n_resamples)
    chunk = max(1, MAX_CELLS_PER_CHUNK // data.size)
    for start in range(0, n_resamples, chunk):
        stop = min(start + chunk, n_resamples)
        indices = rng.integers(0, data.size, size=(stop - start, data.size))
        means[start:stop] = data[indices].mean(axis=1)
    return means


def bootstrap_ci(
    values: ArrayLike,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for the mean of ``values``."""
    data = np.asarray(values, dtype=np.float64)
    _validate(confidence, data.size)
    rng = np.random.default_rng(seed)
    means = _resampled_means(data, n_resamples, rng)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return Estimate(value=float(data.mean()), low=float(low), high=float(high))


def paired_bootstrap_diff(
    system_a: ArrayLike,
    system_b: ArrayLike,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Bootstrap CI for mean(a - b) over the same segments.

    A CI that excludes zero indicates a difference at the chosen confidence level.
    """
    if np.shape(system_a) != np.shape(system_b):
        raise ValueError("paired systems must score the same number of segments")
    differences = np.asarray(system_a, dtype=np.float64) - np.asarray(system_b, dtype=np.float64)
    return bootstrap_ci(differences, n_resamples=n_resamples, confidence=confidence, seed=seed)
