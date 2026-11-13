"""Generation budget tied to source length."""

from __future__ import annotations

import math


def max_new_tokens(source_tokens: int, *, ratio: float = 3.0, floor: int = 32, ceiling: int = 2048) -> int:
    """Token budget for one segment: ``ratio`` times the source length, clamped to [floor, ceiling]."""
    if source_tokens < 0:
        raise ValueError("source_tokens must be non-negative")
    if ratio <= 0 or floor > ceiling:
        raise ValueError("ratio must be positive and floor must not exceed ceiling")
    return min(ceiling, max(floor, math.ceil(source_tokens * ratio)))
