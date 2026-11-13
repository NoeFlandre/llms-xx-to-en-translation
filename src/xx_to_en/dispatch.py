"""Placement policy for tiny Grid'5000 jobs. Pure: the I/O loop lives in ``scripts/g5k/dispatch.py``.

Rules followed (Grid'5000 usage policy, default queue only):
- A job of one hour or less is exempt from the daytime quota when submitted less than 10 minutes
  before it starts. Every job is one hour and is submitted only when its GPU is already free.
- A job cannot cross 09:00 or 19:00. Checked on every day, stricter than the weekend rule.
- Jobs are spread over sites and capped per site and in total.

Turing GPUs have no bf16: they take only models marked fp16_ok, and run them in float16.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

JOB_MINUTES = 60
MAX_JOBS_PER_SITE = 24
MAX_JOBS_TOTAL = 60
# Besteffort is granted, but the usage policy asks for staff contact before large-scale use: keep it modest.
MAX_BESTEFFORT_JOBS = 4
BOUNDARY_HOURS = (9, 19)


@dataclass(frozen=True)
class GpuPool:
    site: str
    cluster: str
    memory_gb: int
    free: int
    bf16: bool = True
    besteffort: bool = False


@dataclass(frozen=True)
class Pending:
    model: str
    memory_gb: int
    fp16_ok: bool = False


@dataclass(frozen=True)
class Placement:
    model: str
    site: str
    cluster: str
    dtype: str = "bfloat16"
    queue: str = "default"


def crosses_boundary(start: datetime, minutes: int = JOB_MINUTES) -> bool:
    """True when a job starting at ``start`` would straddle a 09:00 or 19:00 boundary."""
    end = start + timedelta(minutes=minutes)
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        for hour in BOUNDARY_HOURS:
            boundary = day.replace(hour=hour)
            if start < boundary < end:
                return True
        day += timedelta(days=1)
    return False


def choose_placements(
    pending: Sequence[Pending],
    pools: Sequence[GpuPool],
    *,
    running_models: frozenset[str],
    running_per_site: Mapping[str, int],
    running_besteffort: int = 0,
) -> list[Placement]:
    """Assign pending models, in priority order, to free GPUs.

    Each model gets at most one job at a time. bf16 GPUs are preferred, then the smallest GPU that
    fits, so large GPUs stay free for large models. Turing GPUs only take fp16-tolerant models.
    """
    free = {(pool.site, pool.cluster): pool.free for pool in pools}
    per_site = dict(running_per_site)
    total = sum(per_site.values())
    besteffort_total = running_besteffort
    placements: list[Placement] = []
    for item in pending:
        if item.model in running_models or any(p.model == item.model for p in placements):
            continue
        if total >= MAX_JOBS_TOTAL:
            break
        options = [
            pool
            for pool in pools
            if pool.memory_gb >= item.memory_gb
            and free[(pool.site, pool.cluster)] > 0
            and per_site.get(pool.site, 0) < MAX_JOBS_PER_SITE
            and (pool.bf16 or item.fp16_ok)
            and (not pool.besteffort or besteffort_total < MAX_BESTEFFORT_JOBS)
        ]
        if not options:
            continue
        best = min(
            options,
            key=lambda pool: (pool.besteffort, not pool.bf16, pool.memory_gb, pool.site, pool.cluster),
        )
        dtype = "bfloat16" if best.bf16 else "float16"
        queue = "besteffort" if best.besteffort else "default"
        besteffort_total += best.besteffort
        placements.append(Placement(item.model, best.site, best.cluster, dtype, queue))
        free[(best.site, best.cluster)] -= 1
        per_site[best.site] = per_site.get(best.site, 0) + 1
        total += 1
    return placements
