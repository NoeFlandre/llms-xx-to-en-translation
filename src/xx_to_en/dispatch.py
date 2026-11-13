"""Placement policy for tiny Grid'5000 jobs. Pure: the I/O loop lives in ``scripts/g5k/dispatch.py``.

Rules followed (Grid'5000 usage policy, default queue):
- A job of one hour or less is exempt from the daytime quota when it is submitted less than
  10 minutes before it starts. Every job here is one hour and is submitted only when its GPU is free.
- A job cannot cross 09:00 or 19:00. Checked on every day, which is stricter than the weekend rule.
- Parallel small jobs are spread across sites and capped per site and in total.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

JOB_MINUTES = 60
MAX_JOBS_PER_SITE = 3
MAX_JOBS_TOTAL = 8
BOUNDARY_HOURS = (9, 19)


@dataclass(frozen=True)
class GpuPool:
    site: str
    cluster: str
    memory_gb: int
    free: int


@dataclass(frozen=True)
class Placement:
    model: str
    site: str
    cluster: str


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
    pending: Sequence[tuple[str, int]],
    pools: Sequence[GpuPool],
    *,
    running_models: frozenset[str],
    running_per_site: Mapping[str, int],
) -> list[Placement]:
    """Assign pending models, in priority order, to free GPUs.

    ``pending`` holds (model, required GPU memory in GB). Each model gets at most one job at a time,
    and the smallest GPU that fits is used, so large GPUs stay free for large models.
    """
    free = {(pool.site, pool.cluster): pool.free for pool in pools}
    per_site = dict(running_per_site)
    total = sum(per_site.values())
    placements: list[Placement] = []
    for model, need_gb in pending:
        if model in running_models or any(p.model == model for p in placements):
            continue
        if total >= MAX_JOBS_TOTAL:
            break
        options = [
            pool
            for pool in pools
            if pool.memory_gb >= need_gb
            and free[(pool.site, pool.cluster)] > 0
            and per_site.get(pool.site, 0) < MAX_JOBS_PER_SITE
        ]
        if not options:
            continue
        best = min(options, key=lambda pool: (pool.memory_gb, pool.site, pool.cluster))
        placements.append(Placement(model, best.site, best.cluster))
        free[(best.site, best.cluster)] -= 1
        per_site[best.site] = per_site.get(best.site, 0) + 1
        total += 1
    return placements
