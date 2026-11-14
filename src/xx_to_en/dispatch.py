"""Placement policy for tiny Grid'5000 jobs. Pure: the I/O loop lives in ``scripts/g5k/dispatch.py``.

Rules followed (Grid'5000 usage policy, default queue only):
- A job of one hour or less is exempt from the daytime quota when submitted less than 10 minutes
  before it starts. Every job is one hour and is submitted only when its GPU is already free.
- Default queue: a job cannot cross 09:00 or 19:00, checked every day (stricter than the weekend rule).
  Observed: a default-queue job submitted at 08:27 on a Saturday was scheduled to start at 09:03.
- Jobs are spread over sites and capped per site and in total.

Turing GPUs have no bf16: they take only models marked fp16_ok, and run them in float16.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

JOB_MINUTES = 60
MAX_JOBS_PER_SITE = 24
MAX_JOBS_TOTAL = 60
# Besteffort is granted, but the usage policy asks for staff contact before large-scale use: keep it modest.
MAX_BESTEFFORT_JOBS = 4
BOUNDARY_HOURS = (9, 19)
MAX_JOBS_PER_PAIR = 6
ITEMS_PER_JOB = 4  # generation and scoring items one job can finish within its budget


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
    jobs: int = 1  # parallel jobs wanted for this model and benchmark


@dataclass(frozen=True)
class Placement:
    model: str
    site: str
    cluster: str
    dtype: str = "bfloat16"
    queue: str = "default"
    shard_index: int = 0  # the job takes shards whose index is congruent to this, modulo shard_count
    shard_count: int = 1


def jobs_for_pair(pending_items: int) -> int:
    """Parallel jobs for one model and benchmark: one per ITEMS_PER_JOB pending items, at most the cap."""
    return max(1, min(MAX_JOBS_PER_PAIR, math.ceil(pending_items / ITEMS_PER_JOB)))


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


@dataclass
class _Budget:
    """Running counts for one ``choose_placements`` call: free GPUs, jobs per site and in total."""

    free: dict[tuple[str, str], int]
    per_site: dict[str, int]
    total: int
    besteffort_total: int

    def has_room(self, pool: GpuPool) -> bool:
        """True when ``pool`` has a free GPU and neither its site nor the besteffort cap is full."""
        return (
            self.free[(pool.site, pool.cluster)] > 0
            and self.per_site.get(pool.site, 0) < MAX_JOBS_PER_SITE
            and (not pool.besteffort or self.besteffort_total < MAX_BESTEFFORT_JOBS)
        )

    def take(self, pool: GpuPool) -> None:
        """Count one more job on ``pool``."""
        self.free[(pool.site, pool.cluster)] -= 1
        self.per_site[pool.site] = self.per_site.get(pool.site, 0) + 1
        self.total += 1
        self.besteffort_total += pool.besteffort


def _fits(pool: GpuPool, item: Pending) -> bool:
    """True when the model fits in the GPU memory and the GPU's dtype suits the model."""
    return pool.memory_gb >= item.memory_gb and (pool.bf16 or item.fp16_ok)


def _preference(pool: GpuPool) -> tuple[bool, bool, int, str, str]:
    """Sort key, smallest first: default queue, then bf16, then the smallest GPU so large ones stay free."""
    return (pool.besteffort, not pool.bf16, pool.memory_gb, pool.site, pool.cluster)


def _eligible(item: Pending, pools: Sequence[GpuPool], budget: _Budget) -> list[GpuPool]:
    """Pools that fit the model and still have room for one more job."""
    return [pool for pool in pools if _fits(pool, item) and budget.has_room(pool)]


def _placement(item: Pending, pool: GpuPool, slot: int) -> Placement:
    return Placement(
        item.model,
        pool.site,
        pool.cluster,
        "bfloat16" if pool.bf16 else "float16",
        "besteffort" if pool.besteffort else "default",
        slot,
        item.jobs,
    )


def _place_batch(item: Pending, pools: Sequence[GpuPool], budget: _Budget) -> list[Placement]:
    """Place up to ``item.jobs`` jobs for one model, one per slot, until a cap or the pools stop it."""
    placements: list[Placement] = []
    for slot in range(item.jobs):
        if budget.total >= MAX_JOBS_TOTAL:
            break
        options = _eligible(item, pools, budget)
        if not options:
            break
        best = min(options, key=_preference)
        placements.append(_placement(item, best, slot))
        budget.take(best)
    return placements


def choose_placements(
    pending: Sequence[Pending],
    pools: Sequence[GpuPool],
    *,
    running_models: frozenset[str],
    running_per_site: Mapping[str, int],
    running_besteffort: int = 0,
) -> list[Placement]:
    """Assign pending models, in priority order, to free GPUs.

    A model and benchmark pair gets a new batch of jobs only when none of its jobs is running; the
    batch has up to ``Pending.jobs`` jobs, one per free GPU, each with its own shard index. bf16 GPUs
    are preferred, then the smallest GPU that fits, so large GPUs stay free for large models. Turing
    GPUs only take fp16-tolerant models.
    """
    budget = _Budget(
        free={(pool.site, pool.cluster): pool.free for pool in pools},
        per_site=dict(running_per_site),
        total=sum(running_per_site.values()),
        besteffort_total=running_besteffort,
    )
    placements: list[Placement] = []
    for item in pending:
        if item.model not in running_models:
            placements.extend(_place_batch(item, pools, budget))
    return placements
