from datetime import datetime

import pytest

from xx_to_en.dispatch import (
    MAX_BESTEFFORT_JOBS,
    MAX_JOBS_PER_SITE,
    GpuPool,
    Pending,
    choose_placements,
    crosses_boundary,
)


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        (datetime(2026, 11, 13, 11, 30), False),
        (datetime(2026, 11, 13, 18, 30), True),
        (datetime(2026, 11, 13, 8, 30), True),
        (datetime(2026, 11, 13, 9, 0), False),
        (datetime(2026, 11, 13, 18, 0), False),
        (datetime(2026, 11, 13, 19, 30), False),
    ],
)
def test_crosses_boundary(start, expected):
    assert crosses_boundary(start) is expected


def _pool(site, cluster, memory, free, bf16=True):
    return GpuPool(site, cluster, memory, free, bf16)


def _place(pending, pools, running=frozenset(), per_site=None):
    return choose_placements(pending, pools, running_models=running, running_per_site=per_site or {})


def test_big_model_only_goes_to_a_gpu_large_enough():
    pools = [_pool("nancy", "gres", 48, 5), _pool("nancy", "granite", 96, 1)]
    placements = _place([Pending("big", 80)], pools)
    assert [(p.model, p.cluster) for p in placements] == [("big", "granite")]


def test_best_fit_keeps_large_gpus_for_large_models():
    pools = [_pool("nancy", "gres", 48, 1), _pool("nancy", "granite", 96, 1)]
    assert _place([Pending("small", 16)], pools)[0].cluster == "gres"


def test_running_models_are_not_duplicated():
    pools = [_pool("nancy", "gres", 48, 3)]
    placements = _place([Pending("a", 8), Pending("b", 8)], pools, running=frozenset({"a"}))
    assert [p.model for p in placements] == ["b"]


def test_per_site_and_total_caps_apply():
    pools = [_pool("nancy", "gres", 48, 20), _pool("sophia", "musa", 96, 20)]
    pending = [Pending(f"m{i}", 8) for i in range(30)]
    placements = _place(pending, pools)
    assert len(placements) == 2 * MAX_JOBS_PER_SITE
    assert all(sum(p.site == site for p in placements) <= MAX_JOBS_PER_SITE for site in ("nancy", "sophia"))


def test_no_free_gpu_no_placement():
    assert _place([Pending("a", 8)], [_pool("nancy", "gres", 48, 0)]) == []


def test_default_queue_is_preferred_over_besteffort():
    pools = [_pool("sophia", "esterel45", 96, 5), _pool("nancy", "gres", 48, 5)]
    pools[0] = GpuPool("sophia", "esterel45", 96, 5, True, besteffort=True)
    placement = _place([Pending("big", 40)], pools)[0]
    assert placement.cluster == "gres"
    assert placement.queue == "default"


def test_besteffort_used_when_default_is_full():
    pools = [GpuPool("sophia", "esterel45", 96, 5, True, besteffort=True)]
    placement = _place([Pending("big", 40)], pools)[0]
    assert placement.queue == "besteffort"


def test_turing_pool_takes_only_fp16_models():
    pools = [_pool("nancy", "grue", 16, 5, bf16=False)]
    assert _place([Pending("bf16-only", 8, fp16_ok=False)], pools) == []
    placements = _place([Pending("small", 8, fp16_ok=True)], pools)
    assert placements[0].dtype == "float16"


def test_bf16_pools_preferred_over_turing():
    pools = [_pool("nancy", "grue", 16, 5, bf16=False), _pool("nancy", "gres", 48, 5)]
    placement = _place([Pending("small", 8, fp16_ok=True)], pools)[0]
    assert placement.cluster == "gres"
    assert placement.dtype == "bfloat16"


def test_besteffort_is_capped_while_staff_contact_is_pending():
    pools = [GpuPool("sophia", "esterel36", 40, 20, True, besteffort=True)]
    pending = [Pending(f"m{i}", 8) for i in range(10)]
    placements = _place(pending, pools)
    assert len(placements) == MAX_BESTEFFORT_JOBS
    assert all(p.queue == "besteffort" for p in placements)
