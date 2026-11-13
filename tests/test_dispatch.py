from datetime import datetime

import pytest

from xx_to_en.dispatch import GpuPool, choose_placements, crosses_boundary


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        (datetime(2026, 11, 13, 11, 30), False),  # 11:30-12:30 inside the day window
        (datetime(2026, 11, 13, 18, 30), True),  # crosses 19:00
        (datetime(2026, 11, 13, 8, 30), True),  # crosses 09:00
        (datetime(2026, 11, 13, 9, 0), False),  # starts exactly at 09:00
        (datetime(2026, 11, 13, 18, 0), False),  # ends exactly at 19:00
        (datetime(2026, 11, 13, 19, 30), False),  # night, no boundary inside
    ],
)
def test_crosses_boundary(start, expected):
    assert crosses_boundary(start) is expected


def _pool(site, cluster, memory, free):
    return GpuPool(site, cluster, memory, free)


def test_big_model_only_goes_to_a_gpu_large_enough():
    pools = [_pool("nancy", "gres", 48, 5), _pool("nancy", "granite", 96, 1)]
    placements = choose_placements([("big", 80)], pools, running_models=frozenset(), running_per_site={})
    assert [(p.model, p.cluster) for p in placements] == [("big", "granite")]


def test_best_fit_keeps_large_gpus_for_large_models():
    pools = [_pool("nancy", "gres", 48, 1), _pool("nancy", "granite", 96, 1)]
    placements = choose_placements([("small", 16)], pools, running_models=frozenset(), running_per_site={})
    assert placements[0].cluster == "gres"


def test_running_models_are_not_duplicated():
    pools = [_pool("nancy", "gres", 48, 3)]
    placements = choose_placements(
        [("a", 8), ("b", 8)], pools, running_models=frozenset({"a"}), running_per_site={}
    )
    assert [p.model for p in placements] == ["b"]


def test_per_site_and_total_caps_apply():
    pools = [_pool("nancy", "gres", 48, 10), _pool("rennes", "abacus26", 48, 10)]
    pending = [(f"m{i}", 8) for i in range(12)]
    placements = choose_placements(pending, pools, running_models=frozenset(), running_per_site={})
    assert len(placements) == 6
    assert sum(p.site == "nancy" for p in placements) == 3


def test_no_free_gpu_no_placement():
    pools = [_pool("nancy", "gres", 48, 0)]
    assert choose_placements([("a", 8)], pools, running_models=frozenset(), running_per_site={}) == []
