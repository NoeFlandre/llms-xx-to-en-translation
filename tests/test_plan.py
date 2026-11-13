from xx_to_en.plan import Shard, build_shards, pending
from xx_to_en.results_store import shard_path
from xx_to_en.runner import WorkItem


def _items(benchmark: str, langs: list[str]) -> list[WorkItem]:
    return [WorkItem(benchmark, lang, lang) for lang in langs]


def test_chunks_are_sorted_and_sized_per_benchmark():
    items = _items("ntrex_128", [f"l{i:02d}_Latn" for i in range(20)])
    shards = build_shards("m", items, sizes={"ntrex_128": 8})
    assert [len(s.langs) for s in shards] == [8, 8, 4]
    assert shards[0].langs[0] == "l00_Latn"
    assert [s.index for s in shards] == [0, 1, 2]


def test_shards_group_by_benchmark_in_name_order():
    items = _items("wmt23_xen", ["ces_Latn"]) + _items("flores_plus_devtest", ["fra_Latn", "deu_Latn"])
    shards = build_shards("m", items)
    assert [(s.benchmark, s.langs) for s in shards] == [
        ("flores_plus_devtest", ("deu_Latn", "fra_Latn")),
        ("wmt23_xen", ("ces_Latn",)),
    ]


def test_shard_ids_are_stable_and_zero_padded():
    shard = Shard("m", "flores_plus_devtest", 3, ("fra_Latn",))
    assert shard.shard_id == "flores_plus_devtest-003"


def test_output_path_layout():
    shard = Shard("translategemma-4b-it", "flores_plus_devtest", 0, ("fra_Latn",))
    assert shard_path(shard) == (
        "results/translategemma-4b-it/flores_plus_devtest/flores_plus_devtest-000.parquet"
    )


def test_pending_keeps_plan_order_and_drops_done_shards():
    shards = build_shards("m", _items("wmt22_xen", ["ces_Latn", "deu_Latn"]), sizes={"wmt22_xen": 1})
    remaining = pending(shards, is_done=lambda s: s.index == 0)
    assert [s.shard_id for s in remaining] == ["wmt22_xen-001"]


def test_empty_plan_has_no_shards():
    assert build_shards("m", []) == []
