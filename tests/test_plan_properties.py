import string

from hypothesis import given
from hypothesis import strategies as st

from xx_to_en.plan import _runs, build_shards
from xx_to_en.runner import WorkItem

LANGS = st.lists(st.text(alphabet=string.ascii_lowercase, min_size=3, max_size=3), unique=True, max_size=60)
SIZES = st.integers(min_value=1, max_value=12)


def _items(benchmark, langs):
    return [WorkItem(benchmark, lang, lang) for lang in langs]


@given(LANGS, SIZES)
def test_runs_cover_the_languages_in_order_with_full_runs_first(langs, size):
    runs = _runs(langs, size)
    assert [lang for run in runs for lang in run] == langs
    assert all(len(run) == size for run in runs[:-1])
    assert all(0 < len(run) <= size for run in runs)


@given(LANGS, SIZES)
def test_plan_without_splits_chunks_each_benchmark_in_sorted_order(langs, size):
    shards = build_shards("m", _items("b", langs), sizes={"b": size})
    assert [lang for shard in shards for lang in shard.langs] == sorted(langs)
    assert [shard.index for shard in shards] == list(range(len(shards)))
    assert all(shard.part is None for shard in shards)
    assert len({shard.shard_id for shard in shards}) == len(shards)


@given(LANGS, SIZES)
def test_plan_is_deterministic(langs, size):
    items = _items("b", langs)
    assert build_shards("m", items, sizes={"b": size}) == build_shards("m", items, sizes={"b": size})


@given(
    LANGS,
    SIZES,
    st.integers(min_value=1, max_value=5),
    st.sets(st.integers(min_value=0, max_value=20)),
)
def test_splits_cover_every_language_once_and_keep_finished_shards_whole(langs, size, split, done_indexes):
    items = _items("b", langs)
    whole = build_shards("m", items, sizes={"b": size})
    done = {shard.shard_id for shard in whole if shard.index in done_indexes}
    shards = build_shards(
        "m", items, sizes={"b": size}, splits={"b": split}, is_done=lambda shard: shard.shard_id in done
    )
    assert sorted(lang for shard in shards for lang in shard.langs) == sorted(langs)
    assert len({shard.shard_id for shard in shards}) == len(shards)
    for base in whole:
        own = [shard for shard in shards if shard.index == base.index]
        if base.shard_id in done:
            assert own == [base]
            continue
        assert [shard.part for shard in own] == list(range(len(own)))
        assert tuple(lang for shard in own for lang in shard.langs) == base.langs
        assert all(len(shard.langs) <= split for shard in own)


@given(LANGS, LANGS)
def test_benchmarks_are_planned_independently(first, second):
    items = _items("x", first) + _items("y", second)
    shards = build_shards("m", items, sizes={"x": 3, "y": 4})
    for benchmark, langs in (("x", first), ("y", second)):
        own = [shard for shard in shards if shard.benchmark == benchmark]
        assert [lang for shard in own for lang in shard.langs] == sorted(langs)
