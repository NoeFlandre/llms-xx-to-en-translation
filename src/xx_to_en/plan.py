"""Shard planning: split each model's work into fixed-size language chunks per benchmark.

A shard is the unit of output: one JSONL file in the results dataset. Shard ids are stable, so a
re-run skips shards whose output already exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from xx_to_en.runner import WorkItem

LANGS_PER_SHARD: Mapping[str, int] = {
    "flores_plus_devtest": 8,
    "ntrex_128": 8,
    "tatoeba_xen": 8,
    "wmt22_xen": 8,
    "wmt23_xen": 8,
}
DEFAULT_LANGS_PER_SHARD = 8
NO_SPLITS: Mapping[str, int] = {}


@dataclass(frozen=True)
class Shard:
    model: str
    benchmark: str
    index: int
    langs: tuple[str, ...]
    part: int | None = None  # set on a part of an unfinished shard that was cut into smaller pieces

    @property
    def shard_id(self) -> str:
        base = f"{self.benchmark}-{self.index:03d}"
        return base if self.part is None else f"{base}-{self.part:02d}"


def _never_done(shard: Shard) -> bool:
    return False


def build_shards(
    model: str,
    items: Sequence[WorkItem],
    sizes: Mapping[str, int] = LANGS_PER_SHARD,
    splits: Mapping[str, int] = NO_SPLITS,
    is_done: Callable[[Shard], bool] = _never_done,
) -> list[Shard]:
    """Chunk the model's work items by benchmark, in sorted language order.

    A benchmark in ``splits`` cuts each unfinished shard into parts of that many languages, so a
    shard too slow for one job still finishes, part by part. Finished shards keep their whole-shard id.
    """
    shards: list[Shard] = []
    for benchmark in sorted({item.benchmark for item in items}):
        size = sizes.get(benchmark, DEFAULT_LANGS_PER_SHARD)
        for shard in _chunk(model, benchmark, _langs_of(items, benchmark), size):
            shards.extend(_parts(shard, splits.get(benchmark), is_done))
    return shards


def _parts(shard: Shard, split_size: int | None, is_done: Callable[[Shard], bool]) -> list[Shard]:
    """The shard itself, or its parts when it is unfinished and its benchmark asks for a split."""
    if split_size is None or is_done(shard):
        return [shard]
    return _split(shard, split_size)


def _split(shard: Shard, size: int) -> list[Shard]:
    """Consecutive parts of ``size`` languages of one shard, numbered from 0."""
    return [
        Shard(shard.model, shard.benchmark, shard.index, shard.langs[start : start + size], part)
        for part, start in enumerate(range(0, len(shard.langs), size))
    ]


def _langs_of(items: Sequence[WorkItem], benchmark: str) -> list[str]:
    """Corpus languages of one benchmark's work items, sorted."""
    return sorted(item.corpus_lang for item in items if item.benchmark == benchmark)


def _chunk(model: str, benchmark: str, langs: Sequence[str], size: int) -> list[Shard]:
    """Consecutive runs of ``size`` languages, numbered from 0."""
    return [
        Shard(model, benchmark, index, tuple(langs[start : start + size]))
        for index, start in enumerate(range(0, len(langs), size))
    ]


def pending(shards: Sequence[Shard], is_done: Callable[[Shard], bool]) -> list[Shard]:
    """Shards whose output does not exist yet, in plan order."""
    return [shard for shard in shards if not is_done(shard)]
