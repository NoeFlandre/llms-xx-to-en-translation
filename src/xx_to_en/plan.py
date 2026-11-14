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


@dataclass(frozen=True)
class Shard:
    model: str
    benchmark: str
    index: int
    langs: tuple[str, ...]

    @property
    def shard_id(self) -> str:
        return f"{self.benchmark}-{self.index:03d}"


def build_shards(
    model: str, items: Sequence[WorkItem], sizes: Mapping[str, int] = LANGS_PER_SHARD
) -> list[Shard]:
    """Chunk the model's work items by benchmark, in sorted language order."""
    shards: list[Shard] = []
    for benchmark in sorted({item.benchmark for item in items}):
        size = sizes.get(benchmark, DEFAULT_LANGS_PER_SHARD)
        shards.extend(_chunk(model, benchmark, _langs_of(items, benchmark), size))
    return shards


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
