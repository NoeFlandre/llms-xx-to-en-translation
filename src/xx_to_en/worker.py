"""Shard worker: translate, then score, the pending shards of one model within a time budget.

Phase 1 runs in the generation environment: each pending shard is translated and its hypotheses
are uploaded. Phase 2 runs in the metrics environment: the hypotheses are scored and the scores are
uploaded. A shard is the resume unit: a shard whose output already exists in the dataset is skipped,
so a killed job loses at most one shard.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xx_to_en.benchmarks import Segment
from xx_to_en.cache import OutputCache
from xx_to_en.config import BenchmarkConfig, ModelConfig
from xx_to_en.languages import LanguageMap
from xx_to_en.plan import Shard
from xx_to_en.provenance import created_now
from xx_to_en.runner import Decoding, Translation, translate
from xx_to_en.sources import load_segments
from xx_to_en.tables import read_table, write_table

if TYPE_CHECKING:
    from xx_to_en.metric_runtime import Scorers

PRIVATE_FIELDS = frozenset({"source", "reference"})
SegmentKey = tuple[str, str, str]


def may_start_next(elapsed_s: float, last_shard_s: float | None, budget_s: float) -> bool:
    """Start another shard only if the previous shard's duration still fits in the budget."""
    if last_shard_s is None:
        return elapsed_s < budget_s
    return elapsed_s + last_shard_s <= budget_s


def run_budgeted(
    pending: Iterable[Shard],
    *,
    budget_s: float,
    step: Callable[[Shard], None],
    clock: Callable[[], float] = time.monotonic,
) -> list[Shard]:
    """Generate shards in order while the budget can fit another one. Returns the finished shards."""
    started = clock()
    finished: list[Shard] = []
    last: float | None = None
    for shard in pending:
        if not may_start_next(clock() - started, last, budget_s):
            break
        begin = clock()
        step(shard)
        last = clock() - begin
        finished.append(shard)
    return finished


def hypothesis_rows(
    translations: Sequence[Translation], run: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Public hypothesis records: ids, output, raw output, finish reason, and the run settings."""
    run = dict(run or {})
    return [
        {
            "benchmark": item.segment.benchmark,
            "corpus_lang": item.segment.corpus_lang,
            "bench_lang": item.segment.bench_lang,
            "segment_id": item.segment.segment_id,
            "hypothesis": item.hypothesis,
            "raw_hypothesis": item.raw_text,
            "finish_reason": item.finish_reason,
            "truncated": item.finish_reason == "length",
            "output_tokens": item.output_tokens,
            "max_new_tokens": item.max_new_tokens,
            "prompt_sha256": item.prompt_sha256,
            "cached": item.cached,
            **run,
        }
        for item in translations
    ]


def segment_index(segments: Iterable[Segment]) -> dict[SegmentKey, Segment]:
    return {(s.benchmark, s.corpus_lang, s.segment_id): s for s in segments}


def attach_text(
    rows: Sequence[Mapping[str, Any]], index: Mapping[SegmentKey, Segment]
) -> list[dict[str, Any]]:
    """Add source and reference text to hypothesis rows, looked up by benchmark, language and id."""
    joined = []
    for row in rows:
        segment = index[(row["benchmark"], row["corpus_lang"], row["segment_id"])]
        joined.append({**row, "source": segment.source, "reference": segment.reference})
    return joined


def public_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop source and reference text: they are recovered from the public benchmarks by id."""
    return [{key: value for key, value in row.items() if key not in PRIVATE_FIELDS} for row in rows]


def load_shard_segments(
    shard: Shard, benchmarks: Mapping[str, BenchmarkConfig], language_map: LanguageMap
) -> dict[str, list[Segment]]:
    """Segments of every language in the shard, keyed by corpus code."""
    config = benchmarks[shard.benchmark]
    return {
        lang: load_segments(config, lang, language_map.benchmark_code(lang, shard.benchmark))
        for lang in shard.langs
    }


def generate_shard(
    shard: Shard,
    *,
    model: ModelConfig,
    backend: Any,
    cache: OutputCache,
    benchmarks: Mapping[str, BenchmarkConfig],
    language_map: LanguageMap,
    iso: Callable[[str], str | None],
    decoding: Decoding,
    out_file: Path,
    run: Mapping[str, Any] | None = None,
) -> None:
    """Translate one shard and write its hypothesis records with the run settings."""
    segments = load_shard_segments(shard, benchmarks, language_map)
    translations: list[Translation] = []
    for lang in shard.langs:
        translations.extend(
            translate(segments[lang], model=model, backend=backend, cache=cache, decoding=decoding, iso=iso)
        )
    stamped = {**dict(run or {}), "generated_at": created_now()}
    write_table(out_file, hypothesis_rows(translations, stamped))


def score_shard(
    hypothesis_file: Path,
    out_file: Path,
    *,
    shard: Shard,
    benchmarks: Mapping[str, BenchmarkConfig],
    language_map: LanguageMap,
    scorers: Scorers,
) -> None:
    """Score one shard: chrF++ everywhere, COMET-22 and MetricX-24 where covered, GlotLID for every output."""
    from xx_to_en.metrics import segment_chrf_pp
    from xx_to_en.scoring import comet_inputs, is_neural_covered, merge_scores, metricx_inputs, scatter

    rows = attach_text(
        read_table(hypothesis_file), segment_index(_all_segments(shard, benchmarks, language_map))
    )
    neural = [is_neural_covered(language_map.iso639_1(row["corpus_lang"])) for row in rows]
    covered = [row for row, flag in zip(rows, neural, strict=True) if flag]
    comet = scatter(scorers.comet(comet_inputs(covered)), neural)
    metricx = scatter(scorers.metricx(metricx_inputs(covered)), neural)
    chrf = segment_chrf_pp([row["hypothesis"] for row in rows], [row["reference"] for row in rows])
    codes = scorers.glotlid([row["hypothesis"] for row in rows])
    merged = merge_scores(rows, chrf=chrf, comet=comet, metricx=metricx, codes=codes, neural=neural)
    write_table(out_file, public_rows(merged))


def _all_segments(
    shard: Shard, benchmarks: Mapping[str, BenchmarkConfig], language_map: LanguageMap
) -> list[Segment]:
    loaded = load_shard_segments(shard, benchmarks, language_map)
    return [segment for segments in loaded.values() for segment in segments]
