"""Fetch benchmark data from the Hugging Face Hub and turn it into segments.

Runs on compute nodes. Heavy imports (``datasets``, ``huggingface_hub``) are deferred so the
package imports cleanly without the ``gpu`` extra installed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from xx_to_en.benchmarks import (
    ENGLISH_BENCHMARK,
    Segment,
    pair_by_position,
    pair_flores,
    pair_tatoeba,
    pair_wmt,
)
from xx_to_en.config import BenchmarkConfig

WMT_BENCHMARKS = frozenset({"wmt22_xen", "wmt23_xen"})


def _hub_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")


def _lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return handle.read().splitlines()


def _load_rows(repo: str, config: str) -> list[dict[str, Any]]:
    from datasets import load_dataset  # ty: ignore[import-not-found]

    return [dict(row) for row in load_dataset(repo, config, split="test")]


def _texts(repo: str, config: str) -> list[str]:
    return [row["text"] for row in _load_rows(repo, config)]


def _load_flores(benchmark: BenchmarkConfig, corpus_lang: str, bench_lang: str) -> list[Segment]:
    source = _lines(_hub_file(benchmark.hf_repo, f"devtest/{bench_lang}.jsonl"))
    reference = _lines(_hub_file(benchmark.hf_repo, f"devtest/{ENGLISH_BENCHMARK}.jsonl"))
    return pair_flores(
        source, reference, benchmark=benchmark.name, corpus_lang=corpus_lang, bench_lang=bench_lang
    )


def _load_ntrex(benchmark: BenchmarkConfig, corpus_lang: str, bench_lang: str) -> list[Segment]:
    source = _texts(benchmark.hf_repo, bench_lang)
    reference = _texts(benchmark.hf_repo, ENGLISH_BENCHMARK)
    return pair_by_position(
        source, reference, benchmark=benchmark.name, corpus_lang=corpus_lang, bench_lang=bench_lang
    )


def _load_wmt(benchmark: BenchmarkConfig, corpus_lang: str, bench_lang: str) -> list[Segment]:
    rows = _load_rows(benchmark.hf_repo, f"{bench_lang}-en")
    return pair_wmt(
        rows,
        config=f"{bench_lang}-en",
        benchmark=benchmark.name,
        corpus_lang=corpus_lang,
        bench_lang=bench_lang,
    )


def _load_tatoeba(benchmark: BenchmarkConfig, corpus_lang: str, bench_lang: str) -> list[Segment]:
    lines = _tatoeba_lines(benchmark.hf_repo, bench_lang)
    return pair_tatoeba(lines, benchmark=benchmark.name, corpus_lang=corpus_lang, bench_lang=bench_lang)


def _tatoeba_lines(repo: str, code: str) -> list[str]:
    from huggingface_hub.utils import EntryNotFoundError

    for pair in (f"{code}-eng", f"eng-{code}"):
        try:
            return _lines(_hub_file(repo, f"data/test/tatoeba-test.{pair}.tsv"))
        except EntryNotFoundError:
            continue
    raise FileNotFoundError(f"no Tatoeba test file for {code} against English")


LOADERS: dict[str, Callable[[BenchmarkConfig, str, str], list[Segment]]] = {
    "flores_plus_devtest": _load_flores,
    "ntrex_128": _load_ntrex,
    "tatoeba_xen": _load_tatoeba,
}
LOADERS.update({name: _load_wmt for name in WMT_BENCHMARKS})


def load_segments(benchmark: BenchmarkConfig, corpus_lang: str, bench_lang: str) -> list[Segment]:
    """Fetch and align one benchmark for one language. Raises KeyError for unknown benchmarks."""
    loader = LOADERS[benchmark.name]
    return loader(benchmark, corpus_lang, bench_lang)


def describe(segments: list[Segment]) -> str:
    """Short JSON summary for logs."""
    return json.dumps({"segments": len(segments), "first_id": segments[0].segment_id if segments else None})
