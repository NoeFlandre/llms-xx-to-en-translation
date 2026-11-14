#!/usr/bin/env python3
"""List Tatoeba result shards whose stored source sentences no longer match the pinned Tatoeba files.

Reads each shard's (corpus_lang, segment_id, source) and compares it with the segments the loader
produces now. A shard whose ids or sentences differ was generated from different test-file content.

Run where the Hub is reachable (a Grid'5000 CPU job), e.g.:
    uv run python scripts/check_tatoeba_shards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from xx_to_en.cli import Project
from xx_to_en.sources import load_segments

REPO = "NoeFlandre/llms-xx-to-en-translation"
ROOT = Path(__file__).resolve().parents[1]


def tatoeba_shards() -> list[str]:
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(
        f for f in files if f.startswith("results/") and "/tatoeba_xen/" in f and f.endswith(".parquet")
    )


def mismatches(path: str, project: Project, cache: dict[str, dict[str, str]]) -> list[str]:
    local = hf_hub_download(REPO, path, repo_type="dataset")
    table = pq.read_table(local, columns=["corpus_lang", "segment_id", "source"]).to_pylist()
    benchmark = project.benchmarks["tatoeba_xen"]
    problems = []
    by_lang: dict[str, dict[str, str]] = {}
    for row in table:
        by_lang.setdefault(row["corpus_lang"], {})[row["segment_id"]] = row["source"]
    for corpus, stored in by_lang.items():
        if corpus not in cache:
            code = project.language_map.benchmark_code(corpus, "tatoeba_xen")
            segments = load_segments(benchmark, corpus, code) if code else []
            cache[corpus] = {s.segment_id: s.source for s in segments}
        current = cache[corpus]
        if any(current.get(sid) != text for sid, text in stored.items()):
            problems.append(corpus)
    return problems


def main() -> int:
    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    cache: dict[str, dict[str, str]] = {}
    bad = 0
    shards = tatoeba_shards()
    for path in shards:
        problems = mismatches(path, project, cache)
        if problems:
            bad += 1
            print(f"MISMATCH {path} {sorted(set(problems))}", flush=True)
    print(f"checked {len(shards)} shards, mismatched {bad}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
