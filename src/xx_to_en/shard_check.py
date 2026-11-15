"""List Tatoeba result shards whose stored source sentences no longer match the pinned Tatoeba files.

Reads each shard's (corpus_lang, segment_id, source) and compares it with the segments the loader
produces now. A shard whose ids or sentences differ was generated from different test-file content.

Run where the Hub is reachable (a Grid'5000 CPU job), e.g.:
    uv run xx2en check-shards
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from xx_to_en.cli import Project
from xx_to_en.sources import load_segments

REPO = "NoeFlandre/llms-xx-to-en-translation"
ROOT = Path(__file__).resolve().parents[2]


def tatoeba_shards() -> list[str]:
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(
        f for f in files if f.startswith("results/") and "/tatoeba_xen/" in f and f.endswith(".parquet")
    )


def mismatches(path: str, project: Project, cache: dict[str, dict[str, str]]) -> list[str]:
    local = hf_hub_download(REPO, path, repo_type="dataset")
    table = pq.read_table(local, columns=["corpus_lang", "segment_id", "source"]).to_pylist()
    by_lang: dict[str, dict[str, str]] = {}
    for row in table:
        by_lang.setdefault(row["corpus_lang"], {})[row["segment_id"]] = row["source"]
    return [corpus for corpus, stored in by_lang.items() if _differs(corpus, stored, project, cache)]


def _differs(corpus: str, stored: dict[str, str], project: Project, cache: dict[str, dict[str, str]]) -> bool:
    if corpus not in cache:
        cache[corpus] = _pinned_segments(corpus, project)
    current = cache[corpus]
    return any(current.get(sid) != text for sid, text in stored.items())


def _pinned_segments(corpus: str, project: Project) -> dict[str, str]:
    code = project.language_map.benchmark_code(corpus, "tatoeba_xen")
    segments = load_segments(project.benchmarks["tatoeba_xen"], corpus, code) if code else []
    return {s.segment_id: s.source for s in segments}


def run() -> str:
    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    cache: dict[str, dict[str, str]] = {}
    bad = 0
    shards = tatoeba_shards()
    for path in shards:
        problems = mismatches(path, project, cache)
        if problems:
            bad += 1
            print(f"MISMATCH {path} {sorted(set(problems))}", flush=True)
    return f"checked {len(shards)} shards, mismatched {bad}"
