"""Dataset card rendering: a progress grid, a results table and per-language winners. Plain Markdown."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

BENCHMARK_LABELS: Mapping[str, str] = {
    "flores_plus_devtest": "FLORES+",
    "ntrex_128": "NTREX-128",
    "wmt22_xen": "WMT22",
    "wmt23_xen": "WMT23",
    "tatoeba_xen": "Tatoeba",
}
PENDING = "—"


@dataclass(frozen=True)
class Cell:
    generated: int
    scored: int
    total: int


@dataclass(frozen=True)
class ModelScore:
    model: str
    chrf_pp: float
    overall_chrf_pp: float
    comet22: float | None
    metricx24: float | None
    failure_rate: float
    segments: int


def bar(done: int, total: int, width: int = 6) -> str:
    if total == 0:
        return PENDING
    filled = round(width * done / total)
    return "█" * filled + "░" * (width - filled)


def fmt(value: float | None, digits: int) -> str:
    return PENDING if value is None else f"{value:.{digits}f}"


def progress_table(grid: Mapping[str, Mapping[str, Cell]]) -> str:
    """Shards generated per model and benchmark, with a bar for each cell."""
    header = "| Model | " + " | ".join(BENCHMARK_LABELS.values()) + " |"
    lines = [header, "|---|" + "---|" * len(BENCHMARK_LABELS)]
    for model, cells in grid.items():
        row = []
        for benchmark in BENCHMARK_LABELS:
            cell = cells.get(benchmark)
            if cell is None or cell.total == 0:
                row.append(PENDING)
            else:
                row.append(f"{bar(cell.generated, cell.total)} {cell.generated}/{cell.total}")
        lines.append(f"| {model} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def results_table(rows: Sequence[ModelScore]) -> str:
    if not rows:
        return "_No scores yet. Hypotheses are generated first; scores follow on bf16 GPUs._"
    lines = [
        "| Model | chrF++ ↑ | Volume-weighted chrF++ ↑ | COMET-22 ↑ | MetricX-24 ↓ | Failures | Segments |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: -r.overall_chrf_pp):
        lines.append(
            f"| {row.model} | {row.chrf_pp:.1f} | {row.overall_chrf_pp:.1f} | {fmt(row.comet22, 3)} "
            f"| {fmt(row.metricx24, 2)} | {100 * row.failure_rate:.1f}% | {row.segments:,} |"
        )
    return "\n".join(lines)


def winners_table(winners: Mapping[str, tuple[str, float]]) -> str:
    """Best model per language, in a collapsible block."""
    if not winners:
        return ""
    lines = [
        "<details><summary>Best model per language</summary>",
        "",
        "| Language | Model | chrF++ |",
        "|---|---|---:|",
    ]
    for language, (model, score) in sorted(winners.items()):
        lines.append(f"| {language} | {model} | {score:.1f} |")
    lines += ["", "</details>"]
    return "\n".join(lines)


def render(updated: str, progress: str, results: str, winners: str) -> str:
    return f"""---
license: cc-by-sa-4.0
pretty_name: llms-xx-to-en-translation
task_categories:
  - translation
tags:
  - machine-translation
  - benchmark
  - x-to-en
---

# llms-xx-to-en-translation

Open models translating GeoReSeT corpus languages into English.
[Code](https://github.com/NoeFlandre/llms-xx-to-en-translation) · updated {updated}

## Progress

{progress}

Each cell: shards generated / total. A shard is 8 languages of one benchmark.

## Results

{results}

{winners}

## Files

- `results/<model>/<benchmark>/<shard>.parquet`: hypotheses and token counts
- `scores/<model>/<benchmark>/<shard>.parquet`: per-segment scores and failure flags
- `manifests/<model>/job-<id>.json`: full model config and run settings for each job

Each row carries its own provenance: `model_revision`, `prompt_template`, `prompt_version`, `prompt_sha256`,
`decoding`, `seed`, `budget_*`, `max_new_tokens`, `dtype`, `backend`, `software`, `gpu`, `node`, `job_id`,
`code_commit`, `generated_at`. Generated text is in `hypothesis`; the untouched model output is in
`raw_hypothesis`, with `finish_reason` and `truncated`.

Source and reference text are included in every row, so each shard stands alone.
Benchmark licences apply to those texts: FLORES+ CC-BY-SA-4.0, Tatoeba CC-BY,
NTREX and WMT per their dataset cards.

## Method

X→EN · greedy decoding · seed 0 · thinking off · output budget 3× source tokens.
COMET-22 `Unbabel/wmt22-comet-da` · MetricX-24 hybrid XL (reference-based) · GlotLID v4 · chrF++ (sacrebleu).
Volume-weighted scores weight each language by its corpus rows.
"""
