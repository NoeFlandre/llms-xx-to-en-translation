"""Dataset card rendering: results, per-language winners and trade-off plots. Plain Markdown."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

PENDING = "—"


@dataclass(frozen=True)
class ModelScore:
    model: str
    chrf_pp: float
    overall_chrf_pp: float
    comet22: float | None
    metricx24: float | None
    failure_rate: float
    segments: int


def fmt(value: float | None, digits: int) -> str:
    return PENDING if value is None else f"{value:.{digits}f}"


def results_table(rows: Sequence[ModelScore]) -> str:
    if not rows:
        return "_No scores yet._"
    lines = [
        "| Model | chrF++ ↑ | Volume-weighted chrF++ ↑ | COMET-22 ↑ | MetricX-24 ↓ | Failures | Segments |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row.model} | {row.chrf_pp:.1f} | {row.overall_chrf_pp:.1f} | {fmt(row.comet22, 3)} "
        f"| {fmt(row.metricx24, 2)} | {100 * row.failure_rate:.1f}% | {row.segments:,} |"
        for row in sorted(rows, key=lambda r: -r.overall_chrf_pp)
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


TRADE_OFF_NOTE = (
    "Each point is one model. The red line is the Pareto frontier: no other model is both smaller and "
    "better. Parameters are the total parameter count from each model's config."
)


def render(
    updated: str,
    results: str,
    winners: str,
    trade_offs: str = "",
    split_note: str = "",
) -> str:
    split = f"{split_note}\n\n" if split_note else ""
    trade = f"## Trade-offs\n\n{TRADE_OFF_NOTE}\n\n{trade_offs}\n\n" if trade_offs else ""
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

Open translation models, evaluated on GeoReSeT corpus languages translated into English.
[Code](https://github.com/NoeFlandre/llms-xx-to-en-translation) · updated {updated}

## Results

{results}

{split}{winners}

{trade}## Method

- Benchmarks: FLORES+ devtest, NTREX-128, WMT22 and WMT23 (X→EN), and Tatoeba.
  Shards are 8 languages of one benchmark.
- Decoding: greedy, seed 0, thinking off, output budget 3× source tokens.
- Metrics: chrF++ (sacrebleu); COMET-22 `Unbabel/wmt22-comet-da`; MetricX-24 hybrid XL, reference-based.
- Failures: segments flagged as empty, repetition loops, length outliers, untranslated copies,
  or wrong-language output (GlotLID v4).
- Volume-weighted scores weight each language by its corpus rows.

## Data

- `results/<model>/<benchmark>/<shard>.parquet`: hypotheses and token counts
- `scores/<model>/<benchmark>/<shard>.parquet`: per-segment scores and failure flags
- `manifests/<model>/job-<id>.json`: model config and run settings per job

Each row has `source`, `reference` (benchmark ground truth), `hypothesis` (generated English) and
provenance columns such as `model_revision`, `decoding`, `seed`, `gpu` and `code_commit`. Source and
reference text are included so each shard stands alone; benchmark licences apply to them (FLORES+
CC-BY-SA-4.0, Tatoeba CC-BY, NTREX and WMT per their dataset cards).
"""
