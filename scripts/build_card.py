#!/usr/bin/env python3
"""Rebuild the dataset card from the files in the dataset, save it, and upload it as README.md.

uv run python scripts/build_card.py            # write docs/dataset_card.md and upload it
uv run python scripts/build_card.py --no-upload
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from xx_to_en.card import Cell, ModelScore, progress_table, render, results_table, winners_table
from xx_to_en.cli import Project
from xx_to_en.languages import corpus_languages
from xx_to_en.plan import build_shards
from xx_to_en.report import per_language_winner, volume_weighted_mean
from xx_to_en.results_store import repo_files, score_path, shard_path
from xx_to_en.tables import read_table

ROOT = Path(__file__).resolve().parents[1]
REPO = "NoeFlandre/llms-xx-to-en-translation"
FAILURE_FLAGS = ("empty", "repetition_loop", "length_ratio_outlier", "untranslated_copy", "wrong_language")


def progress(project: Project, files: set[str]) -> dict[str, dict[str, Cell]]:
    grid: dict[str, dict[str, Cell]] = {}
    for name, model in project.models.items():
        cells: dict[str, Cell] = {}
        if model.official_languages != "pending":
            shards = build_shards(name, project.plan(model))
            for benchmark in project.benchmarks:
                own = [s for s in shards if s.benchmark == benchmark]
                cells[benchmark] = Cell(
                    generated=sum(shard_path(s) in files for s in own),
                    scored=sum(score_path(s) in files for s in own),
                    total=len(own),
                )
        grid[name] = cells
    return grid


def score_rows(files: set[str]) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(f for f in files if f.startswith("scores/")):
        model = path.split("/")[1]
        local = hf_hub_download(repo_id=REPO, filename=path, repo_type="dataset")
        rows[model].extend(read_table(Path(local)))
    return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(
    project: Project, rows: dict[str, list[dict]]
) -> tuple[list[ModelScore], dict[str, tuple[str, float]]]:
    volumes = corpus_languages(project.inventory.corpus_rows)
    summaries: list[ModelScore] = []
    per_language: dict[str, dict[str, float]] = {}
    for model, records in rows.items():
        by_lang: dict[str, list[float]] = defaultdict(list)
        for record in records:
            by_lang[record["corpus_lang"]].append(record["chrf_pp"])
        lang_means = {lang: sum(values) / len(values) for lang, values in by_lang.items()}
        per_language[model] = lang_means
        comet = [r["comet22"] for r in records if r.get("comet22") is not None]
        metricx = [r["metricx24"] for r in records if r.get("metricx24") is not None]
        failures = sum(any(r.get(flag) for flag in FAILURE_FLAGS) for r in records)
        summaries.append(
            ModelScore(
                model=model,
                chrf_pp=mean([r["chrf_pp"] for r in records]) or 0.0,
                overall_chrf_pp=volume_weighted_mean(lang_means, volumes) if lang_means else 0.0,
                comet22=mean(comet),
                metricx24=mean(metricx),
                failure_rate=failures / len(records),
                segments=len(records),
            )
        )
    winners = {}
    if per_language:
        best = per_language_winner({m: lm for m, lm in per_language.items()})
        winners = {lang: (model, per_language[model][lang]) for lang, model in best.items()}
    return summaries, winners


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args(argv)
    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    files = repo_files(REPO)
    grid = progress(project, files)
    summaries, winners = summarize(project, score_rows(files))
    updated = datetime.now().strftime("%Y-%m-%d %H:%M CET")
    card = render(
        updated,
        progress_table(grid),
        results_table(summaries),
        winners_table(winners),
    )
    out = ROOT / "docs" / "dataset_card.md"
    out.write_text(card, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}; scored models: {len(summaries)}")
    if not args.no_upload:
        HfApi().upload_file(
            path_or_fileobj=str(out),
            path_in_repo="README.md",
            repo_id=REPO,
            repo_type="dataset",
            commit_message=f"Update dataset card ({updated})",
        )
        print("uploaded README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
