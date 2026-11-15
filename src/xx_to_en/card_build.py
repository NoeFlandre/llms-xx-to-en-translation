"""Rebuild the dataset card from the files in the dataset, save it, and upload it as README.md.

uv run xx2en card              # write docs/dataset_card.md and upload it
uv run xx2en card --no-upload
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from xx_to_en.card import Cell, ModelScore, progress_table, render, results_table, winners_table
from xx_to_en.cli import Project
from xx_to_en.config import ModelConfig
from xx_to_en.languages import corpus_languages
from xx_to_en.plan import Shard
from xx_to_en.plots import Point, tradeoff_svg
from xx_to_en.report import per_language_winner, volume_weighted_mean
from xx_to_en.results_store import repo_files, score_path, shard_path
from xx_to_en.tables import read_table

ROOT = Path(__file__).resolve().parents[2]
REPO = "NoeFlandre/llms-xx-to-en-translation"
FAILURE_FLAGS = ("empty", "repetition_loop", "length_ratio_outlier", "untranslated_copy", "wrong_language")


def progress(project: Project, files: set[str]) -> dict[str, dict[str, Cell]]:
    return {
        name: _model_cells(project, model, files) for name, model in project.models.items() if model.enabled
    }


def _model_cells(project: Project, model: ModelConfig, files: set[str]) -> dict[str, Cell]:
    shards = project.shards(model, files)
    return {
        benchmark: _cell([s for s in shards if s.benchmark == benchmark], files)
        for benchmark in project.benchmarks
    }


def _cell(own: list[Shard], files: set[str]) -> Cell:
    return Cell(
        generated=sum(shard_path(s) in files for s in own),
        scored=sum(score_path(s) in files for s in own),
        total=len(own),
    )


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
) -> tuple[list[ModelScore], dict[str, tuple[str, float]], int]:
    """Per-model scores. Overall chrF++ uses only the languages every scored model covers, so scopes match."""
    volumes = corpus_languages(project.inventory.corpus_rows)
    per_language = {model: _language_means(records) for model, records in rows.items()}
    common = set.intersection(*(set(lm) for lm in per_language.values())) if per_language else set()
    summaries = [
        _model_score(model, records, per_language[model], common, volumes) for model, records in rows.items()
    ]
    return summaries, _winners(per_language), len(common)


def _language_means(records: list[dict]) -> dict[str, float]:
    by_lang: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_lang[record["corpus_lang"]].append(record["chrf_pp"])
    return {lang: sum(v) / len(v) for lang, v in by_lang.items()}


def _model_score(
    model: str, records: list[dict], language_means: dict[str, float], common: set[str], volumes: dict
) -> ModelScore:
    return ModelScore(
        model=model,
        chrf_pp=mean([r["chrf_pp"] for r in records]) or 0.0,
        overall_chrf_pp=_overall_chrf(language_means, common, volumes),
        comet22=_metric_mean(records, "comet22"),
        metricx24=_metric_mean(records, "metricx24"),
        failure_rate=_failure_rate(records),
        segments=len(records),
    )


def _overall_chrf(language_means: dict[str, float], common: set[str], volumes: dict) -> float:
    common_scores = {lang: language_means[lang] for lang in common}
    return volume_weighted_mean(common_scores, volumes) if common_scores else 0.0


def _metric_mean(records: list[dict], key: str) -> float | None:
    return mean([r[key] for r in records if r.get(key) is not None])


def _failure_rate(records: list[dict]) -> float:
    return sum(any(r.get(flag) for flag in FAILURE_FLAGS) for r in records) / len(records)


def _winners(per_language: dict[str, dict[str, float]]) -> dict[str, tuple[str, float]]:
    if not per_language:
        return {}
    best = per_language_winner(per_language)
    return {lang: (model, per_language[model][lang]) for lang, model in best.items()}


TRADE_OFFS = (
    ("volume_chrf", "Volume-weighted chrF++", "overall_chrf_pp", True),
    ("comet22", "COMET-22", "comet22", True),
    ("metricx24", "MetricX-24", "metricx24", False),
)
PLOTS_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def trade_off_plots(summaries: list[ModelScore], project: Project) -> tuple[str, dict[str, str]]:
    """Markdown that embeds one trade-off plot per metric, and the SVG files behind it by repository path."""
    lines: list[str] = []
    files: dict[str, str] = {}
    for slug, title, field, higher_is_better in TRADE_OFFS:
        points = _trade_off_points(summaries, project, field)
        svg = tradeoff_svg(title, points, higher_is_better)
        if not svg:
            continue
        path = f"plots/{slug}.svg"
        files[path] = svg
        lines.append(f"![{title} against parameters]({PLOTS_BASE}/{path})")
    return "\n\n".join(lines), files


def _trade_off_points(summaries: list[ModelScore], project: Project, field: str) -> list[Point]:
    return [
        Point(s.model, project.models[s.model].params_b, getattr(s, field))
        for s in summaries
        if getattr(s, field) is not None
    ]


def run(no_upload: bool) -> None:
    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    files = repo_files(REPO)
    grid = progress(project, files)
    summaries, winners, common_count = summarize(project, score_rows(files))
    updated = datetime.now().strftime("%Y-%m-%d %H:%M CET")
    trade_offs, plot_files = trade_off_plots(summaries, project)
    card = render(
        updated,
        progress_table(grid),
        (
            f"Overall chrF++ uses the {common_count} languages every scored model covers.\n\n"
            if summaries
            else ""
        )
        + results_table(summaries),
        winners_table(winners),
        trade_offs=trade_offs,
    )
    out = ROOT / "docs" / "dataset_card.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card, encoding="utf-8")
    for path, svg in plot_files.items():
        local = ROOT / "docs" / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(svg, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}; scored models: {len(summaries)}; plots: {len(plot_files)}")
    if not no_upload:
        api = HfApi()
        for path, svg in plot_files.items():
            api.upload_file(
                path_or_fileobj=svg.encode("utf-8"),
                path_in_repo=path,
                repo_id=REPO,
                repo_type="dataset",
                commit_message=f"Trade-off plot {path} ({updated})",
            )
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo="README.md",
            repo_id=REPO,
            repo_type="dataset",
            commit_message=f"Update dataset card ({updated})",
        )
        print("uploaded README.md")
