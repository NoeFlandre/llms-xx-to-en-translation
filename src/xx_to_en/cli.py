"""Command-line entry point.

- ``xx2en check-configs``: validate every YAML config.
- ``xx2en coverage``: write the language x benchmark coverage matrix.
- ``xx2en plan --model M``: list the benchmark x language work the model is allowed to run.
- ``xx2en run --model M``: translate on a node (needs the ``gpu`` extra and a GPU).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from xx_to_en.config import BenchmarkConfig, ModelConfig, load_benchmarks, load_language_map, load_models
from xx_to_en.coverage import build_coverage, render_markdown, summarize
from xx_to_en.inventory import Inventory, load_inventory
from xx_to_en.languages import LanguageMap, corpus_languages
from xx_to_en.plan import build_shards, pending
from xx_to_en.runner import WorkItem, plan_work


def check_configs(root: Path) -> str:
    """Validate every config under ``root`` and describe what was loaded."""
    models = load_models(root / "models")
    benchmarks = load_benchmarks(root / "benchmarks")
    load_language_map(root / "languages.yaml")
    return f"ok: {len(models)} models, {len(benchmarks)} benchmarks, languages.yaml"


def write_coverage(configs: Path, inventory_path: Path, out: Path) -> str:
    """Build the language x benchmark matrix and write it as Markdown."""
    inventory = load_inventory(inventory_path)
    language_map = LanguageMap(load_language_map(configs / "languages.yaml"))
    rows = build_coverage(inventory, language_map)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(rows), encoding="utf-8")
    summary = summarize(rows)
    return f"wrote {out} ({summary.configs} configs, {summary.uncovered_rows:,} uncovered rows)"


class Project:
    """Loaded configs and inventory for one configs directory."""

    def __init__(self, configs: Path, inventory_path: Path) -> None:
        self.models: dict[str, ModelConfig] = load_models(configs / "models")
        self.benchmarks: dict[str, BenchmarkConfig] = load_benchmarks(configs / "benchmarks")
        self.language_map = LanguageMap(load_language_map(configs / "languages.yaml"))
        self.inventory: Inventory = load_inventory(inventory_path)

    def model(self, name: str) -> ModelConfig:
        if name not in self.models:
            raise SystemExit(f"unknown model {name!r}; known: {', '.join(sorted(self.models))}")
        return self.models[name]

    def supported_languages(self, model: ModelConfig) -> frozenset[str]:
        """Corpus codes the model officially supports. Refuses models whose list is still pending."""
        if model.official_languages == "pending":
            raise SystemExit(
                f"{model.name}: official_languages is pending; fill it from the model card first"
            )
        if model.official_languages == "all":
            return frozenset(corpus_languages(self.inventory.corpus_rows))
        return frozenset(model.official_languages)

    def plan(self, model: ModelConfig, benchmarks: list[str] | None = None) -> list[WorkItem]:
        names = benchmarks or sorted(self.benchmarks)
        return plan_work(
            names,
            corpus_languages(self.inventory.corpus_rows),
            self.inventory,
            self.language_map,
            supported=self.supported_languages(model),
        )


def plan_command(project: Project, model_name: str, benchmarks: list[str] | None) -> str:
    model = project.model(model_name)
    items = project.plan(model, benchmarks)
    per_benchmark: dict[str, int] = {}
    for item in items:
        per_benchmark[item.benchmark] = per_benchmark.get(item.benchmark, 0) + 1
    return json.dumps(
        {"model": model.name, "work_items": len(items), "per_benchmark": per_benchmark}, indent=2
    )


def make_backend(model: ModelConfig, kind: str, *, seed: int) -> Any:
    """Build the generation backend. Imports are deferred so planning works without the gpu extra."""
    if kind == "vllm":
        from xx_to_en.backends import VllmBackend

        return VllmBackend(model, seed=seed)
    if kind == "transformers":
        from xx_to_en.backends import TransformersBackend

        # NLLB needs its target language forced; MADLAD takes the target from a text prefix instead.
        return TransformersBackend(model, seed=seed, force_target_lang=model.family == "nllb")
    raise SystemExit(f"unknown backend {kind!r}")


def run_command(project: Project, args: argparse.Namespace) -> str:
    from xx_to_en.backends import Backend  # noqa: F401 - type reference for readers
    from xx_to_en.cache import OutputCache
    from xx_to_en.runner import Decoding, run_work_item
    from xx_to_en.sources import load_segments

    model = project.model(args.model)
    if model.access == "not_accepted":
        raise SystemExit(f"{model.name}: access not accepted for this account")
    items = [
        item
        for item in project.plan(model, args.benchmarks)
        if not args.langs or item.corpus_lang in args.langs
    ]
    if not items:
        raise SystemExit("nothing to run for the selected model, benchmarks and languages")
    backend = make_backend(model, args.backend, seed=args.seed)
    cache = OutputCache(Path(args.cache))
    decoding = Decoding(seed=args.seed)
    out_dir = Path(args.out)
    lines = []
    for item in items:
        segments = load_segments(project.benchmarks[item.benchmark], item.corpus_lang, item.bench_lang)
        if args.limit:
            segments = segments[: args.limit]
        summary = run_work_item(
            item,
            segments,
            model=model,
            backend=backend,
            cache=cache,
            out_dir=out_dir,
            decoding=decoding,
            iso=project.language_map.iso639_1,
        )
        lines.append(json.dumps(summary, ensure_ascii=False))
        print(lines[-1], flush=True)
    return f"{len(items)} work items done"


def work_command(project: Project, args: argparse.Namespace) -> str:
    """Generation phase: translate the pending shards of one model, upload each, within the budget."""
    from xx_to_en.cache import OutputCache
    from xx_to_en.results_store import repo_files, shard_path, upload
    from xx_to_en.runner import Decoding
    from xx_to_en.worker import generate_shard, run_budgeted

    started = time.monotonic()
    model = project.model(args.model)
    if model.access == "not_accepted":
        raise SystemExit(f"{model.name}: access not accepted for this account")
    files = repo_files(args.results_repo)
    todo = pending(
        build_shards(model.name, project.plan(model)),
        is_done=lambda shard: shard_path(shard) in files,
    )
    if not todo:
        return f"{model.name}: nothing pending"
    backend = make_backend(model, model.backend, seed=args.seed)
    cache = OutputCache(args.work / "cache")
    budget_s = max(0.0, args.budget_min * 60 - (time.monotonic() - started))

    def generate(shard: Any) -> None:
        out_file = args.work / "hyp" / shard_path(shard)
        generate_shard(
            shard,
            model=model,
            backend=backend,
            cache=cache,
            benchmarks=project.benchmarks,
            language_map=project.language_map,
            iso=project.language_map.iso639_1,
            decoding=Decoding(seed=args.seed),
            out_file=out_file,
        )
        upload(args.results_repo, out_file, shard_path(shard))
        print(f"uploaded {shard_path(shard)}", flush=True)

    finished = run_budgeted(todo, budget_s=budget_s, step=generate)
    return f"{model.name}: generated {len(finished)} of {len(todo)} pending shards"


def score_command(project: Project, args: argparse.Namespace) -> str:
    """Scoring phase: score uploaded hypotheses of one model that have no scores yet, within the budget."""
    from xx_to_en.metric_runtime import Scorers
    from xx_to_en.results_store import download, repo_files, score_path, shard_path, upload
    from xx_to_en.worker import run_budgeted, score_shard

    started = time.monotonic()
    model = project.model(args.model)
    files = repo_files(args.results_repo)
    todo = [
        shard
        for shard in build_shards(model.name, project.plan(model))
        if shard_path(shard) in files and score_path(shard) not in files
    ]
    if not todo:
        return f"{model.name}: nothing to score"
    scorers = Scorers(
        metricx_repo=args.metricx_repo,
        work=args.work / "metricx",
        comet_batch=args.comet_batch,
        metricx_batch=args.metricx_batch,
    )
    budget_s = max(0.0, args.budget_min * 60 - (time.monotonic() - started))

    def score_one(shard: Any) -> None:
        hypotheses = args.work / "hyp" / shard_path(shard)
        if not hypotheses.exists():
            download(args.results_repo, shard_path(shard), hypotheses)
        out_file = args.work / "scores" / score_path(shard)
        score_shard(
            hypotheses,
            out_file,
            shard=shard,
            benchmarks=project.benchmarks,
            language_map=project.language_map,
            scorers=scorers,
        )
        upload(args.results_repo, out_file, score_path(shard))
        print(f"uploaded {score_path(shard)}", flush=True)

    finished = run_budgeted(todo, budget_s=budget_s, step=score_one)
    return f"{model.name}: scored {len(finished)} of {len(todo)} pending shards"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xx2en", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-configs", help="validate all YAML configs")
    check.add_argument("--root", type=Path, default=Path("configs"))

    coverage = commands.add_parser("coverage", help="write the language x benchmark coverage matrix")
    coverage.add_argument("--configs", type=Path, default=Path("configs"))
    coverage.add_argument("--inventory", type=Path, default=Path("data/benchmark_inventory.json"))
    coverage.add_argument("--out", type=Path, default=Path("docs/coverage_matrix.md"))

    for name, help_text in (("plan", "list allowed work for a model"), ("run", "translate on a node")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--model", required=True)
        sub.add_argument("--benchmarks", nargs="+", default=None)
        sub.add_argument("--configs", type=Path, default=Path("configs"))
        sub.add_argument("--inventory", type=Path, default=Path("data/benchmark_inventory.json"))
        if name == "run":
            sub.add_argument("--langs", nargs="+", default=None, help="corpus codes, e.g. fra_Latn")
            sub.add_argument("--limit", type=int, default=0, help="segments per work item (0 = all)")
            sub.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--out", default="outputs")
            sub.add_argument("--cache", default="cache")

    work = commands.add_parser("work", help="translate pending shards of one model (generation environment)")
    work.add_argument("--model", required=True)
    work.add_argument("--results-repo", required=True)
    work.add_argument("--work", type=Path, required=True)
    work.add_argument("--budget-min", type=float, default=38.0, help="wall-clock budget for this job")
    work.add_argument("--seed", type=int, default=0)
    work.add_argument("--configs", type=Path, default=Path("configs"))
    work.add_argument("--inventory", type=Path, default=Path("data/benchmark_inventory.json"))

    score = commands.add_parser("score", help="score generated shards of one model (metrics environment)")
    score.add_argument("--model", required=True)
    score.add_argument("--results-repo", required=True)
    score.add_argument("--work", type=Path, required=True)
    score.add_argument("--metricx-repo", type=Path, default=Path.home() / "src" / "metricx")
    score.add_argument("--comet-batch", type=int, default=64)
    score.add_argument("--metricx-batch", type=int, default=8)
    score.add_argument("--budget-min", type=float, default=20.0, help="wall-clock budget for this job")
    score.add_argument("--configs", type=Path, default=Path("configs"))
    score.add_argument("--inventory", type=Path, default=Path("data/benchmark_inventory.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check-configs":
        print(check_configs(args.root))
    elif args.command == "coverage":
        print(write_coverage(args.configs, args.inventory, args.out))
    elif args.command == "work":
        print(work_command(Project(args.configs, args.inventory), args))
    elif args.command == "score":
        print(score_command(Project(args.configs, args.inventory), args))
    else:
        project = Project(args.configs, args.inventory)
        if args.command == "plan":
            print(plan_command(project, args.model, args.benchmarks))
        else:
            print(run_command(project, args))
    return 0
