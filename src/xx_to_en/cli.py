"""Command-line entry point.

- ``xx2en check-configs``: validate every YAML config.
- ``xx2en coverage``: write the language x benchmark coverage matrix.
- ``xx2en plan --model M``: list the benchmark x language work the model is allowed to run.
- ``xx2en run --model M``: translate on a node (needs the ``gpu`` extra and a GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from xx_to_en.config import BenchmarkConfig, ModelConfig, load_benchmarks, load_language_map, load_models
from xx_to_en.coverage import build_coverage, render_markdown, summarize
from xx_to_en.inventory import Inventory, load_inventory
from xx_to_en.languages import LanguageMap, corpus_languages
from xx_to_en.plan import Shard, build_shards
from xx_to_en.runner import Decoding, WorkItem, plan_work


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
        """Every corpus language: all models run the same language set, so results compare per language."""
        if not model.enabled:
            raise SystemExit(f"{model.name}: disabled in its config")
        return frozenset(corpus_languages(self.inventory.corpus_rows))

    def plan(self, model: ModelConfig, benchmarks: list[str] | None = None) -> list[WorkItem]:
        names = benchmarks or sorted(self.benchmarks)
        return plan_work(
            names,
            corpus_languages(self.inventory.corpus_rows),
            self.inventory,
            self.language_map,
            supported=self.supported_languages(model),
        )

    def shards(self, model: ModelConfig, files: set[str]) -> list[Shard]:
        """The model's shards, given the files already in the results repository.

        Unfinished shards are cut into parts where the model's config asks (``split_shards``).
        """
        from xx_to_en.results_store import shard_path

        return build_shards(
            model.name,
            self.plan(model),
            splits=model.split_shards,
            is_done=lambda shard: shard_path(shard) in files,
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


def make_backend(model: ModelConfig, kind: str, *, seed: int, dtype: str | None = None) -> Any:
    """Build the generation backend. Imports are deferred so planning works without the gpu extra."""
    if kind == "vllm":
        from xx_to_en.backends import VllmBackend

        return VllmBackend(model, seed=seed, dtype=dtype)
    if kind == "transformers":
        from xx_to_en.backends import TransformersBackend

        # NLLB needs its target language forced; MADLAD takes the target from a text prefix instead.
        return TransformersBackend(model, seed=seed, force_target_lang=model.family == "nllb", dtype=dtype)
    raise SystemExit(f"unknown backend {kind!r}")


def _check_access(model: ModelConfig) -> None:
    if model.access == "not_accepted":
        raise SystemExit(f"{model.name}: access not accepted for this account")


def _select_langs(items: list[WorkItem], langs: list[str] | None) -> list[WorkItem]:
    """Keep the work items of the requested corpus languages (all of them when none are given)."""
    return [item for item in items if not langs or item.corpus_lang in langs]


def _translate_items(
    project: Project, model: ModelConfig, items: list[WorkItem], backend: Any, args: argparse.Namespace
) -> None:
    """Translate each work item, printing its summary record as one JSON line."""
    from xx_to_en.cache import OutputCache
    from xx_to_en.runner import run_work_item
    from xx_to_en.sources import load_segments

    cache = OutputCache(Path(args.cache))
    decoding = Decoding(seed=args.seed)
    out_dir = Path(args.out)
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
            iso=project.language_map.prompt_code,
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def run_command(project: Project, args: argparse.Namespace) -> str:
    from xx_to_en.backends import Backend  # noqa: F401 - type reference for readers

    model = project.model(args.model)
    _check_access(model)
    items = _select_langs(project.plan(model, args.benchmarks), args.langs)
    if not items:
        raise SystemExit("nothing to run for the selected model, benchmarks and languages")
    backend = make_backend(model, args.backend, seed=args.seed)
    _translate_items(project, model, items, backend, args)
    return f"{len(items)} work items done"


def remaining_budget_s(budget_min: float, started: float) -> float:
    """Seconds left in a job's wall-clock budget, counted from the monotonic time ``started``."""
    return max(0.0, budget_min * 60 - (time.monotonic() - started))


def select_shards(shards: list[Shard], benchmark: str, shard_index: int, shard_count: int) -> list[Shard]:
    """Shards of ``benchmark`` (every benchmark when empty) in this job's residue class of shard indexes."""
    return [
        shard
        for shard in shards
        if (not benchmark or shard.benchmark == benchmark) and shard.index % shard_count == shard_index
    ]


def _pending_generation(project: Project, model: ModelConfig, files: set[str]) -> list[Shard]:
    """Shards of the model whose hypotheses are not in the results repository yet."""
    from xx_to_en.results_store import shard_path

    return [shard for shard in project.shards(model, files) if shard_path(shard) not in files]


def _model_revision(model: ModelConfig) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(model.hf_repo).sha or ""


def _generation_run(model: ModelConfig, args: argparse.Namespace, decoding: Decoding) -> dict[str, Any]:
    """Provenance columns repeated on every shard this job generates."""
    from xx_to_en.provenance import run_fields

    return run_fields(
        model,
        revision=_model_revision(model),
        decoding=decoding,
        dtype=args.dtype or model.dtype,
        backend=model.backend,
        env=os.environ,
    )


def _upload_job_manifest(
    args: argparse.Namespace, model: ModelConfig, run: dict[str, Any], uploaded: list[str]
) -> None:
    """Upload this job's manifest: its run settings and the shards it committed."""
    from xx_to_en.provenance import manifest
    from xx_to_en.results_store import upload

    job_manifest = args.work / "manifest.json"
    job_manifest.write_text(json.dumps(manifest(model, run, uploaded), indent=2), encoding="utf-8")
    upload(args.results_repo, job_manifest, f"manifests/{model.name}/job-{run['job_id'] or 'local'}.json")


def work_command(project: Project, args: argparse.Namespace) -> str:
    """Generation phase: translate the pending shards of one model, upload each with its provenance."""
    from xx_to_en.cache import OutputCache
    from xx_to_en.results_store import repo_files, shard_path, upload
    from xx_to_en.worker import generate_shard, run_budgeted

    started = time.monotonic()
    model = project.model(args.model)
    _check_access(model)
    files = repo_files(args.results_repo)
    todo = select_shards(
        _pending_generation(project, model, files), args.benchmark, args.shard_index, args.shard_count
    )
    if not todo:
        return f"{model.name}: nothing pending"
    backend = make_backend(model, model.backend, seed=args.seed, dtype=args.dtype or None)
    cache = OutputCache(args.work / "cache")
    decoding = Decoding(seed=args.seed)
    run = _generation_run(model, args, decoding)
    budget_s = remaining_budget_s(args.budget_min, started)
    uploaded: list[str] = []

    def generate(shard: Shard) -> None:
        out_file = args.work / "hyp" / shard_path(shard)
        generate_shard(
            shard,
            model=model,
            backend=backend,
            cache=cache,
            benchmarks=project.benchmarks,
            language_map=project.language_map,
            iso=project.language_map.prompt_code,
            decoding=decoding,
            out_file=out_file,
            run=run,
        )
        upload(args.results_repo, out_file, shard_path(shard))
        uploaded.append(shard_path(shard))
        print(f"uploaded {shard_path(shard)}", flush=True)

    finished = run_budgeted(todo, budget_s=budget_s, step=generate)
    if uploaded:
        _upload_job_manifest(args, model, run, uploaded)
    return f"{model.name}: generated {len(finished)} of {len(todo)} pending shards"


def _unscored_shards(project: Project, model: ModelConfig, files: set[str]) -> list[Shard]:
    """Shards whose hypotheses are in the results repository and whose scores are not."""
    from xx_to_en.results_store import score_path, shard_path

    return [
        shard
        for shard in project.shards(model, files)
        if shard_path(shard) in files and score_path(shard) not in files
    ]


def _make_scorers(args: argparse.Namespace) -> Any:
    from xx_to_en.metric_runtime import Scorers

    return Scorers(
        metricx_repo=args.metricx_repo,
        work=args.work / "metricx",
        comet_batch=args.comet_batch,
        metricx_batch=args.metricx_batch,
    )


def _score_and_upload(shard: Shard, args: argparse.Namespace, project: Project, scorers: Any) -> None:
    """Score one shard, fetching its hypotheses first when they are not local, and upload the scores."""
    from xx_to_en.results_store import download, score_path, shard_path, upload
    from xx_to_en.worker import score_shard

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


def score_command(project: Project, args: argparse.Namespace) -> str:
    """Scoring phase: score uploaded hypotheses of one model that have no scores yet, within the budget."""
    from xx_to_en.results_store import repo_files
    from xx_to_en.worker import run_budgeted

    started = time.monotonic()
    model = project.model(args.model)
    files = repo_files(args.results_repo)
    todo = select_shards(
        _unscored_shards(project, model, files), args.benchmark, args.shard_index, args.shard_count
    )
    if not todo:
        return f"{model.name}: nothing to score"
    scorers = _make_scorers(args)
    budget_s = remaining_budget_s(args.budget_min, started)
    finished = run_budgeted(
        todo, budget_s=budget_s, step=lambda shard: _score_and_upload(shard, args, project, scorers)
    )
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
    work.add_argument("--dtype", default="", help="override the model dtype, e.g. float16 on Turing GPUs")
    work.add_argument("--benchmark", default="", help="only this benchmark")
    work.add_argument("--shard-index", type=int, default=0, help="this job's residue class of shard indexes")
    work.add_argument("--shard-count", type=int, default=1, help="number of parallel jobs for this model")
    work.add_argument("--configs", type=Path, default=Path("configs"))
    work.add_argument("--inventory", type=Path, default=Path("data/benchmark_inventory.json"))

    score = commands.add_parser("score", help="score generated shards of one model (metrics environment)")
    score.add_argument("--model", required=True)
    score.add_argument("--results-repo", required=True)
    score.add_argument("--work", type=Path, required=True)
    score.add_argument("--metricx-repo", type=Path, default=Path.home() / "src" / "metricx")
    score.add_argument("--comet-batch", type=int, default=64)
    score.add_argument("--metricx-batch", type=int, default=32)
    score.add_argument("--budget-min", type=float, default=20.0, help="wall-clock budget for this job")
    score.add_argument("--benchmark", default="", help="only this benchmark")
    score.add_argument("--shard-index", type=int, default=0, help="this job's residue class of shard indexes")
    score.add_argument("--shard-count", type=int, default=1, help="number of parallel jobs for this model")
    score.add_argument("--configs", type=Path, default=Path("configs"))
    score.add_argument("--inventory", type=Path, default=Path("data/benchmark_inventory.json"))
    return parser


def _project(args: argparse.Namespace) -> Project:
    return Project(args.configs, args.inventory)


def _check_configs_cmd(args: argparse.Namespace) -> str:
    return check_configs(args.root)


def _coverage_cmd(args: argparse.Namespace) -> str:
    return write_coverage(args.configs, args.inventory, args.out)


def _plan_cmd(args: argparse.Namespace) -> str:
    return plan_command(_project(args), args.model, args.benchmarks)


def _run_cmd(args: argparse.Namespace) -> str:
    return run_command(_project(args), args)


def _work_cmd(args: argparse.Namespace) -> str:
    return work_command(_project(args), args)


def _score_cmd(args: argparse.Namespace) -> str:
    return score_command(_project(args), args)


COMMANDS: dict[str, Callable[[argparse.Namespace], str]] = {
    "check-configs": _check_configs_cmd,
    "coverage": _coverage_cmd,
    "plan": _plan_cmd,
    "run": _run_cmd,
    "work": _work_cmd,
    "score": _score_cmd,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(COMMANDS[args.command](args))
    return 0
