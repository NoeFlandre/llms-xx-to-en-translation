"""Command-line entry point: ``xx2en check-configs`` and ``xx2en coverage``."""

from __future__ import annotations

import argparse
from pathlib import Path

from xx_to_en.config import load_benchmarks, load_language_map, load_models
from xx_to_en.coverage import build_coverage, render_markdown, summarize
from xx_to_en.inventory import load_inventory
from xx_to_en.languages import LanguageMap


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xx2en", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-configs", help="validate all YAML configs")
    check.add_argument("--root", type=Path, default=Path("configs"))

    coverage = commands.add_parser("coverage", help="write the language x benchmark coverage matrix")
    coverage.add_argument("--configs", type=Path, default=Path("configs"))
    coverage.add_argument("--inventory", type=Path, default=Path("data/benchmark_inventory.json"))
    coverage.add_argument("--out", type=Path, default=Path("docs/coverage_matrix.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check-configs":
        print(check_configs(args.root))
    else:
        print(write_coverage(args.configs, args.inventory, args.out))
    return 0
