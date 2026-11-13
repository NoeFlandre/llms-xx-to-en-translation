"""CRAP score per function: CRAP = comp^2 * (1 - coverage)^3 + comp.

Inputs:
- complexity: radon cyclomatic complexity of each function in ``src/``
- coverage: per-line coverage from ``coverage.json`` (``pytest --cov --cov-report=json:coverage.json``)

Exits non-zero if any function reaches the threshold (default 6).

Usage:
    uv run pytest --cov --cov-report=json:coverage.json
    uv run python scripts/crap.py --threshold 6
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from radon.complexity import cc_visit

ROOT = Path(__file__).resolve().parents[1]


def function_coverage(executed: set[int], missing: set[int], start: int, end: int) -> float:
    """Fraction of executable lines in [start, end] that were executed."""
    lines_in_range = {line for line in executed | missing if start <= line <= end}
    if not lines_in_range:
        return 1.0
    covered = {line for line in executed if start <= line <= end}
    return len(covered) / len(lines_in_range)


def crap_score(complexity: int, coverage: float) -> float:
    return complexity**2 * (1 - coverage) ** 3 + complexity


def _file_coverage(coverage_data: dict, path: Path) -> tuple[set[int], set[int]] | None:
    key = str(path.relative_to(ROOT))
    entry = coverage_data["files"].get(key)
    if entry is None:
        return None
    return set(entry["executed_lines"]), set(entry["missing_lines"])


def scan(source_root: Path, coverage_data: dict) -> list[tuple[float, str, int, int, float]]:
    """Return (crap, file:function, complexity, lineno, coverage) for every function."""
    results = []
    for path in sorted(source_root.rglob("*.py")):
        lines = _file_coverage(coverage_data, path)
        executed, missing = lines if lines else (set(), set())
        for block in _flatten(cc_visit(path.read_text(encoding="utf-8"))):
            end = _function_end(path, block.lineno, block.name)
            coverage = function_coverage(executed, missing, block.lineno, end)
            score = crap_score(block.complexity, coverage)
            label = f"{path.relative_to(ROOT)}:{block.name}"
            results.append((score, label, block.complexity, block.lineno, coverage))
    return sorted(results, reverse=True)


def _flatten(blocks: list) -> list:
    """Top-level functions plus methods of classes (radon reports classes separately)."""
    flat = []
    for block in blocks:
        flat.extend(getattr(block, "methods", []) or [])
        if not hasattr(block, "methods"):
            flat.append(block)
    return flat


DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _function_end(path: Path, lineno: int, name: str) -> int:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, DEFINITIONS) and node.lineno == lineno and node.name == name:
            return node.end_lineno or lineno
    return lineno


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--coverage", type=Path, default=ROOT / "coverage.json")
    parser.add_argument("--source", type=Path, default=ROOT / "src" / "xx_to_en")
    parser.add_argument("--threshold", type=float, default=6.0)
    args = parser.parse_args(argv)

    coverage_data = json.loads(args.coverage.read_text(encoding="utf-8"))
    results = scan(args.source, coverage_data)
    failing = [row for row in results if row[0] >= args.threshold]
    print(f"functions: {len(results)}, max CRAP: {results[0][0]:.2f} ({results[0][1]})")
    for score, name, complexity, lineno, coverage in failing:
        print(f"  FAIL crap={score:.2f} cc={complexity} cov={coverage:.0%} {name}:{lineno}")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
