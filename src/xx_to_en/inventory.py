"""Benchmark language inventories snapshotted from the Hugging Face Hub."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Inventory:
    """Corpus row counts per config and language codes available in each benchmark."""

    corpus_rows: dict[str, int]
    benchmark_codes: dict[str, frozenset[str]]

    def available(self, benchmark: str) -> frozenset[str]:
        return self.benchmark_codes.get(benchmark, frozenset())


def load_inventory(path: Path) -> Inventory:
    """Read ``data/benchmark_inventory.json`` as written by ``scripts/snapshot_benchmarks.py``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    codes = {name: frozenset(entry.get("languages", [])) for name, entry in data["benchmarks"].items()}
    return Inventory(corpus_rows=dict(data["corpus"]["config_rows"]), benchmark_codes=codes)
