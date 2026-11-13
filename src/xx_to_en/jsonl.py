"""JSON-lines helpers for record files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per non-blank line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_records(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write one JSON object per line, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
