"""Parquet tables for shard outputs: hypotheses and scores, one file per shard."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write_table(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write records as one Parquet file (zstd), creating parent directories."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [dict(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def read_table(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet file back into records."""
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()
