"""Count free GPU resources per cluster and GPU model from ``oarnodes`` output on stdin.

A GPU resource is free when it is Alive, has no running job, and is not drained or in maintenance.
Runs locally; the frontend only executes the read-only ``oarnodes`` command.

Usage:
    ssh -o BatchMode=yes nancy oarnodes | python3 scripts/g5k/free_gpus.py --site nancy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

PROPERTY = re.compile(r"(\w+)=([^,]*)")


def records(text: str) -> Iterator[list[str]]:
    """Blank-line separated blocks, one per resource."""
    block: list[str] = []
    for line in text.splitlines():
        if line.strip():
            block.append(line)
        elif block:
            yield block
            block = []
    if block:
        yield block


def parse_record(lines: Sequence[str]) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition(": ")
        if separator:
            fields[key.strip()] = value.strip()
    props = dict(PROPERTY.findall(fields.get("properties", "")))
    return {
        "state": fields.get("state", ""),
        "busy": "jobs" in fields,
        "cluster": props.get("cluster", ""),
        "gpu_model": props.get("gpu_model", ""),
        "drain": props.get("drain", ""),
        "maintenance": props.get("maintenance", ""),
    }


def is_free_gpu(record: dict[str, Any]) -> bool:
    return (
        record["state"] == "Alive"
        and not record["busy"]
        and bool(record["gpu_model"])
        and record["drain"] == "NO"
        and record["maintenance"] == "NO"
    )


def count_free(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for block in records(text):
        record = parse_record(block)
        if is_free_gpu(record):
            counts[f"{record['cluster']}|{record['gpu_model']}"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--site", required=True)
    args = parser.parse_args(argv)
    counts = count_free(sys.stdin.read())
    print(json.dumps({"site": args.site, "free": dict(sorted(counts.items()))}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
