"""Smoke test for the metrics environment: score two toy segments with COMET-22, MetricX-24 and GlotLID."""

from __future__ import annotations

import json
import os
from pathlib import Path

from xx_to_en.metric_runtime import Scorers

ROWS = [
    {"source": "Bonjour le monde.", "hypothesis": "Hello world.", "reference": "Hello, world."},
    {"source": "Le chat dort.", "hypothesis": "The cat sleeps.", "reference": "The cat is sleeping."},
]


def main() -> None:
    scorers = Scorers(
        metricx_repo=Path(os.environ["METRICX_REPO"]),
        work=Path(os.environ["WORK"]) / "metricx",
    )
    comet = scorers.comet([{"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]} for r in ROWS])
    metricx = scorers.metricx(ROWS)
    codes = scorers.glotlid([r["hypothesis"] for r in ROWS])
    print(json.dumps({"comet": comet, "metricx": metricx, "glotlid": codes}), flush=True)


if __name__ == "__main__":
    main()
