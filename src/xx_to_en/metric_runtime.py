"""Model calls for COMET-22, MetricX-24 and GlotLID.

Runs in the metrics environment on Grid5000 nodes (``scripts/g5k/setup_metrics.sh``). Imports are
deferred so the package imports without those dependencies installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from xx_to_en.scoring import glotlid_code, parse_metricx_output

COMET_MODEL = "Unbabel/wmt22-comet-da"
METRICX_MODEL = "google/metricx-24-hybrid-xl-v2p6"
METRICX_TOKENIZER = "google/mt5-xl"
GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILE = "model.bin"


def comet_scores(inputs: Sequence[dict[str, str]], *, batch_size: int = 64, gpus: int = 1) -> list[float]:
    """COMET-22 reference-based scores, one per input and in order."""
    if not inputs:
        return []
    from comet import download_model, load_from_checkpoint  # ty: ignore[import-not-found]

    model = load_from_checkpoint(download_model(COMET_MODEL))
    output = model.predict(list(inputs), batch_size=batch_size, gpus=gpus, progress_bar=False)
    return [float(score) for score in output.scores]


def metricx_scores(
    inputs: Sequence[dict[str, str]],
    *,
    repo: Path,
    work: Path,
    batch_size: int = 8,
    max_input_length: int = 1536,
) -> list[float]:
    """MetricX-24 hybrid XL, reference-based, run through the upstream ``metricx24.predict`` module."""
    if not inputs:
        return []
    work.mkdir(parents=True, exist_ok=True)
    input_file = work / "metricx_input.jsonl"
    output_file = work / "metricx_output.jsonl"
    input_file.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in inputs), encoding="utf-8"
    )
    command = [
        sys.executable,
        "-m",
        "metricx24.predict",
        "--tokenizer",
        METRICX_TOKENIZER,
        "--model_name_or_path",
        METRICX_MODEL,
        "--max_input_length",
        str(max_input_length),
        "--batch_size",
        str(batch_size),
        "--input_file",
        str(input_file),
        "--output_file",
        str(output_file),
    ]
    subprocess.run(command, cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)}, check=True)
    return parse_metricx_output(output_file.read_text(encoding="utf-8").splitlines())


def glotlid_codes(texts: Sequence[str]) -> list[str]:
    """Detected language per text as corpus-style codes (``eng_Latn``). Empty texts give ``""``."""
    import fasttext  # ty: ignore[import-not-found]
    from huggingface_hub import hf_hub_download  # ty: ignore[import-not-found]

    model = fasttext.load_model(hf_hub_download(repo_id=GLOTLID_REPO, filename=GLOTLID_FILE))
    codes = []
    for text in texts:
        flat = text.replace("\n", " ").strip()
        if not flat:
            codes.append("")
            continue
        labels, _ = model.predict(flat)
        codes.append(glotlid_code(labels[0]))
    return codes
