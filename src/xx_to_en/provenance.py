"""Provenance: what was run, with which settings and software. Stored per row and per job."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from xx_to_en.config import ModelConfig
from xx_to_en.prompts import PROMPT_VERSION
from xx_to_en.runner import Decoding

SOFTWARE_PACKAGES = ("vllm", "transformers", "torch")


def created_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def software_versions() -> str:
    found = []
    for name in SOFTWARE_PACKAGES:
        try:
            found.append(f"{name}={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return "; ".join(found)


def run_fields(
    model: ModelConfig,
    *,
    revision: str,
    decoding: Decoding,
    dtype: str,
    backend: str,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Columns repeated on every row of a shard: the settings that produced it."""
    return {
        "model_repo": model.hf_repo,
        "model_revision": revision,
        "prompt_template": model.prompt,
        "prompt_version": PROMPT_VERSION,
        "decoding": "greedy",
        "seed": decoding.seed,
        "budget_ratio": decoding.ratio,
        "budget_floor": decoding.floor,
        "budget_ceiling": decoding.ceiling,
        "thinking": model.thinking,
        "dtype": dtype,
        "backend": backend,
        "software": software_versions(),
        "gpu": env.get("GPU_NAME", ""),
        "node": env.get("NODE_NAME", ""),
        "job_id": env.get("OAR_JOB_ID", ""),
        "code_commit": env.get("CODE_COMMIT", ""),
    }


def manifest(model: ModelConfig, run: Mapping[str, Any], shards: list[str]) -> dict[str, Any]:
    """Per-job record: the full model config, the run settings, and the shards this job uploaded."""
    return {"created_at": created_now(), "model": model.model_dump(), "run": dict(run), "shards": shards}
