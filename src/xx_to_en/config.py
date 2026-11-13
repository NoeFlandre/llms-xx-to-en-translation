"""Load and validate the YAML configs that drive the benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """One candidate model; mirrors ``configs/models/*.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    hf_repo: str
    family: str
    params_b: float = Field(gt=0)
    active_params_b: float = Field(gt=0)
    license: str
    access: Literal["open", "accepted", "not_accepted"]
    backend: Literal["vllm", "transformers"]
    prompt: str
    thinking: Literal["off", "not_applicable"]
    dtype: Literal["bfloat16", "float16", "float32"]
    tensor_parallel: int = Field(ge=1)
    gpu_memory_gb_min: int = Field(gt=0)
    trust_remote_code: bool
    official_languages: list[str] | Literal["pending"]


class BenchmarkConfig(BaseModel):
    """One benchmark; mirrors ``configs/benchmarks/*.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    hf_repo: str
    direction: Literal["X->EN"]
    origin: Literal["native", "english_translated", "mixed"]
    license: str
    access: Literal["open", "restricted"]
    language_map: str = "languages.yaml"
    split: str | None = None
    layout: str | None = None
    text_field: str | None = None
    reference_lang: str | None = None
    config_filter: str | None = None
    notes: str | None = None


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file whose top level must be a mapping."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def _load_directory(directory: Path, model: type[ModelConfig] | type[BenchmarkConfig]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for path in sorted(directory.glob("*.yaml")):
        config = model.model_validate(read_yaml(path))
        if config.name != path.stem:
            raise ValueError(f"{path}: name {config.name!r} must match the file name")
        loaded[config.name] = config
    return loaded


def load_models(directory: Path) -> dict[str, ModelConfig]:
    """Validate every model YAML in ``directory``, keyed by name."""
    return _load_directory(directory, ModelConfig)


def load_benchmarks(directory: Path) -> dict[str, BenchmarkConfig]:
    """Validate every benchmark YAML in ``directory``, keyed by name."""
    return _load_directory(directory, BenchmarkConfig)


def load_language_map(path: Path) -> dict[str, dict[str, str]]:
    """Load ``languages.yaml``: corpus code -> {benchmark name: benchmark code}."""
    data = read_yaml(path)
    mapping: dict[str, dict[str, str]] = {}
    for corpus_code, entry in data.items():
        if not isinstance(entry, dict) or not all(isinstance(value, str) for value in entry.values()):
            raise ValueError(f"{path}: {corpus_code} must map benchmark names to strings")
        mapping[corpus_code] = dict(entry)
    return mapping
