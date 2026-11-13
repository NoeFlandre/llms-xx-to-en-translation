"""Layout and transfer of shard outputs in the Hugging Face results dataset."""

from __future__ import annotations

import shutil
from pathlib import Path

from xx_to_en.plan import Shard

HYPOTHESES_PREFIX = "results"
SCORES_PREFIX = "scores"


def shard_path(shard: Shard, *, prefix: str = HYPOTHESES_PREFIX) -> str:
    """Repository path of a shard's hypotheses, e.g. ``results/<model>/<benchmark>/<shard>.jsonl``."""
    return f"{prefix}/{shard.model}/{shard.benchmark}/{shard.shard_id}.parquet"


def score_path(shard: Shard) -> str:
    """Repository path of a shard's scores."""
    return shard_path(shard, prefix=SCORES_PREFIX)


def repo_files(repo_id: str) -> set[str]:
    """Every file path in the dataset repository (one API call)."""
    from huggingface_hub import HfApi

    return set(HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset"))


def download(repo_id: str, path: str, local: Path) -> None:
    """Copy one file from the dataset repository to ``local``."""
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset")
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, local)


def upload(repo_id: str, local: Path, path: str) -> None:
    """Commit one file to the dataset repository."""
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(local),
        path_in_repo=path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"add {path}",
    )
