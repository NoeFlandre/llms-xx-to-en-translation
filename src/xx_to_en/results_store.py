"""Layout and transfer of shard outputs in the Hugging Face results dataset."""

from __future__ import annotations

from pathlib import Path

from xx_to_en.plan import Shard

HYPOTHESES_PREFIX = "results"
SCORES_PREFIX = "scores"


def shard_path(shard: Shard, *, prefix: str = HYPOTHESES_PREFIX) -> str:
    """Repository path of a shard's hypotheses, e.g. ``results/<model>/<benchmark>/<shard>.jsonl``."""
    return f"{prefix}/{shard.model}/{shard.benchmark}/{shard.shard_id}.jsonl"


def score_path(shard: Shard) -> str:
    """Repository path of a shard's scores."""
    return shard_path(shard, prefix=SCORES_PREFIX)


def is_uploaded(repo_id: str, path: str) -> bool:
    """True when the file already exists in the dataset repository."""
    from huggingface_hub import HfApi  # ty: ignore[import-not-found]

    return HfApi().file_exists(repo_id=repo_id, filename=path, repo_type="dataset")


def upload(repo_id: str, local: Path, path: str) -> None:
    """Commit one file to the dataset repository."""
    from huggingface_hub import HfApi  # ty: ignore[import-not-found]

    HfApi().upload_file(
        path_or_fileobj=str(local),
        path_in_repo=path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"add {path}",
    )
