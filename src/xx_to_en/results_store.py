"""Layout and transfer of shard outputs in the Hugging Face results dataset."""

from __future__ import annotations

from pathlib import Path

from xx_to_en.plan import Shard

RESULTS_PREFIX = "results"


def shard_path(shard: Shard, *, prefix: str = RESULTS_PREFIX) -> str:
    """Repository path of a shard, e.g. ``results/tiny/flores_plus_devtest/flores_plus_devtest-000.jsonl``."""
    return f"{prefix}/{shard.model}/{shard.benchmark}/{shard.shard_id}.jsonl"


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
