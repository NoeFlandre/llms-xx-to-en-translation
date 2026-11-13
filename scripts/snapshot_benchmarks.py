"""Snapshot benchmark language inventories from the Hugging Face Hub.

Metadata only: repository listings, dataset-viewer split and size lists. No benchmark
or corpus rows are downloaded. The output is committed so the coverage matrix can be
rebuilt without network access.

Usage:
    python3 scripts/snapshot_benchmarks.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmark_inventory.json"
TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"
CORPUS_REPO = "NoeFlandre/GeoReSeT-Corpus-Fused"
FLORES_REPO = "openlanguagedata/flores_plus"
NTREX_REPO = "davidstap/NTREX"
WMT22_REPO = "haoranxu/WMT22-Test"
WMT23_REPO = "haoranxu/WMT23-Test"
TATOEBA_REPO = "Helsinki-NLP/tatoeba_mt"


def _token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    return None


def _get_json(url: str) -> Any:
    request = urllib.request.Request(url)
    token = _token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS hosts
        return json.load(response)


def repo_listing(repo: str, folder: str, suffix: str) -> list[str]:
    """Return file stems under ``folder`` in a dataset repository."""
    entries = _get_json(f"https://huggingface.co/api/datasets/{repo}/tree/main/{folder}")
    names = [entry["path"].rsplit("/", 1)[-1] for entry in entries]
    return sorted(name[: -len(suffix)] for name in names if name.endswith(suffix))


def dataset_configs(repo: str) -> list[str]:
    """Return config names exposed by the dataset viewer."""
    payload = _get_json(f"https://datasets-server.huggingface.co/splits?dataset={repo}")
    return sorted({split["config"] for split in payload["splits"]})


def config_row_counts(repo: str) -> dict[str, int]:
    """Return row counts per config from the dataset-viewer size endpoint."""
    payload = _get_json(f"https://datasets-server.huggingface.co/size?dataset={repo}")
    return {item["config"]: int(item["num_rows"]) for item in payload["size"]["configs"]}


def wmt_source_codes(repo: str) -> list[str]:
    """Source codes of X->EN configs (``xx-en``)."""
    return sorted(config.split("-")[0] for config in dataset_configs(repo) if config.endswith("-en"))


def tatoeba_non_english_codes(repo: str = TATOEBA_REPO) -> list[str]:
    """Non-English codes with a test file against English, in either direction."""
    prefix = "tatoeba-test."
    codes: set[str] = set()
    for stem in repo_listing(repo, "data/test", ".tsv"):
        if not stem.startswith(prefix):
            continue
        source, target = stem[len(prefix) :].split("-", 1)
        if source == "eng" and target != "eng":
            codes.add(target)
        if target == "eng" and source != "eng":
            codes.add(source)
    return sorted(codes)


def _try(fetch: Callable[[], list[str]]) -> dict[str, Any]:
    try:
        return {"status": "ok", "languages": fetch()}
    except urllib.error.HTTPError as error:
        return {"status": f"http {error.code}", "languages": []}


def build_inventory() -> dict[str, Any]:
    return {
        "snapshot_date": date.today().isoformat(),
        "corpus": {"repo": CORPUS_REPO, "config_rows": config_row_counts(CORPUS_REPO)},
        "benchmarks": {
            "flores_plus_devtest": {
                "repo": FLORES_REPO,
                **_try(lambda: repo_listing(FLORES_REPO, "devtest", ".jsonl")),
            },
            "ntrex_128": {"repo": NTREX_REPO, **_try(lambda: dataset_configs(NTREX_REPO))},
            "wmt22_xen": {"repo": WMT22_REPO, **_try(lambda: wmt_source_codes(WMT22_REPO))},
            "wmt23_xen": {"repo": WMT23_REPO, **_try(lambda: wmt_source_codes(WMT23_REPO))},
            "tatoeba_xen": {"repo": TATOEBA_REPO, **_try(lambda: tatoeba_non_english_codes())},
        },
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build_inventory(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
