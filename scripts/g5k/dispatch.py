#!/usr/bin/env python3
"""Keep tiny one-hour jobs running on free Grid'5000 GPUs until every shard is in the results dataset.

Runs on the development machine. On the frontends it only runs read-only commands (oarnodes, oarstat,
usagepolicycheck) plus oarsub and oardel for the tiny jobs.

    uv run python scripts/g5k/dispatch.py --dry-run   # one cycle, print placements only
    uv run python scripts/g5k/dispatch.py             # loop every 5 minutes until nothing is pending
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from free_gpus import count_free

from xx_to_en.cli import Project
from xx_to_en.dispatch import GpuPool, choose_placements, crosses_boundary
from xx_to_en.plan import build_shards
from xx_to_en.results_store import score_path, shard_path

ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = Path.home() / ".cache" / "xx2en-dispatch" / "state.json"
REPO = "NoeFlandre/llms-xx-to-en-translation"
SITE_USER = "nflandre"
STALE_WAIT_S = 300
POLL_S = 300
MAX_ATTEMPTS_PER_MODEL = 12
MAX_FAILURES_PER_CLUSTER = 2
REMOTE_JOB = "bash $HOME/src/llms-xx-to-en-translation/scripts/g5k/work_job.sh"
OAR_ID = re.compile(r"OAR_JOB_ID=(\d+)")
STATE_LINE = re.compile(r"state = (\w+)")


def ssh(site: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", site, command],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return "\n".join(line for line in result.stdout.splitlines() if "bashrc" not in line)


def load_pools(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["sites"]


def probe_free(site: str) -> Counter[str]:
    return count_free(ssh(site, "oarnodes 2>/dev/null"))


def job_state(site: str, job_id: int) -> str:
    match = STATE_LINE.search(ssh(site, f"oarstat -j {job_id} -f 2>/dev/null"))
    return match.group(1) if match else "Gone"


def cancel(site: str, job_id: int) -> None:
    ssh(site, f"oardel {job_id} >/dev/null 2>&1")


def submit(site: str, cluster: str, model: str) -> int | None:
    command = (
        f"oarsub -n xx2en-{model} -p \"cluster='{cluster}'\" -l gpu=1,walltime=1:00 "
        f'-O "$HOME/jobs/{model}.%jobid%.out" -E "$HOME/jobs/{model}.%jobid%.err" '
        f'"MODEL={model} RESULTS_REPO={REPO} BUDGET_MIN=30 {REMOTE_JOB}"'
    )
    match = OAR_ID.search(ssh(site, command))
    return int(match.group(1)) if match else None


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"jobs": [], "attempts": {}, "failures": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def refresh_jobs(state: dict[str, Any], now: float) -> None:
    """Drop finished jobs; cancel jobs still waiting after STALE_WAIT_S (they would lose the exemption)."""
    kept = []
    for job in state["jobs"]:
        current = job_state(job["site"], job["id"])
        if current in ("Terminated", "Error", "Gone"):
            continue
        if current == "Waiting" and now - job["submitted"] > STALE_WAIT_S:
            cancel(job["site"], job["id"])
            continue
        kept.append({**job, "state": current})
    state["jobs"] = kept


def pending_models(project: Project, done: set[str]) -> list[tuple[str, int]]:
    """Models with any shard or score still missing, smallest first, with their GPU memory need."""
    out = []
    for name, model in project.models.items():
        if model.official_languages == "pending":
            continue
        shards = build_shards(name, project.plan(model))
        if any(shard_path(s) not in done or score_path(s) not in done for s in shards):
            out.append((model.params_b, name, model.gpu_memory_gb_min))
    return [(name, need) for _, name, need in sorted(out)]


def usage_lines(sites: list[str]) -> list[str]:
    return [f"{site}: {ssh(site, 'usagepolicycheck 2>&1 | tail -1')}" for site in sites]


def cycle(project: Project, pools_config: list[dict[str, Any]], state: dict[str, Any], dry_run: bool) -> None:
    from huggingface_hub import HfApi

    now = datetime.now()
    if crosses_boundary(now):
        print(f"{now:%H:%M}: a one-hour job now would cross 09:00 or 19:00; waiting", flush=True)
        return
    done = set(HfApi().list_repo_files(repo_id=REPO, repo_type="dataset"))
    refresh_jobs(state, time.time())
    pending = [
        (name, need)
        for name, need in pending_models(project, done)
        if state["attempts"].get(name, 0) < MAX_ATTEMPTS_PER_MODEL
    ]
    pools = []
    for entry in pools_config:
        key = f"{entry['cluster']}|{entry['gpu_model']}"
        if state["failures"].get(f"{entry['site']}/{entry['cluster']}", 0) >= MAX_FAILURES_PER_CLUSTER:
            continue
        free = probe_free(entry["site"])[key]
        pools.append(GpuPool(entry["site"], entry["cluster"], entry["memory_gb"], free))
    running_per_site = Counter(job["site"] for job in state["jobs"])
    placements = choose_placements(
        pending,
        pools,
        running_models=frozenset(job["model"] for job in state["jobs"]),
        running_per_site=running_per_site,
    )
    print(
        f"{now:%H:%M}: pending {len(pending)} models; running {len(state['jobs'])}; placements {len(placements)}",
        flush=True,
    )
    for placement in placements:
        print(f"  -> {placement.model} on {placement.site}/{placement.cluster}", flush=True)
        if dry_run:
            continue
        job_id = submit(placement.site, placement.cluster, placement.model)
        if job_id is None:
            key = f"{placement.site}/{placement.cluster}"
            state["failures"][key] = state["failures"].get(key, 0) + 1
            print(f"     submission refused on {key}", flush=True)
            continue
        state["attempts"][placement.model] = state["attempts"].get(placement.model, 0) + 1
        state["jobs"].append(
            {
                "id": job_id,
                "site": placement.site,
                "cluster": placement.cluster,
                "model": placement.model,
                "submitted": time.time(),
                "state": "Waiting",
            }
        )
    save_state(state)
    for line in usage_lines(sorted({p["site"] for p in pools_config})):
        print(f"  policy {line}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sites", type=Path, default=ROOT / "configs" / "sites.yaml")
    args = parser.parse_args(argv)
    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    pools_config = load_pools(args.sites)
    state = load_state()
    while True:
        cycle(project, pools_config, state, args.dry_run)
        if args.dry_run:
            return 0
        if not pending_models(project, set(_done_files())):
            print("all shards and scores are in the dataset", flush=True)
            return 0
        time.sleep(POLL_S)


def _done_files() -> list[str]:
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(repo_id=REPO, repo_type="dataset")


if __name__ == "__main__":
    sys.exit(main())
