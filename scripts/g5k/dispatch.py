#!/usr/bin/env python3
"""Keep tiny one-hour jobs running on free Grid'5000 GPUs until every shard and score is in the dataset.

Runs on the development machine. On the frontends it only runs read-only commands (oarnodes, oarstat,
usagepolicycheck) plus oarsub and oardel for the tiny jobs. Only default-queue clusters are used.

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

from xx_to_en.cli import Project
from xx_to_en.dispatch import GpuPool, Pending, choose_placements, crosses_boundary
from xx_to_en.gpu_probe import count_free
from xx_to_en.plan import build_shards
from xx_to_en.results_store import repo_files, score_path, shard_path

ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = Path.home() / ".cache" / "xx2en-dispatch" / "state.json"
REPO = "NoeFlandre/llms-xx-to-en-translation"
STALE_WAIT_S = 540  # just under 10 minutes: a job that starts later loses the exemption
POLL_S = 300
MAX_NO_PROGRESS = 3  # consecutive jobs that finished without uploading anything
MAX_FAILURES_PER_CLUSTER = 2
REMOTE_JOB = "bash $HOME/src/llms-xx-to-en-translation/scripts/g5k/work_job.sh"
OAR_ID = re.compile(r"OAR_JOB_ID=(\d+)")
STATE_LINE = re.compile(r"state = (\w+)")
PendingInfo = tuple[int, int, float, bool]  # (GPU memory need GB, missing files, size in billions, fp16_ok)


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


def submit(site: str, cluster: str, model: str, dtype: str) -> int | None:
    command = (
        f"oarsub -n xx2en-{model} -p \"cluster='{cluster}'\" -l gpu=1,walltime=1:00 "
        f'-O "$HOME/jobs/{model}.%jobid%.out" -E "$HOME/jobs/{model}.%jobid%.err" '
        f'"MODEL={model} RESULTS_REPO={REPO} BUDGET_MIN=30 DTYPE={dtype} {REMOTE_JOB}"'
    )
    match = OAR_ID.search(ssh(site, command))
    return int(match.group(1)) if match else None


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"jobs": [], "attempts": {}, "failures": {}, "no_progress": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def refresh_jobs(state: dict[str, Any], now: float, remaining: dict[str, int]) -> None:
    """Drop finished jobs, count jobs that ended without progress, cancel jobs still waiting too long."""
    kept = []
    for job in state["jobs"]:
        current = job_state(job["site"], job["id"])
        if current in ("Terminated", "Error", "Gone"):
            if remaining.get(job["model"], 0) >= job["remaining_at_submit"]:
                no_progress = state.setdefault("no_progress", {})
                no_progress[job["model"]] = no_progress.get(job["model"], 0) + 1
            continue
        if current == "Waiting" and now - job["submitted"] > STALE_WAIT_S:
            cancel(job["site"], job["id"])
            continue
        kept.append({**job, "state": current})
    state["jobs"] = kept


def pending_models(project: Project, done: set[str]) -> dict[str, PendingInfo]:
    """Incomplete models with what a placement needs to know about them."""
    out: dict[str, PendingInfo] = {}
    for name, model in project.models.items():
        if model.official_languages == "pending":
            continue
        shards = build_shards(name, project.plan(model))
        missing = sum((shard_path(s) not in done) + (score_path(s) not in done) for s in shards)
        if missing:
            out[name] = (model.gpu_memory_gb_min, missing, model.params_b, model.fp16_ok)
    return out


def site_ready(site: str) -> bool:
    """A site takes jobs only once its generation environment exists."""
    return "ready" in ssh(site, "test -x $HOME/venvs/xx2en/bin/xx2en && echo ready")


def usage_lines(sites: list[str]) -> list[str]:
    return [f"{site}: {ssh(site, 'usagepolicycheck 2>&1 | tail -1')}" for site in sites]


def build_pools(pools_config: list[dict[str, Any]], state: dict[str, Any]) -> list[GpuPool]:
    """Free GPUs per usable cluster, on sites that already have a generation environment."""
    ready = {site for site in {entry["site"] for entry in pools_config} if site_ready(site)}
    counts = {site: probe_free(site) for site in ready}
    pools = []
    for entry in pools_config:
        cluster_key = f"{entry['site']}/{entry['cluster']}"
        if entry["site"] not in ready or state["failures"].get(cluster_key, 0) >= MAX_FAILURES_PER_CLUSTER:
            continue
        free = counts[entry["site"]][f"{entry['cluster']}|{entry['gpu_model']}"]
        pools.append(
            GpuPool(entry["site"], entry["cluster"], entry["memory_gb"], free, bool(entry.get("bf16", True)))
        )
    return pools


def launch(
    placements: list[Any], state: dict[str, Any], dry_run: bool, pending: dict[str, PendingInfo]
) -> None:
    for placement in placements:
        print(
            f"  -> {placement.model} on {placement.site}/{placement.cluster} ({placement.dtype})",
            flush=True,
        )
        if dry_run:
            continue
        job_id = submit(placement.site, placement.cluster, placement.model, placement.dtype)
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
                "remaining_at_submit": pending[placement.model][1],
            }
        )


def cycle(project: Project, pools_config: list[dict[str, Any]], state: dict[str, Any], dry_run: bool) -> None:
    now = datetime.now()
    if crosses_boundary(now):
        print(f"{now:%H:%M}: a one-hour job now would cross 09:00 or 19:00; waiting", flush=True)
        return
    pending = pending_models(project, repo_files(REPO))
    refresh_jobs(state, time.time(), {name: info[1] for name, info in pending.items()})
    no_progress = state.get("no_progress", {})
    ready = sorted(
        (info[2], name, info[0], info[3])
        for name, info in pending.items()
        if no_progress.get(name, 0) < MAX_NO_PROGRESS
    )
    placements = choose_placements(
        [Pending(name, need, fp16) for _, name, need, fp16 in ready],
        build_pools(pools_config, state),
        running_models=frozenset(job["model"] for job in state["jobs"]),
        running_per_site=Counter(job["site"] for job in state["jobs"]),
    )
    summary = f"pending {len(pending)} models; running {len(state['jobs'])}; placements {len(placements)}"
    print(f"{now:%H:%M}: {summary}", flush=True)
    launch(placements, state, dry_run, pending)
    save_state(state)
    for line in usage_lines(sorted({entry["site"] for entry in pools_config})):
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
        if not pending_models(project, repo_files(REPO)):
            print("all shards and scores are in the dataset", flush=True)
            return 0
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
