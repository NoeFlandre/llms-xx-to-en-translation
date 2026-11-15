"""Keep tiny one-hour jobs running on free Grid'5000 GPUs until every shard and score is in the dataset.

Work is split per (model, benchmark), so one model can run several jobs at once. Runs on the development
machine. On the frontends it only runs read-only commands (oarnodes, oarstat, usagepolicycheck) plus
oarsub and oardel for the tiny jobs. Only default-queue clusters are used.

    uv run xx2en dispatch --dry-run   # one cycle, print placements only
    uv run xx2en dispatch             # loop every 5 minutes until nothing is pending
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from xx_to_en.cli import Project
from xx_to_en.dispatch import GpuPool, Pending, choose_placements, crosses_boundary, jobs_for_pair
from xx_to_en.gpu_probe import count_free
from xx_to_en.results_store import repo_files, score_path, shard_path

ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = Path.home() / ".cache" / "xx2en-dispatch" / "state.json"
REPO = "NoeFlandre/llms-xx-to-en-translation"
STALE_WAIT_S = 540
ENDED_STATES = ("Terminated", "Error", "Gone")
POLL_S = 300
MAX_NO_PROGRESS = 3
MAX_FAILURES_PER_CLUSTER = 2
REMOTE_JOB = "bash $HOME/src/llms-xx-to-en-translation/scripts/g5k/work_job.sh"
OAR_ID = re.compile(r"OAR_JOB_ID=(\d+)")
STATE_LINE = re.compile(r"state = (\w+)")
PendingInfo = tuple[int, int, float, bool]


def ssh(site: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", site, command],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
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


def submit(
    site: str,
    cluster: str,
    model: str,
    benchmark: str,
    dtype: str,
    queue: str,
    shard_index: int = 0,
    shard_count: int = 1,
) -> int | None:
    env = (
        f"MODEL={model} BENCHMARK={benchmark} RESULTS_REPO={REPO} BUDGET_MIN=30 DTYPE={dtype} "
        f"SHARD_INDEX={shard_index} SHARD_COUNT={shard_count}"
    )
    queue_flag = " -q besteffort" if queue == "besteffort" else ""
    command = (
        f"oarsub -n xx2en-{model}-{benchmark[:6]} -p \"cluster='{cluster}'\"{queue_flag} "
        "-l gpu=1,walltime=1:00 "
        f'-O "$HOME/jobs/{model}.{benchmark}.%jobid%.out" -E "$HOME/jobs/{model}.{benchmark}.%jobid%.err" '
        f'"{env} {REMOTE_JOB}"'
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
        if current in ENDED_STATES:
            _count_if_no_progress(state, job, remaining)
        elif not _cancel_if_stale(job, current, now):
            kept.append({**job, "state": current})
    state["jobs"] = kept


def _count_if_no_progress(state: dict[str, Any], job: dict[str, Any], remaining: dict[str, int]) -> None:
    if remaining.get(job["model"], 0) >= job["remaining_at_submit"]:
        no_progress = state.setdefault("no_progress", {})
        no_progress[job["model"]] = no_progress.get(job["model"], 0) + 1


def _cancel_if_stale(job: dict[str, Any], current: str, now: float) -> bool:
    if current == "Waiting" and now - job["submitted"] > STALE_WAIT_S:
        cancel(job["site"], job["id"])
        return True
    return False


def pending_work(project: Project, done: set[str]) -> dict[str, PendingInfo]:
    """Incomplete (model, benchmark) work, with what a placement needs to know about it."""
    grouped = _missing_counts(project, done)
    out: dict[str, PendingInfo] = {}
    for (name, benchmark), (generation, scoring) in grouped.items():
        if generation + scoring == 0:
            continue
        model = project.models[name]
        out[f"{name}|{benchmark}"] = (
            model.gpu_memory_gb_min,
            generation + scoring,
            model.params_b,
            model.fp16_ok and generation > 0,
        )
    return out


def _missing_counts(project: Project, done: set[str]) -> dict[tuple[str, str], list[int]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for name, model in project.models.items():
        if not model.enabled:
            continue
        for shard in project.shards(model, done):
            counts = grouped.setdefault((name, shard.benchmark), [0, 0])
            counts[0] += shard_path(shard) not in done
            counts[1] += score_path(shard) not in done
    return grouped


def waiting_orphans(site: str, tracked: set[tuple[str, int]]) -> list[int]:
    """Waiting xx2en jobs on a site that the dispatcher does not track; they would never be cancelled."""
    out = ssh(site, "oarstat -u nflandre 2>/dev/null | grep xx2en | awk '{ if ($(NF-1)==\"W\") print $1 }'")
    return [int(x) for x in out.split() if x.isdigit() and (site, int(x)) not in tracked]


def site_ready(site: str) -> bool:
    """A site takes jobs only once its generation environment exists."""
    return "ready" in ssh(site, "test -x $HOME/venvs/xx2en/bin/xx2en && echo ready")


def usage_lines(sites: list[str]) -> list[str]:
    return [f"{site}: {ssh(site, 'usagepolicycheck 2>&1 | tail -1')}" for site in sites]


def build_pools(pools_config: list[dict[str, Any]], state: dict[str, Any]) -> list[GpuPool]:
    """Free GPUs per usable cluster, on sites that already have a generation environment."""
    ready, counts = _probe_sites(pools_config)
    return [_pool(entry, counts, state) for entry in pools_config if _usable(entry, ready, state)]


def _probe_sites(pools_config: list[dict[str, Any]]) -> tuple[set[str], dict[str, Counter[str]]]:
    sites = {entry["site"] for entry in pools_config}
    ready = {site for site in sites if site_ready(site)}
    return ready, {site: probe_free(site) for site in ready}


def _usable(entry: dict[str, Any], ready: set[str], state: dict[str, Any]) -> bool:
    cluster_key = f"{entry['site']}/{entry['cluster']}"
    return entry["site"] in ready and state["failures"].get(cluster_key, 0) < MAX_FAILURES_PER_CLUSTER


def _pool(entry: dict[str, Any], counts: dict[str, Counter[str]], state: dict[str, Any]) -> GpuPool:
    waiting = sum(
        job["site"] == entry["site"] and job["cluster"] == entry["cluster"] and job["state"] == "Waiting"
        for job in state["jobs"]
    )
    free = max(0, counts[entry["site"]][f"{entry['cluster']}|{entry['gpu_model']}"] - waiting)
    return GpuPool(
        entry["site"],
        entry["cluster"],
        entry["memory_gb"],
        free,
        bool(entry.get("bf16", True)),
        besteffort=entry.get("queue") == "besteffort",
    )


def launch(
    placements: list[Any], state: dict[str, Any], dry_run: bool, pending: dict[str, PendingInfo]
) -> None:
    for placement in placements:
        model, benchmark = placement.model.split("|")
        print(
            f"  -> {model} / {benchmark} on {placement.site}/{placement.cluster} ({placement.dtype})",
            flush=True,
        )
        if dry_run:
            continue
        job_id = submit(
            placement.site,
            placement.cluster,
            model,
            benchmark,
            placement.dtype,
            placement.queue,
            placement.shard_index,
            placement.shard_count,
        )
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
                "queue": placement.queue,
            }
        )


def cycle(project: Project, pools_config: list[dict[str, Any]], state: dict[str, Any], dry_run: bool) -> None:
    now = datetime.now()
    daytime = crosses_boundary(now)
    if daytime:
        print(f"{now:%H:%M}: default queue waits past 09:00 or 19:00; besteffort only", flush=True)
    pending = pending_work(project, repo_files(REPO))
    refresh_jobs(state, time.time(), {key: info[1] for key, info in pending.items()})
    save_state(state)
    _cancel_orphans(pools_config, state)
    placements = _placements(pending, pools_config, state, daytime)
    summary = f"pending {len(pending)} work items; running {len(state['jobs'])}; placements {len(placements)}"
    print(f"{now:%H:%M}: {summary}", flush=True)
    launch(placements, state, dry_run, pending)
    save_state(state)
    for line in usage_lines(sorted({entry["site"] for entry in pools_config})):
        print(f"  policy {line}", flush=True)


def _cancel_orphans(pools_config: list[dict[str, Any]], state: dict[str, Any]) -> None:
    tracked = {(job["site"], job["id"]) for job in state["jobs"]}
    for entry in {entry["site"] for entry in pools_config}:
        for job_id in waiting_orphans(entry, tracked):
            cancel(entry, job_id)
            print(f"  cancelled untracked waiting job {job_id} on {entry}", flush=True)


def _placements(
    pending: dict[str, PendingInfo], pools_config: list[dict[str, Any]], state: dict[str, Any], daytime: bool
) -> list[Any]:
    return choose_placements(
        _pending_pairs(pending, state),
        _pools_for(pools_config, state, daytime),
        running_models=frozenset(job["model"] for job in state["jobs"]),
        running_per_site=Counter(job["site"] for job in state["jobs"]),
        running_besteffort=sum(job.get("queue") == "besteffort" for job in state["jobs"]),
    )


def _pools_for(pools_config: list[dict[str, Any]], state: dict[str, Any], daytime: bool) -> list[GpuPool]:
    pools = build_pools(pools_config, state)
    return [pool for pool in pools if pool.besteffort] if daytime else pools


def _pending_pairs(pending: dict[str, PendingInfo], state: dict[str, Any]) -> list[Pending]:
    return [
        Pending(key, need, fp16, jobs_for_pair(items)) for _, key, need, fp16, items in _ready(pending, state)
    ]


def _ready(pending: dict[str, PendingInfo], state: dict[str, Any]) -> list[tuple[Any, ...]]:
    no_progress = state.get("no_progress", {})
    return sorted(
        (info[2], key, info[0], info[3], info[1])
        for key, info in pending.items()
        if no_progress.get(key, 0) < MAX_NO_PROGRESS
    )


def _finished(project: Project, state: dict[str, Any]) -> bool:
    remaining = pending_work(project, repo_files(REPO))
    no_progress = state.get("no_progress", {})
    blocked = sorted(key for key in remaining if no_progress.get(key, 0) >= MAX_NO_PROGRESS)
    if len(blocked) != len(remaining):
        return False
    gaps = f"; accepted gaps: {', '.join(blocked)}" if blocked else ""
    print(f"all shards and scores are in the dataset{gaps}", flush=True)
    return True


def run(dry_run: bool, sites: Path) -> None:
    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    pools_config = load_pools(sites)
    state = load_state()
    while True:
        try:
            cycle(project, pools_config, state, dry_run)
            if dry_run or _finished(project, state):
                return
        except Exception as error:
            print(f"{datetime.now():%H:%M}: cycle failed, retrying in {POLL_S}s: {error!r}", flush=True)
        time.sleep(POLL_S)
