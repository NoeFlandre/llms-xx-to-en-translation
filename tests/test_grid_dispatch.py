"""The dispatcher loop: an ended job is counted once, even across a kill, and each step does what it says."""

import json
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from xx_to_en import grid_dispatch as dispatch
from xx_to_en.cli import Project
from xx_to_en.dispatch import GpuPool, Placement
from xx_to_en.plan import Shard
from xx_to_en.results_store import score_path, shard_path

ROOT = Path(__file__).resolve().parents[1]
ANY_PROJECT: Any = SimpleNamespace()


def _job(**overrides):
    job = {
        "id": 1,
        "site": "nancy",
        "cluster": "gpu",
        "model": "m|b",
        "submitted": 0.0,
        "state": "Running",
        "remaining_at_submit": 3,
        "queue": "default",
    }
    return {**job, **overrides}


def _state(jobs=(), no_progress=None, failures=None):
    return {
        "jobs": list(jobs),
        "attempts": {},
        "failures": failures or {},
        "no_progress": no_progress or {},
    }


def _fake_project(shards):
    model = SimpleNamespace(enabled=True, gpu_memory_gb_min=16, params_b=1.0, fp16_ok=True)
    disabled = SimpleNamespace(enabled=False)
    return SimpleNamespace(
        models={"m": model, "off": disabled},
        shards=lambda model, files: shards,
    )


def test_ended_jobs_are_saved_before_a_killed_process_could_count_them_again(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(dispatch, "STATE_FILE", state_file)
    monkeypatch.setattr(dispatch, "repo_files", lambda repo: set())
    monkeypatch.setattr(dispatch, "pending_work", lambda project, done: {"m|b": (16, 3, 1.0, False)})
    monkeypatch.setattr(dispatch, "job_state", lambda site, job_id: "Terminated")
    monkeypatch.setattr(dispatch, "waiting_orphans", lambda site, tracked: [])

    def killed(*args):
        raise SystemExit("killed before launching")

    monkeypatch.setattr(dispatch, "launch", killed)
    state = _state(jobs=[_job()])
    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    with pytest.raises(SystemExit):
        dispatch.cycle(project, [], state, dry_run=False)
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["jobs"] == []
    assert saved["no_progress"] == {"m|b": 1}


def test_ssh_returns_stdout_without_bashrc_noise(monkeypatch):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="bashrc: noise\nkeep\n", stderr="")

    monkeypatch.setattr(dispatch.subprocess, "run", fake_run)
    assert dispatch.ssh("nancy", "true") == "keep"


def test_load_pools_reads_the_sites_file(tmp_path):
    path = tmp_path / "sites.yaml"
    path.write_text("sites:\n  - site: nancy\n    cluster: gpu\n", encoding="utf-8")
    assert dispatch.load_pools(path) == [{"site": "nancy", "cluster": "gpu"}]


def test_state_round_trips_and_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "STATE_FILE", tmp_path / "state" / "state.json")
    assert dispatch.load_state() == {"jobs": [], "attempts": {}, "failures": {}, "no_progress": {}}
    state = _state(jobs=[_job()])
    dispatch.save_state(state)
    assert dispatch.load_state() == state


def test_a_job_that_uploads_resets_its_pair_counter(monkeypatch):
    monkeypatch.setattr(dispatch, "job_state", lambda site, job_id: "Terminated")
    state = _state(jobs=[_job()], no_progress={"m|b": 2})
    dispatch.refresh_jobs(state, now=0.0, remaining={"m|b": 2})
    assert state["jobs"] == []
    assert state["no_progress"] == {"m|b": 0}


def test_job_state_reads_the_oar_state_line(monkeypatch):
    monkeypatch.setattr(dispatch, "ssh", lambda site, command: "state = Running\n")
    assert dispatch.job_state("nancy", 5) == "Running"
    monkeypatch.setattr(dispatch, "ssh", lambda site, command: "")
    assert dispatch.job_state("nancy", 5) == "Gone"


def test_cancel_runs_oardel(monkeypatch):
    sent = []
    monkeypatch.setattr(dispatch, "ssh", lambda site, command: sent.append((site, command)) or "")
    dispatch.cancel("nancy", 5)
    assert sent == [("nancy", "oardel 5 >/dev/null 2>&1")]


def test_submit_returns_the_oar_job_id_or_none(monkeypatch):
    sent = []
    monkeypatch.setattr(dispatch, "ssh", lambda site, command: sent.append(command) or "OAR_JOB_ID=42\n")
    assert dispatch.submit("nancy", "gpu", "m", "tatoeba_xen", "bfloat16", "besteffort", 1, 2) == 42
    assert "-q besteffort" in sent[0]
    assert "MODEL=m BENCHMARK=tatoeba_xen" in sent[0]
    assert "SHARD_INDEX=1 SHARD_COUNT=2" in sent[0]
    monkeypatch.setattr(dispatch, "ssh", lambda site, command: "refused")
    assert dispatch.submit("nancy", "gpu", "m", "b", "bfloat16", "default") is None


def test_refresh_jobs_counts_ended_jobs_without_progress_and_drops_them(monkeypatch):
    monkeypatch.setattr(dispatch, "job_state", lambda site, job_id: "Terminated")
    state = _state(jobs=[_job()])
    dispatch.refresh_jobs(state, now=0.0, remaining={"m|b": 3})
    assert state["jobs"] == []
    assert state["no_progress"] == {"m|b": 1}


def test_refresh_jobs_keeps_progressing_jobs_and_cancels_only_stale_waiting_ones(monkeypatch):
    states = {1: "Running", 2: "Waiting", 3: "Waiting"}
    cancelled = []
    monkeypatch.setattr(dispatch, "job_state", lambda site, job_id: states[job_id])
    monkeypatch.setattr(dispatch, "cancel", lambda site, job_id: cancelled.append(job_id))
    now = dispatch.STALE_WAIT_S + 1
    state = _state(
        jobs=[
            _job(id=1),
            _job(id=2, state="Waiting", submitted=0.0),
            _job(id=3, state="Waiting", submitted=now),
        ]
    )
    dispatch.refresh_jobs(state, now=now, remaining={})
    assert cancelled == [2]
    assert [job["id"] for job in state["jobs"]] == [1, 3]
    assert state["jobs"][0]["state"] == "Running"
    assert state["no_progress"] == {}


def test_pending_work_lists_missing_generation_and_scoring_per_pair():
    shard = Shard("m", "b", 0, ("fra_Latn",))
    assert dispatch.pending_work(_fake_project([shard]), done=set()) == {"m|b": (16, 2, 1.0, True)}
    assert dispatch.pending_work(_fake_project([shard]), {shard_path(shard), score_path(shard)}) == {}


def test_waiting_orphans_lists_untracked_waiting_jobs(monkeypatch):
    monkeypatch.setattr(dispatch, "ssh", lambda site, command: "101\n102\n")
    assert dispatch.waiting_orphans("nancy", {("nancy", 101)}) == [102]


def test_site_ready_and_usage_lines_read_remote_output(monkeypatch):
    monkeypatch.setattr(dispatch, "ssh", lambda site, command: "ready\n")
    assert dispatch.site_ready("nancy")
    monkeypatch.setattr(dispatch, "ssh", lambda site, command: "policy ok")
    assert dispatch.usage_lines(["nancy"]) == ["nancy: policy ok"]


def test_probe_free_counts_the_free_gpus_it_is_given(monkeypatch):
    monkeypatch.setattr(dispatch, "ssh", lambda site, command: "")
    assert dispatch.probe_free("nancy") == Counter()


POOLS = [
    {"site": "nancy", "cluster": "gpu", "gpu_model": "L40S", "memory_gb": 48, "queue": "default"},
    {"site": "rennes", "cluster": "old", "gpu_model": "T4", "memory_gb": 16, "bf16": False},
]


def test_build_pools_offers_free_gpus_on_ready_sites_minus_waiting_jobs(monkeypatch):
    monkeypatch.setattr(dispatch, "site_ready", lambda site: site == "nancy")
    monkeypatch.setattr(dispatch, "probe_free", lambda site: Counter({"gpu|L40S": 3}))
    state = _state(jobs=[_job(site="nancy", cluster="gpu", state="Waiting")])
    assert dispatch.build_pools(POOLS, state) == [GpuPool("nancy", "gpu", 48, 2, True, besteffort=False)]


def test_build_pools_skips_clusters_that_failed_too_often(monkeypatch):
    monkeypatch.setattr(dispatch, "site_ready", lambda site: True)
    monkeypatch.setattr(dispatch, "probe_free", lambda site: Counter({"gpu|L40S": 3}))
    state = _state(failures={"nancy/gpu": dispatch.MAX_FAILURES_PER_CLUSTER})
    assert dispatch.build_pools(POOLS, state) == [GpuPool("rennes", "old", 16, 0, False, besteffort=False)]


def test_pools_for_keeps_only_besteffort_pools_during_the_daytime(monkeypatch):
    pools = [
        GpuPool("nancy", "gpu", 48, 1, True, besteffort=True),
        GpuPool("nancy", "gpu2", 48, 1, True, besteffort=False),
    ]
    monkeypatch.setattr(dispatch, "build_pools", lambda pools_config, state: pools)
    assert dispatch._pools_for([], _state(), daytime=True) == [pools[0]]
    assert dispatch._pools_for([], _state(), daytime=False) == pools


def test_ready_skips_pairs_that_hit_the_no_progress_limit():
    pending = {"a|b": (16, 2, 1.0, False), "c|d": (24, 3, 9.0, True)}
    state = _state(no_progress={"c|d": dispatch.MAX_NO_PROGRESS})
    assert dispatch._ready(pending, state) == [(1.0, "a|b", 16, False, 2)]


def test_launch_submits_placements_and_records_the_jobs(monkeypatch, capsys):
    monkeypatch.setattr(dispatch, "submit", lambda *args: 9)
    state = _state()
    dispatch.launch(
        [Placement(model="m|b", site="nancy", cluster="gpu")], state, False, {"m|b": (16, 4, 1.0, False)}
    )
    assert state["attempts"] == {"m|b": 1}
    assert state["jobs"][0]["id"] == 9
    assert state["jobs"][0]["remaining_at_submit"] == 4
    assert "-> m / b on nancy/gpu" in capsys.readouterr().out


def test_launch_counts_refused_submissions_per_cluster(monkeypatch, capsys):
    monkeypatch.setattr(dispatch, "submit", lambda *args: None)
    state = _state()
    dispatch.launch(
        [Placement(model="m|b", site="nancy", cluster="gpu")], state, False, {"m|b": (16, 4, 1.0, False)}
    )
    assert state["failures"] == {"nancy/gpu": 1}
    assert state["jobs"] == []
    assert "submission refused on nancy/gpu" in capsys.readouterr().out


def test_launch_only_prints_on_a_dry_run(monkeypatch):
    monkeypatch.setattr(dispatch, "submit", lambda *args: pytest.fail("no submission on a dry run"))
    state = _state()
    dispatch.launch(
        [Placement(model="m|b", site="nancy", cluster="gpu")], state, True, {"m|b": (16, 4, 1.0, False)}
    )
    assert state["jobs"] == []


def test_cancel_orphans_cancels_untracked_waiting_jobs(monkeypatch):
    cancelled = []
    monkeypatch.setattr(dispatch, "waiting_orphans", lambda site, tracked: [7])
    monkeypatch.setattr(dispatch, "cancel", lambda site, job_id: cancelled.append((site, job_id)))
    dispatch._cancel_orphans([{"site": "nancy"}], _state())
    assert cancelled == [("nancy", 7)]


def test_cycle_dry_run_prints_the_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dispatch, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(dispatch, "pending_work", lambda project, done: {"m|b": (16, 2, 1.0, False)})
    monkeypatch.setattr(dispatch, "repo_files", lambda repo: set())
    monkeypatch.setattr(dispatch, "waiting_orphans", lambda site, tracked: [])
    monkeypatch.setattr(dispatch, "build_pools", lambda pools_config, state: [])
    monkeypatch.setattr(dispatch, "usage_lines", lambda sites: ["nancy: ok"])
    monkeypatch.setattr(dispatch, "crosses_boundary", lambda now: True)
    dispatch.cycle(ANY_PROJECT, [{"site": "nancy"}], _state(), dry_run=True)
    out = capsys.readouterr().out
    assert "besteffort only" in out
    assert "pending 1 work items; running 0; placements 0" in out
    assert "policy nancy: ok" in out


def test_finished_reports_when_only_blocked_pairs_remain(monkeypatch, capsys):
    monkeypatch.setattr(dispatch, "pending_work", lambda project, done: {"a|b": (16, 1, 1.0, False)})
    monkeypatch.setattr(dispatch, "repo_files", lambda repo: set())
    blocked = _state(no_progress={"a|b": dispatch.MAX_NO_PROGRESS})
    assert dispatch._finished(ANY_PROJECT, blocked)
    assert "accepted gaps: a|b" in capsys.readouterr().out
    assert not dispatch._finished(ANY_PROJECT, _state())


def test_finished_is_true_when_nothing_is_pending(monkeypatch, capsys):
    monkeypatch.setattr(dispatch, "pending_work", lambda project, done: {})
    monkeypatch.setattr(dispatch, "repo_files", lambda repo: set())
    assert dispatch._finished(ANY_PROJECT, _state())
    assert "all shards and scores are in the dataset" in capsys.readouterr().out


def _stub_run(monkeypatch, cycle):
    monkeypatch.setattr(dispatch, "Project", lambda configs, inventory: "project")
    monkeypatch.setattr(dispatch, "load_pools", lambda path: [])
    monkeypatch.setattr(dispatch, "load_state", lambda: _state())
    monkeypatch.setattr(dispatch, "cycle", cycle)


def test_run_stops_after_one_dry_cycle(monkeypatch, tmp_path):
    calls = []
    _stub_run(monkeypatch, lambda project, pools, state, dry_run: calls.append(dry_run))
    dispatch.run(dry_run=True, sites=tmp_path / "sites.yaml")
    assert calls == [True]


def test_run_retries_a_failed_cycle_after_sleeping(monkeypatch, capsys):
    attempts = []

    def flaky(project, pools, state, dry_run):
        attempts.append(dry_run)
        if len(attempts) == 1:
            raise RuntimeError("network")

    slept = []
    _stub_run(monkeypatch, flaky)
    monkeypatch.setattr(dispatch.time, "sleep", slept.append)
    dispatch.run(dry_run=True, sites=Path("sites.yaml"))
    assert attempts == [True, True]
    assert slept == [dispatch.POLL_S]
    assert "cycle failed" in capsys.readouterr().out


def test_run_returns_once_the_dataset_is_complete(monkeypatch):
    _stub_run(monkeypatch, lambda project, pools, state, dry_run: None)
    monkeypatch.setattr(dispatch, "_finished", lambda project, state: True)
    monkeypatch.setattr(dispatch.time, "sleep", lambda seconds: pytest.fail("must not sleep"))
    dispatch.run(dry_run=False, sites=Path("sites.yaml"))
