"""The dispatcher script (scripts/g5k/dispatch.py): an ended job is counted once, even across a kill."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from xx_to_en.cli import Project

ROOT = Path(__file__).resolve().parents[1]


def _load_dispatcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dispatch_script", ROOT / "scripts" / "g5k" / "dispatch.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ended_jobs_are_saved_before_a_killed_process_could_count_them_again(tmp_path, monkeypatch):
    dispatch = _load_dispatcher()
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(dispatch, "STATE_FILE", state_file)
    monkeypatch.setattr(dispatch, "repo_files", lambda repo: set())
    monkeypatch.setattr(dispatch, "pending_work", lambda project, done: {"m|b": (16, 3, 1.0, False)})
    monkeypatch.setattr(dispatch, "job_state", lambda site, job_id: "Terminated")
    monkeypatch.setattr(dispatch, "waiting_orphans", lambda site, tracked: [])

    def killed(*args):
        raise SystemExit("killed before launching")

    monkeypatch.setattr(dispatch, "launch", killed)
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
    state = {"jobs": [job], "attempts": {}, "failures": {}, "no_progress": {}}
    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    with pytest.raises(SystemExit):
        dispatch.cycle(project, [], state, dry_run=False)
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["jobs"] == []
    assert saved["no_progress"] == {"m|b": 1}
