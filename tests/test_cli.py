from pathlib import Path

import pytest

from xx_to_en.cli import check_configs, main

ROOT = Path(__file__).resolve().parents[1]


def test_check_configs_reports_counts():
    message = check_configs(ROOT / "configs")
    assert message == "ok: 17 models, 5 benchmarks, languages.yaml"


def test_coverage_command_writes_matrix(tmp_path, capsys):
    out = tmp_path / "coverage.md"
    code = main(
        [
            "coverage",
            "--configs",
            str(ROOT / "configs"),
            "--inventory",
            str(ROOT / "data" / "benchmark_inventory.json"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert out.read_text(encoding="utf-8").startswith("# Language x benchmark coverage")
    assert "wrote" in capsys.readouterr().out


def test_check_configs_command_returns_zero(capsys):
    assert main(["check-configs", "--root", str(ROOT / "configs")]) == 0
    assert "ok:" in capsys.readouterr().out


def test_unknown_command_exits(capsys):
    with pytest.raises(SystemExit):
        main(["nope"])


def test_module_entry_point_runs_check_configs(monkeypatch):
    import runpy

    monkeypatch.setattr("sys.argv", ["xx2en", "check-configs", "--root", str(ROOT / "configs")])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("xx_to_en.__main__", run_name="__main__", alter_sys=True)
    assert exit_info.value.code == 0
