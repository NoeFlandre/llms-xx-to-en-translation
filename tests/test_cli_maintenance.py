import pytest

from xx_to_en import crap
from xx_to_en.cli import main


def test_crap_command_exits_with_the_gate_code(monkeypatch):
    monkeypatch.setattr(crap, "run", lambda coverage, source, threshold: 1)
    with pytest.raises(SystemExit) as exit_info:
        main(["crap"])
    assert exit_info.value.code == 1


def test_crap_command_passes_silently(monkeypatch, capsys):
    monkeypatch.setattr(crap, "run", lambda coverage, source, threshold: 0)
    assert main(["crap"]) == 0
    assert capsys.readouterr().out == ""
