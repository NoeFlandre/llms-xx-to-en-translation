import json

import pytest

from xx_to_en import crap


def test_within_keeps_only_lines_in_the_range():
    assert crap._within({1, 2, 5, 9}, 2, 5) == {2, 5}


def test_function_coverage_is_the_executed_share_of_the_range():
    assert crap.function_coverage({2, 3}, {4, 5}, 2, 5) == pytest.approx(0.5)
    assert crap.function_coverage({2, 3, 4, 5}, set(), 2, 5) == 1.0
    assert crap.function_coverage(set(), set(), 2, 5) == 1.0


def test_crap_score_combines_complexity_and_missing_coverage():
    assert crap.crap_score(3, 1.0) == 3
    assert crap.crap_score(3, 0.0) == 12


def test_file_coverage_reads_executed_and_missing_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(crap, "ROOT", tmp_path)
    data = {"files": {"m.py": {"executed_lines": [1, 2], "missing_lines": [3]}}}
    assert crap._file_coverage(data, tmp_path / "m.py") == ({1, 2}, {3})
    assert crap._file_coverage(data, tmp_path / "other.py") is None


def test_function_end_finds_the_last_line_of_a_named_function(tmp_path):
    source = tmp_path / "m.py"
    source.write_text(
        "def f():\n    return 1\n\n\nclass C:\n    def g(self):\n        return 2\n", encoding="utf-8"
    )
    assert crap._function_end(source, 1, "f") == 2
    assert crap._function_end(source, 6, "g") == 7
    assert crap._function_end(source, 99, "missing") == 99


def test_scan_scores_every_function_and_method(tmp_path, monkeypatch):
    monkeypatch.setattr(crap, "ROOT", tmp_path)
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "m.py").write_text(
        "def f(x):\n    if x:\n        return 1\n    return 2\n\n"
        "class C:\n    def g(self):\n        return 3\n",
        encoding="utf-8",
    )
    coverage = {"files": {"pkg/m.py": {"executed_lines": [1, 2, 3, 4, 7, 8], "missing_lines": []}}}
    scores = {label: score for score, label, _, _, _ in crap.scan(package, coverage)}
    assert scores == {"pkg/m.py:f": 2.0, "pkg/m.py:g": 1.0}


def test_run_fails_when_a_function_reaches_the_threshold(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(crap, "ROOT", tmp_path)
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "m.py").write_text("def f(x):\n    if x:\n        return 1\n    return 2\n", encoding="utf-8")
    coverage_file = tmp_path / "coverage.json"
    coverage_file.write_text(
        json.dumps({"files": {"pkg/m.py": {"executed_lines": [], "missing_lines": [1, 2, 3, 4]}}}),
        encoding="utf-8",
    )
    assert crap.run(coverage_file, package, threshold=2.0) == 1
    assert "FAIL" in capsys.readouterr().out
    assert crap.run(coverage_file, package, threshold=50.0) == 0
