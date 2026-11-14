import pytest
from huggingface_hub.utils import EntryNotFoundError

from xx_to_en import sources


def _fake_hub(monkeypatch, available: set[str]) -> list[str]:
    requested: list[str] = []

    def hub_file(repo: str, path: str) -> str:
        requested.append(path)
        if path not in available:
            raise EntryNotFoundError("missing")
        return path

    monkeypatch.setattr(sources, "_hub_file", hub_file)
    monkeypatch.setattr(sources, "_lines", lambda path: [f"line from {path}"])
    return requested


def test_tatoeba_uses_the_first_direction_that_exists(monkeypatch):
    requested = _fake_hub(monkeypatch, {"data/test/tatoeba-test.fra-eng.tsv"})
    assert sources._tatoeba_lines("repo", "fra") == ["line from data/test/tatoeba-test.fra-eng.tsv"]
    assert requested == ["data/test/tatoeba-test.fra-eng.tsv"]


def test_tatoeba_falls_back_to_the_english_first_direction(monkeypatch):
    requested = _fake_hub(monkeypatch, {"data/test/tatoeba-test.eng-fra.tsv"})
    assert sources._tatoeba_lines("repo", "fra") == ["line from data/test/tatoeba-test.eng-fra.tsv"]
    assert requested == ["data/test/tatoeba-test.fra-eng.tsv", "data/test/tatoeba-test.eng-fra.tsv"]


def test_tatoeba_raises_when_neither_direction_exists(monkeypatch):
    _fake_hub(monkeypatch, set())
    with pytest.raises(FileNotFoundError, match="no Tatoeba test file for fra"):
        sources._tatoeba_lines("repo", "fra")
