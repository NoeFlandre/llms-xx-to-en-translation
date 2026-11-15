from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from xx_to_en import card_build
from xx_to_en.card import Cell
from xx_to_en.cli import Project
from xx_to_en.languages import corpus_languages
from xx_to_en.plan import Shard
from xx_to_en.results_store import score_path, shard_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project():
    return Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")


def test_progress_has_no_cells_for_disabled_models():
    fake: Any = SimpleNamespace(models={"m": SimpleNamespace(enabled=False)}, benchmarks={"b": None})
    assert card_build.progress(fake, set()) == {"m": {}}


def test_progress_counts_planned_shards_per_benchmark(project):
    name = next(n for n, model in project.models.items() if model.enabled)
    cells = card_build.progress(project, set())[name]
    assert set(cells) == set(project.benchmarks)
    assert all(cell.generated == 0 and cell.scored == 0 for cell in cells.values())
    assert sum(cell.total for cell in cells.values()) > 0


def test_cell_counts_generated_and_scored_shards():
    own = [Shard("m", "b", 0, ("fra_Latn",)), Shard("m", "b", 1, ("deu_Latn",))]
    files = {shard_path(own[0]), score_path(own[0]), shard_path(own[1])}
    assert card_build._cell(own, files) == Cell(generated=2, scored=1, total=2)


def test_language_means_average_each_language(monkeypatch):
    records = [
        {"corpus_lang": "fra_Latn", "chrf_pp": 40.0},
        {"corpus_lang": "fra_Latn", "chrf_pp": 50.0},
    ]
    assert card_build._language_means(records) == {"fra_Latn": 45.0}


def test_metric_mean_skips_missing_values():
    records = [{"comet22": 0.8}, {"comet22": None}, {}]
    assert card_build._metric_mean(records, "comet22") == pytest.approx(0.8)
    assert card_build._metric_mean([{}], "comet22") is None


def test_failure_rate_is_the_share_of_rows_with_any_failure_flag():
    assert card_build._failure_rate([{"empty": True}, {}]) == 0.5


def test_overall_chrf_weights_languages_by_volume_and_is_zero_without_scores():
    means = {"fra_Latn": 40.0, "deu_Latn": 60.0}
    volumes = {"fra_Latn": 1, "deu_Latn": 3}
    common = {"fra_Latn", "deu_Latn"}
    assert card_build._overall_chrf(means, common, volumes) == pytest.approx(55.0)
    assert card_build._overall_chrf(means, set(), volumes) == 0.0


def test_winners_pick_the_best_model_per_language():
    per_language = {"a": {"fra_Latn": 40.0, "deu_Latn": 50.0}, "b": {"fra_Latn": 45.0, "deu_Latn": 30.0}}
    assert card_build._winners(per_language) == {"deu_Latn": ("a", 50.0), "fra_Latn": ("b", 45.0)}
    assert card_build._winners({}) == {}


def test_summarize_scores_each_model_over_the_languages_all_models_share(project):
    langs = sorted(corpus_languages(project.inventory.corpus_rows))[:2]
    records = [
        {"corpus_lang": lang, "chrf_pp": 40.0 + index, "comet22": 0.8, "metricx24": 2.0, "empty": False}
        for index, lang in enumerate(langs)
    ]
    summaries, winners, common_count = card_build.summarize(project, {"m1": records, "m2": records})
    assert {summary.model for summary in summaries} == {"m1", "m2"}
    assert all(summary.segments == 2 for summary in summaries)
    assert common_count == 2
    assert set(winners) == set(langs)


def test_summarize_of_nothing_is_empty(project):
    assert card_build.summarize(project, {}) == ([], {}, 0)


def test_score_rows_groups_records_by_model(monkeypatch):
    monkeypatch.setattr(card_build, "hf_hub_download", lambda repo_id, filename, repo_type: filename)
    monkeypatch.setattr(card_build, "read_table", lambda path: [{"file": str(path)}])
    rows = card_build.score_rows({"scores/m1/a.parquet", "scores/m1/b.parquet", "results/m1/c.parquet"})
    assert list(rows) == ["m1"]
    assert len(rows["m1"]) == 2


def test_mean_of_nothing_is_none():
    assert card_build.mean([]) is None
    assert card_build.mean([1.0, 3.0]) == 2.0


def test_run_writes_the_card_and_uploads_it_unless_told_not_to(tmp_path, monkeypatch):
    monkeypatch.setattr(card_build, "ROOT", tmp_path)
    (tmp_path / "configs").symlink_to(ROOT / "configs")
    (tmp_path / "data").symlink_to(ROOT / "data")
    monkeypatch.setattr(card_build, "repo_files", lambda repo: set())
    monkeypatch.setattr(card_build, "score_rows", lambda files: {})
    uploads = []

    class FakeApi:
        def upload_file(self, **kwargs):
            uploads.append(kwargs)

    monkeypatch.setattr(card_build, "HfApi", FakeApi)
    card_build.run(no_upload=True)
    card = tmp_path / "docs" / "dataset_card.md"
    assert "## Known gaps" in card.read_text(encoding="utf-8")
    assert uploads == []
    card_build.run(no_upload=False)
    assert uploads[0]["path_in_repo"] == "README.md"
