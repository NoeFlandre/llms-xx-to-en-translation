from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from xx_to_en import shard_check


def test_tatoeba_shards_lists_only_result_parquets(monkeypatch):
    class FakeApi:
        def list_repo_files(self, repo, repo_type):
            return [
                "results/m/tatoeba_xen/a.parquet",
                "results/m/flores/b.parquet",
                "scores/m/tatoeba_xen/a.parquet",
                "results/m/tatoeba_xen/readme.md",
            ]

    monkeypatch.setattr(shard_check, "HfApi", FakeApi)
    assert shard_check.tatoeba_shards() == ["results/m/tatoeba_xen/a.parquet"]


def test_mismatches_flags_corpora_whose_sources_changed(tmp_path, monkeypatch):
    path = tmp_path / "shard.parquet"
    rows = [
        {"corpus_lang": "fra_Latn", "segment_id": "1", "source": "bonjour"},
        {"corpus_lang": "deu_Latn", "segment_id": "1", "source": "hallo"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    monkeypatch.setattr(shard_check, "hf_hub_download", lambda repo_id, filename, repo_type: str(path))
    segments = {
        "fra_Latn": [SimpleNamespace(segment_id="1", source="bonjour")],
        "deu_Latn": [SimpleNamespace(segment_id="1", source="changed")],
    }
    monkeypatch.setattr(shard_check, "load_segments", lambda benchmark, corpus, code: segments[corpus])
    project: Any = SimpleNamespace(
        benchmarks={"tatoeba_xen": "bench"},
        language_map=SimpleNamespace(benchmark_code=lambda corpus, benchmark: corpus[:3]),
    )
    assert shard_check.mismatches("results/x.parquet", project, {}) == ["deu_Latn"]


def test_corpora_without_a_benchmark_code_have_no_pinned_segments(monkeypatch):
    monkeypatch.setattr(shard_check, "load_segments", lambda benchmark, corpus, code: [])
    project: Any = SimpleNamespace(
        benchmarks={"tatoeba_xen": "bench"},
        language_map=SimpleNamespace(benchmark_code=lambda corpus, benchmark: None),
    )
    assert shard_check._pinned_segments("fra_Latn", project) == {}


def test_run_reports_each_mismatched_shard(monkeypatch, capsys):
    monkeypatch.setattr(shard_check, "Project", lambda configs, inventory: "project")
    monkeypatch.setattr(shard_check, "tatoeba_shards", lambda: ["a.parquet", "b.parquet"])
    monkeypatch.setattr(
        shard_check,
        "mismatches",
        lambda path, project, cache: ["fra_Latn"] if path == "a.parquet" else [],
    )
    assert shard_check.run() == "checked 2 shards, mismatched 1"
    assert "MISMATCH a.parquet ['fra_Latn']" in capsys.readouterr().out
