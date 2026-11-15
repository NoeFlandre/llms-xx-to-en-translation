from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from xx_to_en import worker
from xx_to_en.benchmarks import Segment
from xx_to_en.cache import OutputCache
from xx_to_en.config import BenchmarkConfig, ModelConfig
from xx_to_en.languages import LanguageMap
from xx_to_en.metric_runtime import Scorers
from xx_to_en.plan import Shard
from xx_to_en.runner import Decoding, Translation
from xx_to_en.tables import read_table
from xx_to_en.worker import (
    attach_text,
    generate_shard,
    hypothesis_rows,
    may_start_next,
    public_rows,
    run_budgeted,
    segment_index,
)


def _segment(segment_id: str = "0", lang: str = "fra_Latn") -> Segment:
    return Segment("flores_plus_devtest", lang, lang, segment_id, "Bonjour.", "Hello.")


def test_may_start_next_before_any_shard_only_checks_elapsed_time():
    assert may_start_next(0.0, None, 100.0)
    assert not may_start_next(100.0, None, 100.0)


def test_may_start_next_reserves_room_for_the_last_shard():
    assert may_start_next(50.0, 40.0, 100.0)
    assert not may_start_next(70.0, 40.0, 100.0)


def test_run_budgeted_stops_when_the_next_shard_would_overrun():
    clock_state = {"now": 0.0}
    calls: list[str] = []

    def step(shard: Shard) -> None:
        calls.append(shard.shard_id)
        clock_state["now"] += 10.0

    shards = [Shard("m", "wmt22_xen", i, ("ces_Latn",)) for i in range(5)]
    finished = run_budgeted(shards, budget_s=25.0, step=step, clock=lambda: clock_state["now"])
    assert [s.index for s in finished] == [0, 1]
    assert calls == ["wmt22_xen-000", "wmt22_xen-001"]


def test_run_budgeted_with_no_budget_does_nothing():
    shards = [Shard("m", "wmt22_xen", 0, ("ces_Latn",))]
    assert run_budgeted(shards, budget_s=0.0, step=lambda s: None, clock=lambda: 0.0) == []


def test_hypothesis_rows_carry_ids_and_output_but_no_text():
    translation = Translation(_segment(), "Hello.", output_tokens=2, cached=False)
    row = hypothesis_rows([translation])[0]
    assert row["hypothesis"] == "Hello."
    assert row["segment_id"] == "0"
    assert row["source"] == "Bonjour."
    assert row["reference"] == "Hello."


def test_attach_text_joins_by_benchmark_language_and_id():
    rows = hypothesis_rows([Translation(_segment("7"), "Hello.", 2, False)])
    index = segment_index([_segment("7")])
    joined = attach_text(rows, index)
    assert joined[0]["source"] == "Bonjour."
    assert joined[0]["reference"] == "Hello."


def test_attach_text_fails_loudly_on_unknown_segment():
    rows = hypothesis_rows([Translation(_segment("7"), "Hello.", 2, False)])
    with pytest.raises(KeyError):
        attach_text(rows, segment_index([_segment("8")]))


def test_public_rows_keep_source_and_reference():
    joined = attach_text(
        hypothesis_rows([Translation(_segment(), "Hello.", 2, False)]),
        segment_index([_segment()]),
    )
    cleaned = public_rows(joined)[0]
    assert cleaned["source"] == "Bonjour."
    assert cleaned["reference"] == "Hello."
    assert cleaned["hypothesis"] == "Hello."


def test_hypothesis_file_round_trip(tmp_path: Path):
    from xx_to_en.tables import read_table, write_table

    path = tmp_path / "shard.parquet"
    rows = hypothesis_rows([Translation(_segment(), "Hello.", 2, True)])
    write_table(path, rows)
    assert read_table(path) == rows


def test_score_rows_with_missing_metrics_survive_parquet(tmp_path: Path):
    from xx_to_en.tables import read_table, write_table

    rows = [
        {"segment_id": "0", "comet22": None, "chrf_pp": 40.0},
        {"segment_id": "1", "comet22": 0.8, "chrf_pp": 50.0},
    ]
    path = tmp_path / "scores.parquet"
    write_table(path, rows)
    assert read_table(path) == rows


class _FakeScorers:
    """Records what it was asked to score; COMET and MetricX return constants, GlotLID echoes English."""

    def __init__(self) -> None:
        self.comet_inputs: list[dict[str, str]] = []
        self.metricx_inputs: list[dict[str, str]] = []
        self.glotlid_texts: list[str] = []

    def comet(self, inputs: list[dict[str, str]]) -> list[float]:
        self.comet_inputs = list(inputs)
        return [0.75 for _ in inputs]

    def metricx(self, inputs: list[dict[str, str]]) -> list[float]:
        self.metricx_inputs = list(inputs)
        return [2.5 for _ in inputs]

    def glotlid(self, texts: list[str]) -> list[str]:
        self.glotlid_texts = list(texts)
        return ["eng_Latn" for _ in texts]


def test_score_shard_scores_neural_languages_only_and_writes_public_rows(tmp_path: Path, monkeypatch):
    from xx_to_en import worker
    from xx_to_en.config import BenchmarkConfig
    from xx_to_en.languages import LanguageMap
    from xx_to_en.tables import read_table, write_table

    segments = {
        "fra_Latn": Segment("flores_plus_devtest", "fra_Latn", "fra_Latn", "0", "Bonjour.", "Hello."),
        "ceb_Latn": Segment("flores_plus_devtest", "ceb_Latn", "ceb_Latn", "0", "Kumusta.", "Hello."),
    }
    monkeypatch.setattr(worker, "load_segments", lambda config, lang, bench: [segments[lang]])
    shard = Shard("m", "flores_plus_devtest", 0, ("ceb_Latn", "fra_Latn"))
    benchmark = BenchmarkConfig(
        name="flores_plus_devtest",
        description="test",
        hf_repo="org/bench",
        direction="X->EN",
        origin="native",
        license="cc-by-4.0",
        access="open",
    )
    hypotheses = tmp_path / "hyp.parquet"
    write_table(
        hypotheses,
        hypothesis_rows(
            [
                Translation(segments["fra_Latn"], "Hello there.", 2, False),
                Translation(segments["ceb_Latn"], "Hello.", 1, False),
            ]
        ),
    )
    scorers = _FakeScorers()
    out = tmp_path / "scores.parquet"

    worker.score_shard(
        hypotheses,
        out,
        shard=shard,
        benchmarks={"flores_plus_devtest": benchmark},
        language_map=LanguageMap({"fra_Latn": {"iso639_1": "fr"}}),
        scorers=cast(Scorers, scorers),
    )

    assert scorers.comet_inputs == [{"src": "Bonjour.", "mt": "Hello there.", "ref": "Hello."}]
    assert scorers.metricx_inputs == [
        {"source": "Bonjour.", "hypothesis": "Hello there.", "reference": "Hello."}
    ]
    assert scorers.glotlid_texts == ["Hello there.", "Hello."]
    by_lang = {row["corpus_lang"]: row for row in read_table(out)}
    assert by_lang["fra_Latn"]["comet22"] == 0.75
    assert by_lang["fra_Latn"]["metricx24"] == 2.5
    assert by_lang["fra_Latn"]["neural_covered"] is True
    assert by_lang["ceb_Latn"]["comet22"] is None
    assert by_lang["ceb_Latn"]["metricx24"] is None
    assert by_lang["ceb_Latn"]["neural_covered"] is False
    assert by_lang["ceb_Latn"]["detected_lang"] == "eng_Latn"
    assert isinstance(by_lang["fra_Latn"]["chrf_pp"], float)
    assert by_lang["fra_Latn"]["source"] == "Bonjour."
    assert by_lang["fra_Latn"]["reference"] == "Hello."


class _FakeTranslate:
    """Stands in for ``runner.translate``: one translation per segment, echoing its reference."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, segments: list[Segment], **settings: Any) -> list[Translation]:
        self.calls.append({"segments": list(segments), **settings})
        return [
            Translation(segment, segment.reference, output_tokens=1, cached=False) for segment in segments
        ]


@pytest.fixture
def fake_translate(monkeypatch: pytest.MonkeyPatch) -> _FakeTranslate:
    """Replaces translation and segment loading; pins the creation time stamped on each row."""
    fake = _FakeTranslate()
    monkeypatch.setattr(worker, "translate", fake)
    monkeypatch.setattr(worker, "load_segments", lambda config, lang, bench: [_segment(lang=lang)])
    monkeypatch.setattr(worker, "created_now", lambda: "2026-01-02T03:04:05+00:00")
    return fake


_MODEL = cast(ModelConfig, SimpleNamespace(name="fake-model"))
_BACKEND = object()
_DECODING = Decoding(seed=3)


def _iso(code: str) -> str | None:
    return None


def _generate_shard(tmp_path: Path, run: dict[str, Any] | None = None) -> Path:
    out = tmp_path / "hyp.parquet"
    benchmark = BenchmarkConfig(
        name="flores_plus_devtest",
        description="test",
        hf_repo="org/bench",
        direction="X->EN",
        origin="native",
        license="cc-by-4.0",
        access="open",
    )
    generate_shard(
        Shard("m", "flores_plus_devtest", 0, ("ceb_Latn", "fra_Latn")),
        model=_MODEL,
        backend=_BACKEND,
        cache=OutputCache(tmp_path),
        benchmarks={"flores_plus_devtest": benchmark},
        language_map=LanguageMap({}),
        iso=_iso,
        decoding=_DECODING,
        out_file=out,
        run=run,
    )
    return out


def test_generate_shard_translates_languages_in_shard_order_and_writes_their_rows(
    tmp_path: Path, fake_translate: _FakeTranslate
):
    out = _generate_shard(tmp_path, run={"run_id": "r1"})

    assert [call["segments"][0].corpus_lang for call in fake_translate.calls] == ["ceb_Latn", "fra_Latn"]
    rows = read_table(out)
    assert [row["corpus_lang"] for row in rows] == ["ceb_Latn", "fra_Latn"]
    assert [row["hypothesis"] for row in rows] == ["Hello.", "Hello."]
    assert all(row["run_id"] == "r1" for row in rows)
    assert all(row["generated_at"] == "2026-01-02T03:04:05+00:00" for row in rows)


def test_generate_shard_passes_model_backend_cache_decoding_and_iso_to_translate(
    tmp_path: Path, fake_translate: _FakeTranslate
):
    _generate_shard(tmp_path)

    call = fake_translate.calls[0]
    assert call["model"] is _MODEL
    assert call["backend"] is _BACKEND
    assert call["cache"].root == tmp_path
    assert call["decoding"] is _DECODING
    assert call["iso"] is _iso


def test_generate_shard_without_run_settings_stamps_only_the_creation_time(
    tmp_path: Path, fake_translate: _FakeTranslate
):
    rows = read_table(_generate_shard(tmp_path))

    assert rows[0]["generated_at"] == "2026-01-02T03:04:05+00:00"
    assert "run_id" not in rows[0]
