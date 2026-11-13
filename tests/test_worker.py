from pathlib import Path

import pytest

from xx_to_en.benchmarks import Segment
from xx_to_en.plan import Shard
from xx_to_en.runner import Translation
from xx_to_en.worker import (
    attach_text,
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
        clock_state["now"] += 10.0  # every shard takes 10 s

    shards = [Shard("m", "wmt22_xen", i, ("ces_Latn",)) for i in range(5)]
    finished = run_budgeted(shards, budget_s=25.0, step=step, clock=lambda: clock_state["now"])
    # shard 0 runs 0-10 s, shard 1 runs 10-20 s; at 20 s, 20 + 10 > 25, so shard 2 is not started
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
