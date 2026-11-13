import pytest

from xx_to_en.cache import OutputCache
from xx_to_en.lengths import max_new_tokens
from xx_to_en.metrics import corpus_chrf_pp, segment_chrf_pp
from xx_to_en.report import per_language_winner, volume_weighted_mean


def test_chrf_pp_identical_outputs_score_100():
    refs = ["The cat sleeps on the mat.", "Paris is the capital of France."]
    assert corpus_chrf_pp(refs, refs) == pytest.approx(100.0)


def test_chrf_pp_prefers_closer_output():
    refs = ["The cat sleeps on the mat."]
    assert corpus_chrf_pp(["The cat sleeps on the mat."], refs) > corpus_chrf_pp(["Dogs run fast."], refs)


def test_segment_chrf_pp_returns_one_score_per_segment():
    scores = segment_chrf_pp(["a b c", "d e f"], ["a b c", "x y z"])
    assert len(scores) == 2
    assert scores[0] == pytest.approx(100.0)
    assert scores[1] < scores[0]


def test_chrf_pp_rejects_length_mismatch_and_empty_input():
    with pytest.raises(ValueError, match="hypotheses"):
        corpus_chrf_pp(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="no segments"):
        corpus_chrf_pp([], [])


def test_cache_key_is_order_independent_and_sensitive_to_values():
    assert OutputCache.key(model="m", text="t") == OutputCache.key(text="t", model="m")
    assert OutputCache.key(model="m", text="t") != OutputCache.key(model="m", text="u")


def test_cache_round_trip_and_get_or_compute(tmp_path):
    cache = OutputCache(tmp_path)
    key = OutputCache.key(model="m", text="bonjour")
    assert cache.get(key) is None
    calls = []

    def compute():
        calls.append(1)
        return "hello"

    assert cache.get_or_compute(key, compute) == "hello"
    assert cache.get_or_compute(key, compute) == "hello"
    assert calls == [1]
    assert cache.get(key) == "hello"
    assert not list(tmp_path.rglob("*.tmp"))


def test_max_new_tokens_scales_and_clamps():
    assert max_new_tokens(5) == 32
    assert max_new_tokens(40) == 120
    assert max_new_tokens(10_000) == 2048


def test_max_new_tokens_rejects_bad_inputs():
    with pytest.raises(ValueError):
        max_new_tokens(-1)
    with pytest.raises(ValueError):
        max_new_tokens(5, ratio=0)
    with pytest.raises(ValueError):
        max_new_tokens(5, floor=100, ceiling=10)


def test_volume_weighted_mean_ignores_unweighted_languages():
    scores = {"a": 10.0, "b": 20.0, "c": 99.0}
    volumes = {"a": 1, "b": 3, "c": 0}
    assert volume_weighted_mean(scores, volumes) == pytest.approx(17.5)


def test_volume_weighted_mean_needs_overlap():
    with pytest.raises(ValueError, match="positive volume"):
        volume_weighted_mean({"a": 1.0}, {"b": 5})


def test_per_language_winner_and_ties():
    table = {
        "model-x": {"fra": 40.0, "deu": 30.0},
        "model-y": {"fra": 35.0, "deu": 30.0, "ita": 50.0},
    }
    assert per_language_winner(table) == {"deu": "model-x", "fra": "model-x", "ita": "model-y"}
