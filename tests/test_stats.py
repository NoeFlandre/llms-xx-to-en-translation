import numpy as np
import pytest

from xx_to_en.stats import bootstrap_ci, paired_bootstrap_diff


def test_ci_of_constant_values_is_degenerate():
    estimate = bootstrap_ci([3.0] * 50, n_resamples=200)
    assert estimate.value == pytest.approx(3.0)
    assert estimate.low == pytest.approx(3.0)
    assert estimate.high == pytest.approx(3.0)


def test_ci_brackets_the_mean_and_is_reproducible():
    values = np.linspace(0, 10, 101)
    first = bootstrap_ci(values, n_resamples=500, seed=7)
    second = bootstrap_ci(values, n_resamples=500, seed=7)
    assert first == second
    assert first.low <= first.value <= first.high


def test_wider_confidence_gives_wider_interval():
    values = list(range(200))
    narrow = bootstrap_ci(values, n_resamples=2000, confidence=0.5, seed=1)
    wide = bootstrap_ci(values, n_resamples=2000, confidence=0.99, seed=1)
    assert (wide.high - wide.low) > (narrow.high - narrow.low)


def test_paired_identical_systems_have_zero_difference():
    scores = [1.0, 2.0, 3.0, 4.0]
    estimate = paired_bootstrap_diff(scores, scores, n_resamples=100)
    assert estimate == paired_bootstrap_diff(scores, scores, n_resamples=100)
    assert estimate.value == 0.0 and estimate.low == 0.0 and estimate.high == 0.0


def test_paired_clear_winner_excludes_zero():
    better = [10.0 + i * 0.01 for i in range(300)]
    worse = [5.0 + i * 0.01 for i in range(300)]
    estimate = paired_bootstrap_diff(better, worse, n_resamples=1000)
    assert estimate.low > 0


def test_paired_requires_same_length():
    with pytest.raises(ValueError, match="same number"):
        paired_bootstrap_diff([1.0], [1.0, 2.0])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_ci([])


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
def test_confidence_must_be_open_interval(confidence):
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0, 2.0], confidence=confidence)


def test_chunked_resampling_matches_single_chunk(monkeypatch):
    from xx_to_en import stats

    values = list(range(40))
    baseline = bootstrap_ci(values, n_resamples=300, seed=3)
    monkeypatch.setattr(stats, "MAX_CELLS_PER_CHUNK", 40 * 7)
    chunked = bootstrap_ci(values, n_resamples=300, seed=3)
    assert chunked == baseline
