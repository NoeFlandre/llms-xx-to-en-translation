"""MetricX-24 batching and loading, driven by fake torch, tokenizer and model."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from xx_to_en import metric_runtime
from xx_to_en.metric_runtime import (
    COMET_MODEL,
    GLOTLID_FILE,
    GLOTLID_REPO,
    Scorers,
    _length_batches,
    _metricx_rows,
)
from xx_to_en.scoring import metricx_text


class _FakeTensor:
    def __init__(self, data: Any, device: Any = None) -> None:
        self.data = data
        self.device = device


class _Predictions:
    """Stands in for the output tensor; ``float`` and ``cpu`` return the same object, as they do in torch."""

    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def float(self) -> _Predictions:
        return self

    def cpu(self) -> _Predictions:
        return self

    def tolist(self) -> list[Any]:
        return list(self._values)


class _FakeMetricXModel:
    """Predicts each row's unpadded length, so every score identifies the row it was computed for."""

    def __init__(self) -> None:
        self.batch_lengths: list[list[int]] = []
        self.use_cache: list[bool] = []

    def __call__(self, *, input_ids: _FakeTensor, attention_mask: _FakeTensor, use_cache: bool) -> Any:
        lengths = [sum(mask) for mask in attention_mask.data]
        self.batch_lengths.append(lengths)
        self.use_cache.append(use_cache)
        return SimpleNamespace(predictions=_Predictions([float(length) for length in lengths]))


class _FakeMetricXTokenizer:
    """One id per whitespace word, then an end-of-sequence id that the caller drops."""

    pad_token_id = 0

    def __call__(
        self, text: str, *, max_length: int, truncation: bool, padding: bool
    ) -> dict[str, list[int]]:
        assert truncation is True
        assert padding is False
        ids = [len(word) for word in text.split()][:max_length]
        return {"input_ids": [*ids, 1]}


_FAKE_TORCH = SimpleNamespace(
    no_grad=nullcontext, tensor=_FakeTensor, cuda=SimpleNamespace(empty_cache=lambda: None)
)


def _row(extra_words: int) -> dict[str, str]:
    source = " ".join(["w"] * (extra_words + 1))
    return {"source": source, "hypothesis": "h", "reference": "r"}


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch) -> list[_FakeMetricXModel]:
    """Replaces the MetricX loader and the torch module. Returns the model of each load, in order."""
    models: list[_FakeMetricXModel] = []

    def fake_load(repo: Path) -> tuple[Any, Any, str]:
        model = _FakeMetricXModel()
        models.append(model)
        return _FakeMetricXTokenizer(), model, "cpu"

    monkeypatch.setattr(metric_runtime, "_load_metricx", fake_load)
    monkeypatch.setitem(sys.modules, "torch", _FAKE_TORCH)
    return models


def _scorers(batch: int = 32, comet_batch: int = 64) -> Scorers:
    return Scorers(
        metricx_repo=Path("unused"),
        work=Path("unused"),
        metricx_batch=batch,
        comet_batch=comet_batch,
        max_input_length=99,
    )


def test_metricx_scores_follow_input_order_across_length_batches(loaded: list[_FakeMetricXModel]):
    inputs = [_row(extra) for extra in (3, 0, 2, 5, 1)]

    scores = _scorers(batch=2).metricx(inputs)

    assert scores == [float(len(metricx_text(row).split())) for row in inputs]


def test_metricx_batches_are_sorted_by_length_then_cut(loaded: list[_FakeMetricXModel]):
    inputs = [_row(extra) for extra in (3, 0, 2, 5, 1)]

    _scorers(batch=2).metricx(inputs)

    lengths = sorted(len(metricx_text(row).split()) for row in inputs)
    assert loaded[0].batch_lengths == [lengths[0:2], lengths[2:4], lengths[4:5]]
    assert loaded[0].use_cache == [False, False, False]


def test_metricx_with_no_inputs_loads_nothing(loaded: list[_FakeMetricXModel]):
    assert _scorers().metricx([]) == []
    assert loaded == []


def test_metricx_loads_the_model_once_and_reuses_it(loaded: list[_FakeMetricXModel]):
    scorers = _scorers()
    scorers.metricx([_row(0)])
    scorers.metricx([_row(1)])
    assert len(loaded) == 1


class _SmallGpuMetricXModel(_FakeMetricXModel):
    """Runs out of memory for any batch of more than ``limit`` rows, as a busy GPU does."""

    def __init__(self, limit: int, message: str = "CUDA out of memory. Tried to allocate 384 MiB") -> None:
        super().__init__()
        self.limit = limit
        self.message = message

    def __call__(self, *, input_ids: _FakeTensor, attention_mask: _FakeTensor, use_cache: bool) -> Any:
        if len(attention_mask.data) > self.limit:
            raise RuntimeError(self.message)
        return super().__call__(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)


def _use_model(monkeypatch: pytest.MonkeyPatch, model: _FakeMetricXModel) -> None:
    monkeypatch.setattr(metric_runtime, "_load_metricx", lambda repo: (_FakeMetricXTokenizer(), model, "cpu"))
    monkeypatch.setitem(sys.modules, "torch", _FAKE_TORCH)


def test_metricx_splits_a_batch_that_runs_out_of_memory(monkeypatch: pytest.MonkeyPatch):
    model = _SmallGpuMetricXModel(limit=1)
    _use_model(monkeypatch, model)
    inputs = [_row(extra) for extra in (3, 0, 2, 5)]

    scores = _scorers(batch=4).metricx(inputs)

    assert scores == [float(len(metricx_text(row).split())) for row in inputs]
    assert model.batch_lengths
    assert all(len(batch) == 1 for batch in model.batch_lengths)


def test_metricx_reraises_out_of_memory_for_a_single_row(monkeypatch: pytest.MonkeyPatch):
    _use_model(monkeypatch, _SmallGpuMetricXModel(limit=0))
    with pytest.raises(RuntimeError, match="out of memory"):
        _scorers(batch=1).metricx([_row(0)])


def test_metricx_reraises_errors_that_are_not_out_of_memory(monkeypatch: pytest.MonkeyPatch):
    _use_model(monkeypatch, _SmallGpuMetricXModel(limit=0, message="shape mismatch"))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _scorers(batch=4).metricx([_row(0), _row(1)])


def test_length_batches_order_rows_by_length_before_cutting():
    assert _length_batches([[1, 2, 3], [1], [1, 2]], 2) == [[1, 2], [0]]


def test_metricx_rows_drop_the_final_token_and_respect_max_length():
    inputs = [{"source": "a b c d", "hypothesis": "x", "reference": "y"}]
    rows = _metricx_rows(inputs, _FakeMetricXTokenizer(), 2)
    assert [len(row) for row in rows] == [2]


class _FakeComet:
    """Stands in for a loaded COMET checkpoint: scores each sample by its hypothesis length (an int)."""

    def __init__(self) -> None:
        self.predict_calls: list[dict[str, Any]] = []

    def predict(
        self, samples: list[dict[str, str]], *, batch_size: int, gpus: int, progress_bar: bool
    ) -> Any:
        self.predict_calls.append(
            {"samples": samples, "batch_size": batch_size, "gpus": gpus, "progress_bar": progress_bar}
        )
        return SimpleNamespace(scores=[len(sample["mt"]) for sample in samples])


@pytest.fixture
def comet(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replaces the comet package. Returns the checkpoint paths loaded and the models, in load order."""
    checkpoints: list[str] = []
    models: list[_FakeComet] = []

    def download_model(name: str) -> str:
        return f"/cache/{name}"

    def load_from_checkpoint(path: str) -> _FakeComet:
        checkpoints.append(path)
        model = _FakeComet()
        models.append(model)
        return model

    monkeypatch.setitem(
        sys.modules,
        "comet",
        SimpleNamespace(download_model=download_model, load_from_checkpoint=load_from_checkpoint),
    )
    return SimpleNamespace(checkpoints=checkpoints, models=models)


def test_comet_with_no_inputs_loads_nothing(comet: SimpleNamespace):
    assert _scorers().comet([]) == []
    assert comet.models == []


def test_comet_scores_follow_input_order_and_are_floats(comet: SimpleNamespace):
    inputs = [{"src": "a", "mt": "xyz", "ref": "r"}, {"src": "b", "mt": "x", "ref": "r"}]

    scores = _scorers().comet(inputs)

    assert scores == [3.0, 1.0]
    assert all(isinstance(score, float) for score in scores)


def test_comet_predicts_with_the_configured_batch_on_one_gpu_without_progress_bar(comet: SimpleNamespace):
    inputs = [{"src": "a", "mt": "b", "ref": "c"}]

    _scorers(comet_batch=7).comet(inputs)

    assert comet.models[0].predict_calls == [
        {"samples": inputs, "batch_size": 7, "gpus": 1, "progress_bar": False}
    ]


def test_comet_loads_the_checkpoint_once_and_reuses_it(comet: SimpleNamespace):
    scorers = _scorers()
    scorers.comet([{"src": "a", "mt": "b", "ref": "c"}])
    scorers.comet([{"src": "a", "mt": "bb", "ref": "c"}])
    assert comet.checkpoints == [f"/cache/{COMET_MODEL}"]
    assert len(comet.models[0].predict_calls) == 2


class _FakeFastText:
    """Stands in for a fastText model: labels each exact text with the label given for it."""

    def __init__(self, labels: dict[str, str]) -> None:
        self._labels = labels
        self.texts: list[str] = []

    def predict(self, text: str) -> tuple[list[str], list[float]]:
        self.texts.append(text)
        return [f"__label__{self._labels[text]}"], [0.99]


GLOTLID_LABELS = {"Hello there.": "eng_Latn", "Bonjour le monde.": "fra_Latn"}


@pytest.fixture
def glotlid(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replaces fastText and the Hub download. Returns the downloads and the models, in load order."""
    downloads: list[tuple[str, str]] = []
    models: list[_FakeFastText] = []

    def hf_hub_download(*, repo_id: str, filename: str) -> str:
        downloads.append((repo_id, filename))
        return f"/cache/{filename}"

    def load_model(path: str) -> _FakeFastText:
        model = _FakeFastText(GLOTLID_LABELS)
        models.append(model)
        return model

    monkeypatch.setitem(sys.modules, "fasttext", SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=hf_hub_download))
    return SimpleNamespace(downloads=downloads, models=models)


def test_glotlid_flattens_each_text_and_returns_its_top_label(glotlid: SimpleNamespace):
    codes = _scorers().glotlid(["Hello\nthere.", "  Bonjour le monde.\n"])

    assert codes == ["eng_Latn", "fra_Latn"]
    assert glotlid.models[0].texts == ["Hello there.", "Bonjour le monde."]


def test_glotlid_blank_texts_get_empty_codes_without_prediction(glotlid: SimpleNamespace):
    codes = _scorers().glotlid(["", " \n "])

    assert codes == ["", ""]
    assert glotlid.models[0].texts == []


def test_glotlid_mixes_blank_and_predicted_texts_in_input_order(glotlid: SimpleNamespace):
    codes = _scorers().glotlid(["", "Hello there."])

    assert codes == ["", "eng_Latn"]
    assert glotlid.models[0].texts == ["Hello there."]


def test_glotlid_loads_the_model_once_from_the_hub_and_reuses_it(glotlid: SimpleNamespace):
    scorers = _scorers()
    scorers.glotlid(["Hello there."])
    scorers.glotlid(["Bonjour le monde."])
    assert glotlid.downloads == [(GLOTLID_REPO, GLOTLID_FILE)]
    assert len(glotlid.models) == 1
