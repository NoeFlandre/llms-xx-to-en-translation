import json
from pathlib import Path

import pytest

from tests.fakes import FakeBackend
from xx_to_en.benchmarks import Segment
from xx_to_en.cache import OutputCache
from xx_to_en.config import ModelConfig
from xx_to_en.inventory import Inventory
from xx_to_en.languages import LanguageMap
from xx_to_en.prompts import make_prompt, template_names
from xx_to_en.runner import (
    Decoding,
    Translation,
    WorkItem,
    failure_counts,
    plan_work,
    run_work_item,
    throughput,
    translate,
    write_jsonl,
)


def _model(prompt: str = "generic_chat") -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "name": "toy-model",
            "hf_repo": "org/toy-model",
            "family": "toy",
            "params_b": 1.0,
            "active_params_b": 1.0,
            "license": "apache-2.0",
            "access": "open",
            "backend": "vllm",
            "prompt": prompt,
            "thinking": "not_applicable",
            "dtype": "bfloat16",
            "tensor_parallel": 1,
            "gpu_memory_gb_min": 8,
            "trust_remote_code": False,
            "official_languages": ["fra_Latn"],
        }
    )


def _segment(index: int, source: str, reference: str = "ref") -> Segment:
    return Segment("flores_plus_devtest", "fra_Latn", "fra_Latn", str(index), source, reference)


def _inventory() -> Inventory:
    return Inventory(
        corpus_rows={"deu_Latn": 10, "fra_Latn": 5, "ceb_Latn": 2, "eng_Latn": 99},
        benchmark_codes={
            "flores_plus_devtest": frozenset({"deu_Latn", "fra_Latn"}),
            "tatoeba_xen": frozenset({"ceb"}),
        },
    )


def test_plan_respects_model_support_and_availability():
    items = plan_work(
        ["flores_plus_devtest", "tatoeba_xen"],
        ["deu_Latn", "fra_Latn", "ceb_Latn"],
        _inventory(),
        LanguageMap({}),
        supported=frozenset({"fra_Latn", "ceb_Latn"}),
    )
    assert items == [
        WorkItem("flores_plus_devtest", "fra_Latn", "fra_Latn"),
        WorkItem("tatoeba_xen", "ceb_Latn", "ceb"),
    ]


def test_plan_skips_languages_with_empty_mapping():
    mapping = LanguageMap({"deu_Latn": {"flores_plus_devtest": ""}})
    items = plan_work(
        ["flores_plus_devtest"], ["deu_Latn"], _inventory(), mapping, supported=frozenset({"deu_Latn"})
    )
    assert items == []


def test_plan_without_restriction_includes_everything_available():
    items = plan_work(["flores_plus_devtest"], ["deu_Latn", "fra_Latn"], _inventory(), LanguageMap({}))
    assert {item.corpus_lang for item in items} == {"deu_Latn", "fra_Latn"}


def test_translate_uses_cache_on_second_pass(tmp_path: Path):
    backend = FakeBackend()
    cache = OutputCache(tmp_path / "cache")
    segments = [_segment(0, "bonjour le monde"), _segment(1, "le chat dort")]
    first = translate(segments, model=_model(), backend=backend, cache=cache)
    second = translate(segments, model=_model(), backend=backend, cache=cache)
    assert [t.hypothesis for t in first] == ["BONJOUR LE MONDE", "LE CHAT DORT"]
    assert not any(t.cached for t in first)
    assert all(t.cached for t in second)
    assert [t.hypothesis for t in second] == [t.hypothesis for t in first]
    assert backend.generate_calls == [2]


def test_cache_separates_different_seeds(tmp_path: Path):
    backend = FakeBackend()
    cache = OutputCache(tmp_path)
    segments = [_segment(0, "un deux trois")]
    translate(segments, model=_model(), backend=backend, cache=cache, decoding=Decoding(seed=0))
    again = translate(segments, model=_model(), backend=backend, cache=cache, decoding=Decoding(seed=1))
    assert not again[0].cached


def test_budget_is_capped_by_ceiling(tmp_path: Path):
    backend = FakeBackend()
    long_source = " ".join(["word"] * 50)
    translations = translate(
        [_segment(0, long_source)],
        model=_model(),
        backend=backend,
        cache=OutputCache(tmp_path),
        decoding=Decoding(ratio=10.0, floor=4, ceiling=20),
    )
    assert translations[0].output_tokens <= 20


def test_write_jsonl_round_trip(tmp_path: Path):
    segment = _segment(7, "source", "reference")
    translation = Translation(segment, "hypothesis", output_tokens=3, cached=False)
    path = tmp_path / "nested" / "out.jsonl"
    write_jsonl(path, [translation])
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["segment_id"] == "7"
    assert record["hypothesis"] == "hypothesis"
    assert record["reference"] == "reference"
    assert record["cached"] is False


def test_failure_counts_flag_empty_and_copy_and_loops():
    segments = [_segment(0, "Le chat."), _segment(1, "Un chien."), _segment(2, "Oui.")]
    translations = [
        Translation(segments[0], "", 0, False),
        Translation(segments[1], "un chien.", 2, False),
        Translation(segments[2], "yes yes yes yes yes", 5, False),
    ]
    counts = failure_counts(translations)
    assert counts["empty"] == 1
    assert counts["untranslated_copy"] == 1
    assert counts["repetition_loop"] == 1


def test_throughput_handles_zero_time():
    assert throughput([], 0.0)["output_tokens_per_second"] == 0.0
    translation = Translation(_segment(0, "a"), "A", 10, False)
    assert throughput([translation], 2.0)["output_tokens_per_second"] == pytest.approx(5.0)


def test_run_work_item_writes_file_and_summary(tmp_path: Path):
    item = WorkItem("flores_plus_devtest", "fra_Latn", "fra_Latn")
    summary = run_work_item(
        item,
        [_segment(0, "bonjour le monde")],
        model=_model(),
        backend=FakeBackend(),
        cache=OutputCache(tmp_path / "cache"),
        out_dir=tmp_path / "out",
    )
    assert Path(str(summary["path"])).exists()
    assert summary["segments"] == 1.0
    assert summary["failures"] == {"untranslated_copy": 1}


def test_prompt_template_registry():
    assert "generic_chat" in template_names()
    backend = FakeBackend()
    prompt = make_prompt("generic_chat", _segment(0, "bonjour"), backend)
    assert "Translate the following text into English" in prompt.text
    with pytest.raises(ValueError, match="no prompt template"):
        make_prompt("missing", _segment(0, "bonjour"), backend)
