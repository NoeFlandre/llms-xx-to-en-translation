from pathlib import Path

import pytest
from pydantic import ValidationError

from xx_to_en.cli import Project
from xx_to_en.config import ModelConfig
from xx_to_en.plan import LANGS_PER_SHARD, build_shards

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def project():
    return Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")


def test_split_shards_name_real_benchmarks(project):
    for model in project.models.values():
        assert set(model.split_shards) <= set(project.benchmarks), model.name


def test_split_shard_sizes_are_positive_integers_by_schema(project):
    base = project.models["nllb-200-3.3b"].model_dump()
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({**base, "split_shards": {"tatoeba_xen": 0}})


def test_default_sizes_name_real_benchmarks(project):
    assert set(LANGS_PER_SHARD) <= set(project.benchmarks)


def test_every_enabled_model_plans_each_work_item_exactly_once(project):
    for name, model in project.models.items():
        if not model.enabled:
            continue
        items = project.plan(model)
        shards = build_shards(name, items, splits=model.split_shards)
        covered = sorted((shard.benchmark, lang) for shard in shards for lang in shard.langs)
        expected = sorted((item.benchmark, item.corpus_lang) for item in items)
        assert covered == expected, name
        assert len({shard.shard_id for shard in shards}) == len(shards), name
