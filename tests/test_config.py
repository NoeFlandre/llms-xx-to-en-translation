from pathlib import Path

import pytest

from xx_to_en.config import load_benchmarks, load_language_map, load_models

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_all_model_configs_validate():
    models = load_models(CONFIGS / "models")
    assert len(models) == 17
    assert models["gemma-4-31b-it"].hf_repo == "google/gemma-4-31B-it"
    assert models["hy-mt2-30b-a3b"].active_params_b == 3.0


def test_all_benchmark_configs_validate():
    benchmarks = load_benchmarks(CONFIGS / "benchmarks")
    assert set(benchmarks) == {
        "flores_plus_devtest",
        "ntrex_128",
        "wmt22_xen",
        "wmt23_xen",
        "tatoeba_xen",
    }
    assert benchmarks["wmt22_xen"].origin == "native"
    assert benchmarks["ntrex_128"].origin == "english_translated"


def test_no_wmt24_benchmark_is_configured():
    assert not any("24" in name for name in load_benchmarks(CONFIGS / "benchmarks"))


def test_language_map_loads_and_empty_means_no_equivalent():
    mapping = load_language_map(CONFIGS / "languages.yaml")
    assert mapping["cmn_Hani"]["tatoeba_xen"] == "zho"
    assert mapping["uzn_Cyrl"]["flores_plus_devtest"] == ""


def test_name_must_match_file_stem(tmp_path: Path):
    source = (CONFIGS / "models" / "lfm3-1.7b.yaml").read_text(encoding="utf-8")
    (tmp_path / "other-name.yaml").write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="must match the file name"):
        load_models(tmp_path)


def test_unknown_key_is_rejected(tmp_path: Path):
    source = (CONFIGS / "models" / "lfm3-1.7b.yaml").read_text(encoding="utf-8")
    (tmp_path / "lfm3-1.7b.yaml").write_text(source + "surprise: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_models(tmp_path)


def test_top_level_must_be_mapping(tmp_path: Path):
    (tmp_path / "bad.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_benchmarks(tmp_path)


def test_language_map_values_must_be_strings(tmp_path: Path):
    path = tmp_path / "languages.yaml"
    path.write_text("fra_Latn:\n  flores_plus_devtest: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strings"):
        load_language_map(path)
