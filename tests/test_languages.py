import pytest

from xx_to_en.languages import LanguageMap, corpus_languages, is_corpus_language


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fra_Latn", True),
        ("cmn_Hani", True),
        ("eng_Latn", True),
        ("wikipedia", False),
        ("all", False),
        ("fra_latn", False),
        ("fr_Latn", False),
    ],
)
def test_is_corpus_language(name, expected):
    assert is_corpus_language(name) is expected


def test_corpus_languages_excludes_english_and_auxiliary_configs():
    rows = {"eng_Latn": 10, "fra_Latn": 5, "wikipedia": 7, "all": 99, "srp_Latn": 2}
    assert corpus_languages(rows) == {"fra_Latn": 5, "srp_Latn": 2}


def test_identity_is_the_default_for_full_codes():
    mapping = LanguageMap({})
    assert mapping.benchmark_code("fra_Latn", "flores_plus_devtest") == "fra_Latn"


def test_tatoeba_defaults_to_iso_639_3_prefix():
    mapping = LanguageMap({})
    assert mapping.benchmark_code("deu_Latn", "tatoeba_xen") == "deu"


def test_explicit_mapping_overrides_default():
    mapping = LanguageMap({"cmn_Hani": {"tatoeba_xen": "zho", "flores_plus_devtest": "cmn_Hans"}})
    assert mapping.benchmark_code("cmn_Hani", "tatoeba_xen") == "zho"
    assert mapping.benchmark_code("cmn_Hani", "flores_plus_devtest") == "cmn_Hans"


def test_empty_string_means_no_equivalent():
    mapping = LanguageMap({"mar_Latn": {"flores_plus_devtest": ""}})
    assert mapping.benchmark_code("mar_Latn", "flores_plus_devtest") == ""
