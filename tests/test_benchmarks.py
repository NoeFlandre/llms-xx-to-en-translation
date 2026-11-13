import pytest

from xx_to_en.benchmarks import pair_by_position, pair_flores, pair_tatoeba, pair_wmt


def _jsonl(*records: dict) -> list[str]:
    import json

    return [json.dumps(record) for record in records]


def test_flores_aligns_on_id_and_skips_missing_ids():
    source = _jsonl({"id": 2, "text": "deux"}, {"id": 1, "text": "un"}, {"id": 3, "text": ""})
    reference = _jsonl({"id": 1, "text": "one"}, {"id": 2, "text": "two"}, {"id": 3, "text": "three"})
    segments = pair_flores(
        source, reference, benchmark="flores_plus_devtest", corpus_lang="fra_Latn", bench_lang="fra_Latn"
    )
    assert [(s.segment_id, s.source, s.reference) for s in segments] == [
        ("1", "un", "one"),
        ("2", "deux", "two"),
    ]
    assert all(s.corpus_lang == "fra_Latn" and s.benchmark == "flores_plus_devtest" for s in segments)


def test_position_alignment_requires_equal_lengths():
    with pytest.raises(ValueError, match="sources vs"):
        pair_by_position(
            ["a"], ["x", "y"], benchmark="ntrex_128", corpus_lang="fra_Latn", bench_lang="fra_Latn"
        )


def test_position_alignment_keeps_order_and_ids():
    segments = pair_by_position(
        ["a", "b"], ["x", "y"], benchmark="ntrex_128", corpus_lang="fra_Latn", bench_lang="fra_Latn"
    )
    assert [(s.segment_id, s.source, s.reference) for s in segments] == [("0", "a", "x"), ("1", "b", "y")]


def test_wmt_reads_source_and_english_reference_from_config_key():
    rows = [{"cs-en": {"cs": "Ahoj.", "en": "Hello."}}, {"cs-en": {"cs": "Dobrý den.", "en": "Good day."}}]
    segments = pair_wmt(rows, config="cs-en", benchmark="wmt22_xen", corpus_lang="ces_Latn", bench_lang="cs")
    assert [(s.source, s.reference) for s in segments] == [("Ahoj.", "Hello."), ("Dobrý den.", "Good day.")]


def test_tatoeba_keeps_orientation_for_x_to_english_files():
    lines = [
        "sourceLang\ttargetLang\tsourceString\ttargetString",
        "deu\teng\tHallo Welt\tHello world",
    ]
    segments = pair_tatoeba(lines, benchmark="tatoeba_xen", corpus_lang="deu_Latn", bench_lang="deu")
    assert [(s.source, s.reference) for s in segments] == [("Hallo Welt", "Hello world")]


def test_tatoeba_swaps_columns_for_eng_first_files_and_skips_empty_rows():
    lines = [
        "sourceLang\ttargetLang\tsourceString\ttargetString",
        "eng\tfra\tGood morning.\tBonjour.",
        "eng\tfra\tEmpty target.\t",
    ]
    segments = pair_tatoeba(lines, benchmark="tatoeba_xen", corpus_lang="fra_Latn", bench_lang="fra")
    assert [(s.source, s.reference) for s in segments] == [("Bonjour.", "Good morning.")]
