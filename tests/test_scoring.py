from pathlib import Path

import pytest

from xx_to_en.jsonl import read_records, write_records
from xx_to_en.scoring import (
    ENGLISH,
    NEURAL_ISO_639_1,
    comet_inputs,
    glotlid_code,
    is_neural_covered,
    merge_scores,
    metricx_inputs,
    pad_sequences,
    parse_metricx_output,
    scatter,
)


def _record(source="Bonjour le monde.", hypothesis="Hello world.", reference="Hello world."):
    return {
        "benchmark": "flores_plus_devtest",
        "corpus_lang": "fra_Latn",
        "bench_lang": "fra_Latn",
        "segment_id": "0",
        "source": source,
        "reference": reference,
        "hypothesis": hypothesis,
        "output_tokens": 3,
        "cached": False,
    }


def test_comet_inputs_use_comet_field_names():
    assert comet_inputs([_record()]) == [
        {"src": "Bonjour le monde.", "mt": "Hello world.", "ref": "Hello world."}
    ]


def test_metricx_inputs_use_metricx_field_names():
    assert metricx_inputs([_record()]) == [
        {"source": "Bonjour le monde.", "hypothesis": "Hello world.", "reference": "Hello world."}
    ]


def test_parse_metricx_output_reads_predictions_and_skips_blank_lines():
    lines = ['{"hypothesis": "a", "prediction": 1.5}', "", '{"prediction": 0.25}']
    assert parse_metricx_output(lines) == [1.5, 0.25]


def test_glotlid_code_strips_the_label_prefix_only_when_present():
    assert glotlid_code("__label__fra_Latn") == "fra_Latn"
    assert glotlid_code("eng_Latn") == "eng_Latn"


def test_scatter_places_values_at_selected_positions():
    assert scatter([1.0, 2.0], [True, False, True]) == [1.0, None, 2.0]


def test_scatter_rejects_length_mismatch():
    with pytest.raises(ValueError, match="selected positions"):
        scatter([1.0], [True, True])


def test_neural_coverage_by_iso_code():
    assert "fr" in NEURAL_ISO_639_1
    assert is_neural_covered("fr")
    assert not is_neural_covered("ti")
    assert not is_neural_covered(None)


def test_merge_flags_untranslated_output_and_keeps_record_fields():
    merged = merge_scores(
        [_record(hypothesis="Bonjour le monde.")],
        chrf=[10.0],
        comet=[0.5],
        metricx=[3.0],
        codes=["fra_Latn"],
        neural=[True],
    )
    row = merged[0]
    assert row["wrong_language"] is True
    assert row["untranslated_copy"] is True
    assert row["comet22"] == 0.5
    assert row["chrf_pp"] == 10.0
    assert row["segment_id"] == "0"


def test_merge_clean_english_output_has_no_flags():
    merged = merge_scores(
        [_record()],
        chrf=[100.0],
        comet=[0.9],
        metricx=[1.0],
        codes=[ENGLISH],
        neural=[True],
    )
    assert merged[0]["wrong_language"] is False
    assert merged[0]["empty"] is False
    assert merged[0]["untranslated_copy"] is False


def test_empty_output_is_flagged_empty_but_not_wrong_language():
    merged = merge_scores(
        [_record(hypothesis="   ")],
        chrf=[0.0],
        comet=[None],
        metricx=[None],
        codes=[""],
        neural=[False],
    )
    assert merged[0]["empty"] is True
    assert merged[0]["wrong_language"] is False


def test_merge_rejects_mismatched_sequence_lengths():
    with pytest.raises(ValueError, match="record count"):
        merge_scores(
            [_record()],
            chrf=[1.0, 2.0],
            comet=[None],
            metricx=[None],
            codes=[ENGLISH],
            neural=[False],
        )


def test_jsonl_round_trip(tmp_path: Path):
    path = tmp_path / "nested" / "rows.jsonl"
    rows = [_record(), _record(hypothesis="Bonjour")]
    write_records(path, rows)
    assert read_records(path) == rows


def test_pad_sequences_right_pads_and_masks_the_padding():
    ids, masks = pad_sequences([[5, 6, 7], [9]], pad_id=0)
    assert ids == [[5, 6, 7], [9, 0, 0]]
    assert masks == [[1, 1, 1], [1, 0, 0]]


def test_pad_sequences_on_empty_input():
    assert pad_sequences([], pad_id=0) == ([], [])
