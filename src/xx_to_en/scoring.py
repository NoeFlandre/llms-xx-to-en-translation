"""Metric inputs and score merging for translation records.

Pure functions only. Model calls live in ``xx_to_en.metric_runtime`` and run on Grid5000 nodes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from xx_to_en.failures import detect

ENGLISH = "eng_Latn"
GLOTLID_PREFIX = "__label__"

# ISO 639-1 codes of the XLM-R pretraining languages (inherited by COMET-22) and of the mT5
# languages (inherited by MetricX-24), taken as their intersection. Approximate: check against the
# XLM-R and mT5 papers before publishing. Languages outside this set get chrF++ only.
NEURAL_ISO_639_1: frozenset[str] = frozenset(
    {
        "af",
        "am",
        "ar",
        "as",
        "az",
        "be",
        "bg",
        "bn",
        "br",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "eo",
        "es",
        "et",
        "eu",
        "fa",
        "ff",
        "fi",
        "fr",
        "fy",
        "ga",
        "gd",
        "gl",
        "gu",
        "ha",
        "he",
        "hi",
        "hr",
        "hu",
        "hy",
        "id",
        "ig",
        "is",
        "it",
        "ja",
        "jv",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "ku",
        "ky",
        "la",
        "lo",
        "lt",
        "lv",
        "mg",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "my",
        "ne",
        "nl",
        "no",
        "om",
        "or",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sa",
        "sd",
        "si",
        "sk",
        "sl",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "th",
        "tl",
        "tr",
        "ug",
        "uk",
        "ur",
        "uz",
        "vi",
        "xh",
        "yi",
        "zh",
    }
)


def is_neural_covered(iso_639_1: str | None) -> bool:
    """True when COMET-22 and MetricX-24 are expected to be meaningful for this source language."""
    return iso_639_1 is not None and iso_639_1 in NEURAL_ISO_639_1


def comet_inputs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """COMET input rows: source, hypothesis and reference."""
    return [
        {"src": str(record["source"]), "mt": str(record["hypothesis"]), "ref": str(record["reference"])}
        for record in records
    ]


def metricx_inputs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """MetricX input rows, with the field names the upstream predict module reads."""
    return [
        {
            "source": str(record["source"]),
            "hypothesis": str(record["hypothesis"]),
            "reference": str(record["reference"]),
        }
        for record in records
    ]


def parse_metricx_output(lines: Iterable[str]) -> list[float]:
    """Scores from MetricX predict output, one JSON object per line with a ``prediction`` field."""
    return [float(json.loads(line)["prediction"]) for line in lines if line.strip()]


def metricx_text(row: Mapping[str, str]) -> str:
    """MetricX-24 reference-based input string, built exactly as the upstream predict module builds it."""
    return "source: " + row["source"] + " candidate: " + row["hypothesis"] + " reference: " + row["reference"]


def pad_sequences(sequences: Sequence[Sequence[int]], pad_id: int) -> tuple[list[list[int]], list[list[int]]]:
    """Right-pad token id sequences to a common length; return the padded ids and attention masks."""
    width = max((len(sequence) for sequence in sequences), default=0)
    ids = [list(sequence) + [pad_id] * (width - len(sequence)) for sequence in sequences]
    masks = [[1] * len(sequence) + [0] * (width - len(sequence)) for sequence in sequences]
    return ids, masks


def glotlid_code(raw_label: str) -> str:
    """Strip the fastText label prefix: ``__label__fra_Latn`` becomes ``fra_Latn``."""
    return raw_label.removeprefix(GLOTLID_PREFIX)


def scatter(values: Sequence[float], mask: Sequence[bool]) -> list[float | None]:
    """Place ``values`` at the True positions of ``mask``; other positions get None."""
    if len(values) != sum(mask):
        raise ValueError(f"{len(values)} values for {sum(mask)} selected positions")
    iterator = iter(values)
    return [next(iterator) if selected else None for selected in mask]


def merge_scores(
    records: Sequence[Mapping[str, Any]],
    *,
    chrf: Sequence[float],
    comet: Sequence[float | None],
    metricx: Sequence[float | None],
    codes: Sequence[str],
    neural: Sequence[bool],
) -> list[dict[str, Any]]:
    """Add scores and failure flags to each record. Every sequence must match the record count."""
    counts = {len(records), len(chrf), len(comet), len(metricx), len(codes), len(neural)}
    if len(counts) != 1:
        raise ValueError("all score sequences must match the record count")
    merged = []
    for record, chrf_score, comet_score, metricx_score, code, covered in zip(
        records, chrf, comet, metricx, codes, neural, strict=True
    ):
        hypothesis = str(record["hypothesis"])
        flags = detect(str(record["source"]), hypothesis)
        merged.append(
            {
                **record,
                "chrf_pp": chrf_score,
                "comet22": comet_score,
                "metricx24": metricx_score,
                "detected_lang": code,
                "neural_covered": covered,
                "wrong_language": bool(hypothesis.strip()) and code != ENGLISH,
                "empty": flags.empty,
                "repetition_loop": flags.repetition_loop,
                "length_ratio_outlier": flags.length_ratio_outlier,
                "untranslated_copy": flags.untranslated_copy,
            }
        )
    return merged
