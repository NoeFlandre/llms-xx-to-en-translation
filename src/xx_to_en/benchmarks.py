"""Turn raw benchmark records into X->EN segments.

Pure functions only: they take records that were already read from disk or from the Hub.
Fetching lives in ``xx_to_en.sources``. Every segment keeps its benchmark name, corpus code
and benchmark code so outputs can be joined back to the corpus language table.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ENGLISH_BENCHMARK = "eng_Latn"
TATOEBA_ENGLISH = "eng"
WMT_ENGLISH = "en"


@dataclass(frozen=True)
class Segment:
    benchmark: str
    corpus_lang: str
    bench_lang: str
    segment_id: str
    source: str
    reference: str


def pair_flores(
    source_lines: Iterable[str],
    reference_lines: Iterable[str],
    *,
    benchmark: str,
    corpus_lang: str,
    bench_lang: str,
) -> list[Segment]:
    """FLORES+ devtest: JSON lines with ``id`` and ``text``; aligned on ``id``."""
    source = {str(record["id"]): record["text"] for record in map(json.loads, source_lines) if record["text"]}
    reference = {str(record["id"]): record["text"] for record in map(json.loads, reference_lines)}
    shared = sorted(set(source) & set(reference), key=int)
    return [
        Segment(benchmark, corpus_lang, bench_lang, segment_id, source[segment_id], reference[segment_id])
        for segment_id in shared
    ]


def pair_by_position(
    source_texts: Sequence[str],
    reference_texts: Sequence[str],
    *,
    benchmark: str,
    corpus_lang: str,
    bench_lang: str,
) -> list[Segment]:
    """Benchmarks whose sources and references are row-aligned (NTREX-128)."""
    if len(source_texts) != len(reference_texts):
        raise ValueError(
            f"{benchmark} {bench_lang}: {len(source_texts)} sources vs {len(reference_texts)} references"
        )
    return [
        Segment(benchmark, corpus_lang, bench_lang, str(index), source, reference)
        for index, (source, reference) in enumerate(zip(source_texts, reference_texts, strict=True))
    ]


def pair_wmt(
    rows: Iterable[Mapping[str, Any]], *, config: str, benchmark: str, corpus_lang: str, bench_lang: str
) -> list[Segment]:
    """WMT rows shaped ``{"xx-en": {"xx": source, "en": reference}}``."""
    segments = []
    for index, row in enumerate(rows):
        pair = row[config]
        segments.append(
            Segment(benchmark, corpus_lang, bench_lang, str(index), pair[bench_lang], pair[WMT_ENGLISH])
        )
    return segments


def pair_tatoeba(lines: Iterable[str], *, benchmark: str, corpus_lang: str, bench_lang: str) -> list[Segment]:
    """Tatoeba test TSV with columns sourceLang, targetLang, sourceString, targetString.

    Files are stored as ``eng-X`` or ``X-eng``; rows are oriented so the non-English side is the source.
    """
    reader = csv.DictReader(lines, delimiter="\t")
    segments = []
    for index, row in enumerate(reader):
        if row["sourceLang"] == TATOEBA_ENGLISH:
            source, reference = row["targetString"], row["sourceString"]
        else:
            source, reference = row["sourceString"], row["targetString"]
        if source and reference:
            segments.append(Segment(benchmark, corpus_lang, bench_lang, str(index), source, reference))
    return segments
