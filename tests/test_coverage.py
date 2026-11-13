from pathlib import Path

from xx_to_en.config import load_language_map
from xx_to_en.coverage import (
    STATUS_ENGLISH,
    STATUS_NONE,
    STATUS_TATOEBA,
    STATUS_WMT,
    build_coverage,
    render_markdown,
    summarize,
)
from xx_to_en.inventory import Inventory, load_inventory
from xx_to_en.languages import LanguageMap

ROOT = Path(__file__).resolve().parents[1]


def _inventory() -> Inventory:
    return Inventory(
        corpus_rows={
            "eng_Latn": 1000,
            "deu_Latn": 100,
            "cmn_Hani": 50,
            "fra_Latn": 30,
            "mar_Latn": 5,
            "ceb_Latn": 7,
            "wikipedia": 999,
        },
        benchmark_codes={
            "flores_plus_devtest": frozenset({"fra_Latn", "cmn_Hans"}),
            "ntrex_128": frozenset({"fra_Latn"}),
            "wmt22_xen": frozenset({"de"}),
            "wmt23_xen": frozenset({"de", "zh"}),
            "tatoeba_xen": frozenset({"deu", "fra", "zho"}),
        },
    )


def _rows():
    mapping = LanguageMap(
        {
            "cmn_Hani": {"flores_plus_devtest": "cmn_Hans", "wmt23_xen": "zh"},
            "deu_Latn": {"wmt22_xen": "de", "wmt23_xen": "de"},
        }
    )
    return build_coverage(_inventory(), mapping)


def test_rows_cover_only_non_english_language_configs():
    codes = [row.code for row in _rows()]
    assert codes == ["ceb_Latn", "cmn_Hani", "deu_Latn", "fra_Latn", "mar_Latn"]


def test_status_priority_wmt_then_tatoeba_then_english_origin():
    statuses = {row.code: row.status for row in _rows()}
    assert statuses["deu_Latn"] == STATUS_WMT
    assert statuses["cmn_Hani"] == STATUS_WMT
    assert statuses["fra_Latn"] == STATUS_TATOEBA
    assert statuses["ceb_Latn"] == STATUS_NONE


def test_english_origin_only_when_no_native_benchmark_covers_it():
    inventory = _inventory()
    rows = build_coverage(inventory, LanguageMap({}))
    by_code = {row.code: row for row in rows}
    assert by_code["mar_Latn"].status == STATUS_NONE
    inventory.benchmark_codes["ntrex_128"] = frozenset({"mar_Latn"})
    rows = build_coverage(inventory, LanguageMap({}))
    assert {row.code: row for row in rows}["mar_Latn"].status == STATUS_ENGLISH


def test_empty_mapping_entry_blocks_match():
    mapping = LanguageMap({"mar_Latn": {"ntrex_128": ""}})
    rows = build_coverage(_inventory(), mapping, benchmarks=("ntrex_128",))
    assert {row.code: row for row in rows}["mar_Latn"].covered == frozenset()


def test_summary_counts_rows_and_native_share():
    summary = summarize(_rows())
    assert summary.configs == 5
    assert summary.total_rows == 7 + 50 + 100 + 30 + 5
    assert summary.native_rows == 50 + 100 + 30
    assert summary.uncovered_rows == 7 + 5
    assert summary.status_counts[STATUS_WMT] == 2


def test_render_markdown_has_table_and_summary():
    text = render_markdown(_rows())
    assert text.startswith("# Language x benchmark coverage (X->EN)")
    assert "| deu_Latn | 100 | yes | yes | yes | yes | yes |" in text or "| deu_Latn | 100 |" in text
    assert "WMT24 is not used" in text
    assert "- Non-English configs: 5" in text


def test_render_handles_no_rows():
    text = render_markdown([])
    assert "Non-English configs: 0" in text


def test_committed_inventory_matches_documented_findings():
    inventory = load_inventory(ROOT / "data" / "benchmark_inventory.json")
    language_map = LanguageMap(load_language_map(ROOT / "configs" / "languages.yaml"))
    summary = summarize(build_coverage(inventory, language_map))
    assert summary.configs == 94
    assert summary.total_rows == 71_490_138
    assert summary.uncovered_rows == 351
    assert summary.status_counts[STATUS_WMT] == 8
