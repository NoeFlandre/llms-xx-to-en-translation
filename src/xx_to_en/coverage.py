"""Language x benchmark coverage for X->EN."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from xx_to_en.inventory import Inventory
from xx_to_en.languages import LanguageMap, corpus_languages

BENCHMARKS: tuple[str, ...] = ("flores_plus_devtest", "ntrex_128", "wmt22_xen", "wmt23_xen", "tatoeba_xen")
HEADERS: dict[str, str] = {
    "flores_plus_devtest": "FLORES+",
    "ntrex_128": "NTREX-128",
    "wmt22_xen": "WMT22 X-EN",
    "wmt23_xen": "WMT23 X-EN",
    "tatoeba_xen": "Tatoeba X-EN",
}
WMT = frozenset({"wmt22_xen", "wmt23_xen"})
ENGLISH_ORIGIN = frozenset({"flores_plus_devtest", "ntrex_128"})
NATIVE = WMT | {"tatoeba_xen"}
STATUS_WMT = "WMT (native)"
STATUS_TATOEBA = "Tatoeba only (mixed origin)"
STATUS_ENGLISH = "English-origin only"
STATUS_NONE = "**uncovered**"


@dataclass(frozen=True)
class CoverageRow:
    code: str
    rows: int
    covered: frozenset[str]

    @property
    def status(self) -> str:
        if self.covered & WMT:
            return STATUS_WMT
        if "tatoeba_xen" in self.covered:
            return STATUS_TATOEBA
        if self.covered & ENGLISH_ORIGIN:
            return STATUS_ENGLISH
        return STATUS_NONE

    @property
    def native(self) -> bool:
        return bool(self.covered & NATIVE)


@dataclass(frozen=True)
class CoverageSummary:
    configs: int
    total_rows: int
    native_rows: int
    uncovered_rows: int
    status_counts: dict[str, int]


def _is_covered(code: str, benchmark: str, inventory: Inventory, language_map: LanguageMap) -> bool:
    wanted = language_map.benchmark_code(code, benchmark)
    return bool(wanted) and wanted in inventory.available(benchmark)


def build_coverage(
    inventory: Inventory, language_map: LanguageMap, benchmarks: Sequence[str] = BENCHMARKS
) -> list[CoverageRow]:
    """One row per non-English corpus language, sorted by code."""
    rows: list[CoverageRow] = []
    for code, count in sorted(corpus_languages(inventory.corpus_rows).items()):
        covered = frozenset(
            benchmark for benchmark in benchmarks if _is_covered(code, benchmark, inventory, language_map)
        )
        rows.append(CoverageRow(code=code, rows=count, covered=covered))
    return rows


def _rows_where(rows: Sequence[CoverageRow], predicate: Callable[[CoverageRow], bool]) -> int:
    return sum(row.rows for row in rows if predicate(row))


def summarize(rows: Sequence[CoverageRow]) -> CoverageSummary:
    return CoverageSummary(
        configs=len(rows),
        total_rows=_rows_where(rows, lambda row: True),
        native_rows=_rows_where(rows, lambda row: row.native),
        uncovered_rows=_rows_where(rows, lambda row: row.status == STATUS_NONE),
        status_counts=dict(Counter(row.status for row in rows)),
    )


def _table_line(row: CoverageRow, benchmarks: Sequence[str]) -> str:
    marks = " | ".join("yes" if benchmark in row.covered else "-" for benchmark in benchmarks)
    return f"| {row.code} | {row.rows:,} | {marks} | {row.status} |"


def render_markdown(rows: Sequence[CoverageRow], benchmarks: Sequence[str] = BENCHMARKS) -> str:
    summary = summarize(rows)
    header = "| corpus config | rows | " + " | ".join(HEADERS[b] for b in benchmarks) + " | status |"
    divider = "|---|---:|" + ":-:|" * len(benchmarks) + "---|"
    native_share = 100 * summary.native_rows / summary.total_rows if summary.total_rows else 0.0
    lines = [
        "# Language x benchmark coverage (X->EN)",
        "",
        "WMT22 and WMT23 X->EN are the only WMT sets used. WMT24 is not used.",
        "",
        header,
        divider,
        *(_table_line(row, benchmarks) for row in rows),
        "",
        f"- Non-English configs: {summary.configs} ({summary.total_rows:,} rows)",
        f"- Status counts: {summary.status_counts}",
        f"- Rows with a native-origin benchmark (WMT or Tatoeba): "
        f"{summary.native_rows:,} ({native_share:.1f}%)",
        f"- Rows with no benchmark: {summary.uncovered_rows:,}",
        "",
    ]
    return "\n".join(lines)
