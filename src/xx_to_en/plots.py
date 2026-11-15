"""Performance against parameter count, one SVG per metric, with the Pareto frontier highlighted."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape

WIDTH = 760
HEIGHT = 480
LEFT = 72
RIGHT = 24
TOP = 64
BOTTOM = 64
LABEL_FLIP = 150
X_TICKS = (0.5, 1, 2, 5, 10, 20, 50, 100)
POINT = "#4c72b0"
FRONTIER = "#c44e52"
GRID = "#e8e8e8"
AXIS = "#999999"
TEXT = "#333333"
FONT = "Helvetica, Arial, sans-serif"


@dataclass(frozen=True)
class Point:
    label: str
    params_b: float
    value: float


def pareto_frontier(points: Sequence[Point], higher_is_better: bool) -> list[Point]:
    """Points that no other point beats on both smaller parameter count and better value."""
    sign = 1.0 if higher_is_better else -1.0
    frontier: list[Point] = []
    best = -math.inf
    for point in sorted(points, key=lambda p: (p.params_b, -sign * p.value)):
        if sign * point.value > best:
            frontier.append(point)
            best = sign * point.value
    return frontier


@dataclass(frozen=True)
class _Scale:
    x_low: float
    x_high: float
    y_low: float
    y_high: float

    def x_of(self, params_b: float) -> float:
        return LEFT + (math.log10(params_b) - self.x_low) / (self.x_high - self.x_low) * (
            WIDTH - LEFT - RIGHT
        )

    def y_of(self, value: float) -> float:
        return TOP + (self.y_high - value) / (self.y_high - self.y_low) * (HEIGHT - TOP - BOTTOM)


def _scale(points: Sequence[Point]) -> _Scale:
    xs = [math.log10(p.params_b) for p in points]
    x_low, x_high = min(xs), max(xs)
    if x_high - x_low < 0.1:
        x_low, x_high = x_low - 0.05, x_high + 0.05
    values = [p.value for p in points]
    pad = (max(values) - min(values)) * 0.1 or 1.0
    return _Scale(x_low, x_high, min(values) - pad, max(values) + pad)


def _header(title: str, subtitle: str) -> list[str]:
    return [
        f'<text x="{LEFT}" y="30" font-size="17" font-weight="600" fill="{TEXT}">{escape(title)}</text>',
        f'<text x="{LEFT}" y="50" font-size="12" fill="#777777">{escape(subtitle)}</text>',
        f'<circle cx="{WIDTH - RIGHT - 190}" cy="26" r="5" fill="{FRONTIER}"/>',
        f'<text x="{WIDTH - RIGHT - 180}" y="30" font-size="12" fill="{TEXT}">Pareto frontier</text>',
        f'<circle cx="{WIDTH - RIGHT - 80}" cy="26" r="5" fill="{POINT}"/>',
        f'<text x="{WIDTH - RIGHT - 70}" y="30" font-size="12" fill="{TEXT}">other models</text>',
    ]


def _axes(scale: _Scale, decimals: int) -> list[str]:
    parts = [
        f'<line x1="{LEFT}" y1="{HEIGHT - BOTTOM}" x2="{WIDTH - RIGHT}" y2="{HEIGHT - BOTTOM}"'
        f' stroke="{AXIS}"/>',
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{HEIGHT - BOTTOM}" stroke="{AXIS}"/>',
        f'<text x="{(LEFT + WIDTH - RIGHT) / 2}" y="{HEIGHT - 14}" text-anchor="middle" font-size="12" '
        f'fill="{TEXT}">parameters (billions, log scale)</text>',
    ]
    parts += [_x_tick(scale, tick) for tick in X_TICKS if scale.x_low <= math.log10(tick) <= scale.x_high]
    parts += [_y_tick(scale, step, decimals) for step in range(5)]
    return parts


def _x_tick(scale: _Scale, tick: float) -> str:
    x = scale.x_of(tick)
    return (
        f'<line x1="{x:.1f}" y1="{HEIGHT - BOTTOM}" x2="{x:.1f}" y2="{HEIGHT - BOTTOM + 5}" stroke="{AXIS}"/>'
        f'<text x="{x:.1f}" y="{HEIGHT - BOTTOM + 19}" text-anchor="middle" font-size="11" '
        f'fill="{TEXT}">{tick:g}</text>'
    )


def _y_tick(scale: _Scale, step: int, decimals: int) -> str:
    value = scale.y_low + (scale.y_high - scale.y_low) * step / 4
    y = scale.y_of(value)
    return (
        f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" stroke="{GRID}"/>'
        f'<text x="{LEFT - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="{TEXT}">'
        f"{value:.{decimals}f}</text>"
    )


def _frontier_line(scale: _Scale, frontier: Sequence[Point]) -> str:
    if len(frontier) < 2:
        return ""
    line = " ".join(f"{scale.x_of(p.params_b):.1f},{scale.y_of(p.value):.1f}" for p in frontier)
    return (
        f'<polyline points="{line}" fill="none" stroke="{FRONTIER}" stroke-width="2.5" '
        'stroke-linejoin="round"/>'
    )


def _marker(scale: _Scale, point: Point, colour: str) -> str:
    x, y = scale.x_of(point.params_b), scale.y_of(point.value)
    left = x > WIDTH - RIGHT - LABEL_FLIP
    label_x, anchor = (x - 9, "end") if left else (x + 9, "start")
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colour}" stroke="white" stroke-width="1.5"/>'
        f'<text x="{label_x:.1f}" y="{y + 4:.1f}" font-size="11" fill="{TEXT}" text-anchor="{anchor}" '
        'stroke="white" stroke-width="3" paint-order="stroke">'
        f"{escape(point.label)}</text>"
    )


def _markers(scale: _Scale, points: Sequence[Point], frontier_labels: set[str]) -> list[str]:
    return [_marker(scale, p, FRONTIER if p.label in frontier_labels else POINT) for p in points]


def tradeoff_svg(title: str, points: Sequence[Point], higher_is_better: bool, decimals: int = 2) -> str:
    if not points:
        return ""
    scale = _scale(points)
    frontier = pareto_frontier(points, higher_is_better)
    subtitle = "Higher is better" if higher_is_better else "Lower is better"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="{FONT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
    ]
    parts += _header(title, subtitle)
    parts += _axes(scale, decimals)
    parts.append(_frontier_line(scale, frontier))
    parts += _markers(scale, points, {p.label for p in frontier})
    parts.append("</svg>")
    return "\n".join(parts)


WINS_WIDTH = 760
WINS_ROW = 26
WINS_HEADER = 64
WINS_LABEL = 240


def _wins_row(index: int, name: str, count: int, top: int) -> str:
    y = WINS_HEADER + index * WINS_ROW
    bar = (WINS_WIDTH - WINS_LABEL - 90) * count / top
    return (
        f'<text x="{WINS_LABEL - 10}" y="{y + 17}" text-anchor="end" font-size="12" fill="{TEXT}">'
        f"{escape(name)}</text>"
        f'<rect x="{WINS_LABEL}" y="{y + 5}" width="{bar:.1f}" height="{WINS_ROW - 8}" fill="{POINT}"/>'
        f'<text x="{WINS_LABEL + bar + 8:.1f}" y="{y + 17}" font-size="12" fill="{TEXT}">{count}</text>'
    )


def wins_svg(title: str, counts: Sequence[tuple[str, int]]) -> str:
    """Horizontal bars: how many languages each model is the best at."""
    if not counts:
        return ""
    top = max(max(count for _, count in counts), 1)
    height = WINS_HEADER + WINS_ROW * len(counts) + 16
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WINS_WIDTH}" height="{height}" '
        f'viewBox="0 0 {WINS_WIDTH} {height}" font-family="{FONT}">',
        f'<rect width="{WINS_WIDTH}" height="{height}" fill="white"/>',
        f'<text x="{LEFT}" y="30" font-size="17" font-weight="600" fill="{TEXT}">{escape(title)}</text>',
        f'<text x="{LEFT}" y="50" font-size="12" fill="#777777">'
        "Number of languages where the model has the best chrF++</text>",
    ]
    parts += [_wins_row(index, name, count, top) for index, (name, count) in enumerate(counts)]
    parts.append("</svg>")
    return "\n".join(parts)
