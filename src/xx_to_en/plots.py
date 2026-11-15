"""Performance against parameter count, one SVG per metric, with the Pareto frontier highlighted."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape

WIDTH = 720
HEIGHT = 440
LEFT = 70
RIGHT = 20
TOP = 40
BOTTOM = 60
X_TICKS = (0.5, 1, 2, 5, 10, 20, 50, 100)
POINT = "#4c72b0"
FRONTIER = "#c44e52"


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


def _axes(scale: _Scale, direction: str, title: str) -> list[str]:
    parts = [
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        f'<text x="{WIDTH / 2}" y="24" text-anchor="middle" font-size="16">'
        f"{escape(title)} ({direction})</text>",
        f'<line x1="{LEFT}" y1="{HEIGHT - BOTTOM}" x2="{WIDTH - RIGHT}" y2="{HEIGHT - BOTTOM}"'
        ' stroke="black"/>',
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{HEIGHT - BOTTOM}" stroke="black"/>',
        f'<text x="{(LEFT + WIDTH - RIGHT) / 2}" y="{HEIGHT - 12}" text-anchor="middle" font-size="12">'
        "parameters (billions, log scale)</text>",
    ]
    parts += [_x_tick(scale, tick) for tick in X_TICKS if scale.x_low <= math.log10(tick) <= scale.x_high]
    parts += [_y_tick(scale, step) for step in range(5)]
    return parts


def _x_tick(scale: _Scale, tick: float) -> str:
    x = scale.x_of(tick)
    return (
        f'<line x1="{x:.1f}" y1="{HEIGHT - BOTTOM}" x2="{x:.1f}" y2="{HEIGHT - BOTTOM + 5}" stroke="black"/>'
        f'<text x="{x:.1f}" y="{HEIGHT - BOTTOM + 18}" text-anchor="middle" font-size="11">{tick:g}</text>'
    )


def _y_tick(scale: _Scale, step: int) -> str:
    value = scale.y_low + (scale.y_high - scale.y_low) * step / 4
    y = scale.y_of(value)
    return (
        f'<line x1="{LEFT - 5}" y1="{y:.1f}" x2="{LEFT}" y2="{y:.1f}" stroke="black"/>'
        f'<text x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.2f}</text>'
    )


def _frontier_line(scale: _Scale, frontier: Sequence[Point]) -> str:
    if len(frontier) < 2:
        return ""
    line = " ".join(f"{scale.x_of(p.params_b):.1f},{scale.y_of(p.value):.1f}" for p in frontier)
    return f'<polyline points="{line}" fill="none" stroke="{FRONTIER}" stroke-width="2"/>'


def _marker(scale: _Scale, point: Point, colour: str) -> str:
    x, y = scale.x_of(point.params_b), scale.y_of(point.value)
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}"/>'
        f'<text x="{x + 6:.1f}" y="{y - 6:.1f}" font-size="10">{escape(point.label)}</text>'
    )


def _markers(scale: _Scale, points: Sequence[Point], frontier_labels: set[str]) -> list[str]:
    return [_marker(scale, p, FRONTIER if p.label in frontier_labels else POINT) for p in points]


def tradeoff_svg(title: str, points: Sequence[Point], higher_is_better: bool) -> str:
    if not points:
        return ""
    scale = _scale(points)
    frontier = pareto_frontier(points, higher_is_better)
    direction = "higher is better" if higher_is_better else "lower is better"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="sans-serif">'
    ]
    parts += _axes(scale, direction, title)
    parts.append(_frontier_line(scale, frontier))
    parts += _markers(scale, points, {p.label for p in frontier})
    parts.append("</svg>")
    return "\n".join(parts)
