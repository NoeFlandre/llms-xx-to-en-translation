import re

from xx_to_en.plots import Point, _label_box, _placements, _scale, pareto_frontier, tradeoff_svg, wins_svg


def test_frontier_keeps_points_no_smaller_model_beats():
    points = [
        Point("small-weak", 1.0, 40.0),
        Point("small-good", 1.0, 45.0),
        Point("mid-worse", 5.0, 44.0),
        Point("big-best", 20.0, 50.0),
    ]
    assert [p.label for p in pareto_frontier(points, higher_is_better=True)] == ["small-good", "big-best"]


def test_frontier_for_lower_is_better_values():
    points = [Point("a", 1.0, 3.0), Point("b", 2.0, 2.0), Point("c", 3.0, 2.5)]
    assert [p.label for p in pareto_frontier(points, higher_is_better=False)] == ["a", "b"]


def test_svg_labels_each_model_and_draws_the_frontier():
    points = [Point("tiny", 0.5, 30.0), Point("big", 20.0, 50.0), Point("odd&one", 2.0, 20.0)]
    svg = tradeoff_svg("chrF++", points, higher_is_better=True)
    assert svg.startswith("<svg")
    assert "tiny" in svg
    assert "big" in svg
    assert "odd&amp;one" in svg
    assert "<polyline" in svg
    assert "Higher is better" in svg


def test_svg_says_when_lower_is_better():
    svg = tradeoff_svg("MetricX-24", [Point("a", 1.0, 2.0), Point("b", 10.0, 1.0)], higher_is_better=False)
    assert "Lower is better" in svg


def test_labels_near_the_right_edge_are_set_to_the_left_of_their_marker():
    points = [Point("small", 0.5, 30.0), Point("largest", 50.0, 50.0)]
    svg = tradeoff_svg("chrF++", points, higher_is_better=True)
    assert 'text-anchor="end"' in svg
    assert 'text-anchor="start"' in svg


def test_decimals_follow_the_metric():
    points = [Point("a", 1.0, 2.5), Point("b", 10.0, 3.5)]
    three = tradeoff_svg("COMET-22", points, higher_is_better=True, decimals=3)
    one = tradeoff_svg("chrF++", points, higher_is_better=True, decimals=1)
    assert re.search(r">\d\.\d{3}</text>", three)
    assert re.search(r">\d\.\d</text>", one)
    assert not re.search(r">\d\.\d{3}</text>", one)


def test_crowded_labels_do_not_overlap_each_other():
    points = [
        Point("alpha-model", 8.0, 66.0),
        Point("bravo-model", 8.1, 66.05),
        Point("charlie-model", 8.2, 65.95),
        Point("delta-model", 8.05, 66.1),
        Point("far", 1.0, 50.0),
        Point("farther", 30.0, 70.0),
    ]
    placed = _placements(_scale(points), points)
    boxes = [_label_box(point.label, *position) for point, position in zip(points, placed, strict=True)]
    for index, box in enumerate(boxes):
        assert not any(box.overlaps(other) for other in boxes[index + 1 :])


def test_svg_without_points_is_empty():
    assert tradeoff_svg("chrF++", [], higher_is_better=True) == ""


def test_svg_with_one_model_draws_no_frontier_line():
    svg = tradeoff_svg("chrF++", [Point("only", 3.0, 40.0)], higher_is_better=True)
    assert "<polyline" not in svg
    assert "only" in svg


def test_wins_chart_lists_every_model_with_its_count():
    svg = wins_svg("Best model per language", [("gemma-4-31b-it", 5), ("hy-mt2-7b", 0)])
    assert "gemma-4-31b-it" in svg
    assert ">5</text>" in svg
    assert "hy-mt2-7b" in svg


def test_wins_chart_is_empty_without_models():
    assert wins_svg("Best model per language", []) == ""
