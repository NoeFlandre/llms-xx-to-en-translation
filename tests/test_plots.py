from xx_to_en.plots import Point, pareto_frontier, tradeoff_svg


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
    assert "higher is better" in svg


def test_svg_says_when_lower_is_better():
    svg = tradeoff_svg("MetricX-24", [Point("a", 1.0, 2.0), Point("b", 10.0, 1.0)], higher_is_better=False)
    assert "lower is better" in svg


def test_svg_without_points_is_empty():
    assert tradeoff_svg("chrF++", [], higher_is_better=True) == ""


def test_svg_with_one_model_draws_no_frontier_line():
    svg = tradeoff_svg("chrF++", [Point("only", 3.0, 40.0)], higher_is_better=True)
    assert "<polyline" not in svg
    assert "only" in svg
