from xx_to_en.card import Cell, ModelScore, bar, progress_table, render, results_table, winners_table


def test_bar_scales_and_handles_empty_totals():
    assert bar(0, 4) == "░░░░░░"
    assert bar(4, 4) == "██████"
    assert bar(0, 0) == "—"


def test_progress_table_shows_generated_over_total_and_pending_cells():
    grid = {"m": {"flores_plus_devtest": Cell(2, 0, 4)}}
    table = progress_table(grid)
    assert "2/4" in table
    assert "| m |" in table
    assert "—" in table


def test_results_table_sorts_by_volume_weighted_score_and_formats_missing_values():
    rows = [
        ModelScore("low", 40.0, 41.0, None, None, 0.0, 10),
        ModelScore("high", 50.0, 52.5, 0.851, 1.23, 0.02, 20),
    ]
    table = results_table(rows)
    assert table.index("| high |") < table.index("| low |")
    assert "52.5" in table and "0.851" in table and "2.0%" in table
    assert "—" in table


def test_results_table_says_so_when_there_are_no_scores():
    assert "No scores yet" in results_table([])


def test_winners_table_is_collapsible_and_sorted_by_language():
    table = winners_table({"fra_Latn": ("a", 40.0), "deu_Latn": ("b", 50.0)})
    assert table.startswith("<details>")
    assert table.index("deu_Latn") < table.index("fra_Latn")


def test_render_has_front_matter_and_sections():
    card = render("2026-11-13 14:00 CET", "P", "R", "")
    assert card.startswith("---\nlicense: cc-by-sa-4.0")
    for section in ("## Progress", "## Results", "## Files", "## Method"):
        assert section in card
