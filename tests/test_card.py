from xx_to_en.card import ModelScore, render, results_table, winners_table


def test_results_table_sorts_by_volume_weighted_score_and_formats_missing_values():
    rows = [
        ModelScore("low", 40.0, 41.0, None, None, 0.0, 10),
        ModelScore("high", 50.0, 52.5, 0.851, 1.23, 0.02, 20),
    ]
    table = results_table(rows)
    assert table.index("| high |") < table.index("| low |")
    assert "52.5" in table
    assert "0.851" in table
    assert "2.0%" in table
    assert "—" in table


def test_results_table_says_so_when_there_are_no_scores():
    assert "No scores yet" in results_table([])


def test_winners_table_is_collapsible_and_sorted_by_language():
    table = winners_table({"fra_Latn": ("a", 40.0), "deu_Latn": ("b", 50.0)})
    assert table.startswith("<details>")
    assert table.index("deu_Latn") < table.index("fra_Latn")


def test_render_has_front_matter_and_sections():
    card = render("2026-11-13 14:00 CET", "R", "")
    assert card.startswith("---\nlicense: cc-by-sa-4.0")
    for section in ("## Results", "## Method", "## Data"):
        assert section in card
    assert "## Progress" not in card
    assert "## Known gaps" not in card


def test_render_adds_trade_offs_and_split_note_only_when_given():
    card = render("now", "R", "", trade_offs="![x](plots/x.svg)", split_note="`nllb-200-3.3b`: note")
    assert card.index("`nllb-200-3.3b`") < card.index("## Trade-offs") < card.index("## Method")
    bare = render("now", "R", "")
    assert "## Trade-offs" not in bare
    assert "`nllb-200-3.3b`" not in bare
