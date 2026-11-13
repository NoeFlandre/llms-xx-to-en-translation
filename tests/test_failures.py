from xx_to_en.failures import (
    detect,
    has_repetition_loop,
    is_length_outlier,
    is_untranslated_copy,
    is_wrong_language,
    words,
)


def test_words_are_casefolded_and_nfkc_normalised():
    assert words("Ｈｅｌｌｏ, World!") == ["hello", "world"]


def test_repetition_loop_detected_for_repeated_word():
    assert has_repetition_loop("the the the the the")


def test_repetition_loop_detected_for_repeated_phrase():
    assert has_repetition_loop("I love it I love it I love it I love it")


def test_normal_sentence_is_not_a_loop():
    assert not has_repetition_loop("The museum opened in 1892 near the river.")


def test_two_repeats_are_not_a_loop():
    assert not has_repetition_loop("very very good")


def test_length_outlier_both_directions():
    source = "one two three four five six seven eight nine ten"
    assert is_length_outlier(source, "")
    assert is_length_outlier(source, " ".join(["word"] * 40))
    assert not is_length_outlier(source, "a b c d e f g h i j")


def test_length_outlier_skipped_for_empty_source():
    assert not is_length_outlier("", "anything at all")


def test_untranslated_copy_ignores_case_and_punctuation():
    assert is_untranslated_copy("Paris, France.", "paris france")
    assert not is_untranslated_copy("Paris, France.", "Paris is in France.")


def test_untranslated_copy_requires_source_words():
    assert not is_untranslated_copy("!!!", "!!!")


def test_wrong_language_requires_identifier():
    assert is_wrong_language("Bonjour", None) is None


def test_wrong_language_uses_identifier_and_skips_empty():
    labels = {"Bonjour le monde": "fra_Latn", "The sky": "eng_Latn"}

    def identify(text: str) -> str:
        return labels[text]

    assert is_wrong_language("Bonjour le monde", identify) is True
    assert is_wrong_language("The sky", identify) is False
    assert is_wrong_language("   ", identify) is False


def test_detect_clean_translation_has_no_flags():
    flags = detect("Le chat dort.", "The cat sleeps.", identify=lambda _: "eng_Latn")
    assert not flags.any
    assert flags.wrong_language is False


def test_detect_flags_empty_output():
    flags = detect("Le chat dort.", "   ")
    assert flags.empty
    assert flags.length_ratio_outlier
    assert flags.any


def test_detect_without_identifier_leaves_wrong_language_unknown():
    flags = detect("Le chat dort.", "The cat sleeps.")
    assert flags.wrong_language is None
    assert not flags.any
