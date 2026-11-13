import pytest

from tests.fakes import FakeBackend
from xx_to_en.backends import clean_output
from xx_to_en.benchmarks import Segment
from xx_to_en.prompts import MADLAD_TARGET_TAG, PLAMO_STOP, make_prompt, template_names


def _segment(corpus_lang: str, source: str = "Bonjour le monde.") -> Segment:
    return Segment("flores_plus_devtest", corpus_lang, corpus_lang, "0", source, "Hello world.")


ISO = {"fra_Latn": "fr", "deu_Latn": "de"}.get


def test_every_model_template_is_registered():
    expected = {
        "generic_chat",
        "translategemma",
        "plamo3_translate",
        "hy_mt2",
        "nllb_tag",
        "madlad_tag",
    }
    assert expected <= template_names()


def test_translategemma_uses_codes_in_the_content_list():
    backend = FakeBackend()
    make_prompt("translategemma", _segment("fra_Latn"), backend, ISO)
    content = backend.last_messages[0]["content"]
    assert backend.last_messages[0]["role"] == "user"
    assert content == [
        {
            "type": "text",
            "source_lang_code": "fr",
            "target_lang_code": "en",
            "text": "Bonjour le monde.",
        }
    ]


def test_translategemma_rejects_languages_without_iso_639_1():
    with pytest.raises(ValueError, match="no ISO 639-1 code"):
        make_prompt("translategemma", _segment("ceb_Latn"), FakeBackend(), ISO)


def test_plamo_raw_format_adds_bos_and_stop_strings():
    prompt = make_prompt("plamo3_translate", _segment("jpn_Jpan", "会議は来週です。"), FakeBackend())
    assert prompt.add_special_tokens is True
    assert prompt.stop == PLAMO_STOP
    assert prompt.text.startswith("<|plamo:tag|>dataset<|plamo:msg|>translation")
    assert "<|plamo:tag|>input<|plamo:msg|>会議は来週です。" in prompt.text
    assert prompt.text.endswith("<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>")


def test_hy_mt2_instruction_is_sent_as_one_user_message():
    backend = FakeBackend()
    make_prompt("hy_mt2", _segment("fra_Latn"), backend)
    assert backend.last_messages[0]["role"] == "user"
    assert "Translate the following text into English" in backend.last_messages[0]["content"]
    assert backend.last_messages[0]["content"].endswith("Bonjour le monde.")


def test_nllb_prompt_carries_the_source_tag_not_text_prefixes():
    prompt = make_prompt("nllb_tag", _segment("fra_Latn"), FakeBackend())
    assert prompt.text == "Bonjour le monde."
    assert prompt.src_lang == "fra_Latn"


def test_madlad_prompt_prefixes_the_english_target_tag():
    prompt = make_prompt("madlad_tag", _segment("fra_Latn"), FakeBackend())
    assert prompt.text == MADLAD_TARGET_TAG + "Bonjour le monde."
    assert prompt.text.startswith("<2en> ")


def test_generic_chat_still_renders_an_instruction():
    backend = FakeBackend()
    make_prompt("generic_chat", _segment("fra_Latn"), backend)
    assert "Translate the following text into English" in backend.last_messages[0]["content"]


def test_iso_lookup_may_be_a_mapping_or_missing():
    backend = FakeBackend()
    with pytest.raises(ValueError):
        make_prompt("translategemma", _segment("fra_Latn"), backend, {})
    prompt = make_prompt("translategemma", _segment("fra_Latn"), backend, {"fra_Latn": "fr"})
    assert prompt.text


def test_clean_output_drops_reasoning_blocks():
    assert clean_output("<|channel>thought\nplan<channel|>Hello world.") == "Hello world."
    assert clean_output("reasoning</think>Bonjour") == "Bonjour"
    assert clean_output("  plain output  ") == "plain output"
