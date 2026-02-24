import pytest

pytestmark = pytest.mark.basic


def test_languages():
    from knwler.language import load_languages, set_language, get_current_language

    langs = load_languages()
    assert isinstance(langs, dict)
    assert "en" in langs

    set_language("en")
    assert get_current_language() == "en"

    set_language("de")
    assert get_current_language() == "de"

    set_language("nonexistent")
    assert get_current_language() == "en"


def test_prompts():
    from knwler.language import get_prompt

    prompt = get_prompt("language_detection", sample="Test text")
    assert isinstance(prompt, str)
    assert "Detect the language" in prompt
