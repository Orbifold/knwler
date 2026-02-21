import pytest
from unittest.mock import patch, MagicMock
from knwler.llm import (
    discover_schema,
    Config,
    Schema,
    load_languages,
    set_language,
    get_language,
    get_prompt
)
import json



@pytest.fixture
def config():
    """Create a test Config instance."""
    return Config(
        use_openai=False,
        temperature=0.1,
        num_predict=1024,
        use_cache=False,
    )


@pytest.fixture
def sample_text():
    """Create a sample text for testing."""
    return "John works at Acme Corp in New York. Jane lives in Boston and uses Python. Acme Corp partners with Tech Solutions."


def test_languages(config, sample_text):
    langs = load_languages()
    assert isinstance(langs, dict)
    assert len(langs) == 5
    assert "en" in langs
    assert "name" in langs["en"]
    assert "prompts" in langs["en"]
    assert "ui" in langs["en"]
    assert "console" in langs["en"]
    assert len(langs["en"]["prompts"]["community_labeling"]) > 0


def test_discover_schema_short_text(config, sample_text):
    """Test schema discovery with text shorter than sample_size."""
    with patch("knwler.llm.llm_generate") as mock_generate:
        mock_response = '{"entity_types": ["person", "organization"], "relation_types": ["works_at"], "reasoning": "Found people and companies"}'
        mock_generate.return_value = mock_response

        result = discover_schema(sample_text, config, sample_size=4000)

        assert isinstance(result, Schema)
        assert result.entity_types == ["person", "organization"]
        assert result.relation_types == ["works_at"]
        assert result.reasoning == "Found people and companies"
        assert result.discovery_time > 0


def test_discover_schema_long_text(config):
    """Test schema discovery with text longer than sample_size."""
    long_text = "word " * 5000

    with patch("knwler.llm.llm_generate") as mock_generate:
        mock_response = '{"entity_types": ["type1"], "relation_types": ["rel1"], "reasoning": "test"}'
        mock_generate.return_value = mock_response

        result = discover_schema(long_text, config, sample_size=1000)

        assert isinstance(result, Schema)
        mock_generate.assert_called_once()


def test_discover_schema_empty_response(config, sample_text):
    """Test schema discovery when LLM returns empty entity_types."""
    with patch("knwler.llm.llm_generate") as mock_generate:
        mock_generate.return_value = "{}"

        result = discover_schema(sample_text, config)

        assert result.entity_types == config.default_entity_types
        assert result.relation_types == config.default_relation_types
        assert result.reasoning == "Discovery failed, using defaults"


def test_discover_schema_max_limits(config, sample_text):
    """Test that entity and relation types are limited by max parameters."""
    with patch("knwler.llm.llm_generate") as mock_generate:
        many_types = [f"type {i}" for i in range(30)]
        many_rels = [f"rel {i}" for i in range(30)]
        mock_response = json.dumps(
            {
                "entity_types": many_types,
                "relation_types": many_rels,
                "reasoning": "test",
            }
        )
        mock_generate.return_value = mock_response

        result = discover_schema(
            sample_text, config, max_entity_types=5, max_relation_types=10
        )

        assert len(result.entity_types) == 5
        assert len(result.relation_types) == 10


def test_discover_schema_invalid_json(config, sample_text):
    """Test schema discovery with invalid JSON response."""
    with patch("knwler.llm.llm_generate") as mock_generate:
        mock_generate.return_value = "invalid json"

        result = discover_schema(sample_text, config)

        assert result.entity_types == config.default_entity_types
        assert result.relation_types == config.default_relation_types


def test_discover_schema_partial_response(config, sample_text):
    """Test schema discovery with incomplete response data."""
    with patch("knwler.llm.llm_generate") as mock_generate:
        mock_response = '{"entity_types": ["person"], "reasoning": "partial"}'
        mock_generate.return_value = mock_response

        result = discover_schema(sample_text, config)

        assert result.entity_types == ["person"]
        assert result.relation_types == []
        assert result.reasoning == "partial"


def test_discover_schema_custom_prompt(config, sample_text):
    """Test that get_prompt is called with correct parameters."""
    with (
        patch("knwler.llm.get_prompt") as mock_get_prompt,
        patch("knwler.llm.llm_generate") as mock_generate,
    ):
        mock_get_prompt.return_value = "custom prompt"
        mock_generate.return_value = (
            '{"entity_types": ["test"], "relation_types": [], "reasoning": ""}'
        )

        discover_schema(
            sample_text,
            config,
            sample_size=4000,
            max_entity_types=15,
            max_relation_types=20,
        )

        mock_get_prompt.assert_called_once_with(
            "schema_discovery",
            sample=sample_text,
            max_entity_types=15,
            max_relation_types=20,
        )


def test_get_prompt_current_language(config):
    """Test get_prompt returns localized prompt in current language."""
    with patch("knwler.llm.get_lang") as mock_get_lang:
        mock_get_lang.return_value = {"prompts": {"test_key": "Hello {name}"}}

        result = get_prompt("test_key", name="World")
        assert result == "Hello World"


def test_get_prompt_fallback_to_english(config):
    """Test get_prompt falls back to English when key not in current language."""
    with (
        patch("knwler.llm.get_lang") as mock_get_lang,
        patch("knwler.llm.load_languages") as mock_load_langs,
    ):
        mock_get_lang.return_value = {"prompts": {}}
        mock_load_langs.return_value = {
            "en": {"prompts": {"test_key": "Fallback {value}"}}
        }

        result = get_prompt("test_key", value="text")
        assert result == "Fallback text"


def test_get_prompt_missing_key(config):
    """Test get_prompt returns empty string when key doesn't exist."""
    with (
        patch("knwler.llm.get_lang") as mock_get_lang,
        patch("knwler.llm.load_languages") as mock_load_langs,
    ):
        mock_get_lang.return_value = {"prompts": {}}
        mock_load_langs.return_value = {"en": {"prompts": {}}}

        result = get_prompt("nonexistent_key")
        assert result == ""


def test_get_prompt_no_kwargs(config):
    """Test get_prompt without keyword arguments."""
    with patch("knwler.llm.get_lang") as mock_get_lang:
        mock_get_lang.return_value = {"prompts": {"simple": "Static text"}}

        result = get_prompt("simple")
        assert result == "Static text"


def test_get_prompt_multiple_kwargs(config):
    """Test get_prompt with multiple keyword arguments."""
    with patch("knwler.llm.get_lang") as mock_get_lang:
        mock_get_lang.return_value = {
            "prompts": {"multi": "{greeting} {name}, welcome to {place}"}
        }

        result = get_prompt("multi", greeting="Hello", name="Alice", place="Wonderland")
        assert result == "Hello Alice, welcome to Wonderland"


def test_set_language_valid(config):
    """Test set_language with a valid language code."""
    with patch("knwler.llm.load_languages") as mock_load_langs:
        mock_load_langs.return_value = {"ap": {}, "de": {}, "fr": {}}

        set_language("ap")

        assert get_language() == "ap"


def test_set_language_invalid_fallback(config):
    """Test set_language with an invalid code falls back to English."""
    with (
        patch("knwler.llm.load_languages") as mock_load_langs,
        patch("knwler.llm.console.print") as mock_print,
    ):
        mock_load_langs.return_value = {"en": {}, "de": {}}

        set_language("invalid_lang")

        assert get_language() == "en"
        mock_print.assert_called_once()
        assert "invalid_lang" in str(mock_print.call_args)


def test_set_language_english(config):
    """Test set_language with English code."""
    with patch("knwler.llm.load_languages") as mock_load_langs:
        mock_load_langs.return_value = {"en": {}}

        set_language("en")

        assert get_language() == "en"


def test_set_language_multiple_valid(config):
    """Test set_language switches between multiple valid languages."""
    with patch("knwler.llm.load_languages") as mock_load_langs:
        mock_load_langs.return_value = {"en": {}, "es": {}, "it": {}}

        set_language("es")
        assert get_language() == "es"

        set_language("it")
        assert get_language() == "it"
