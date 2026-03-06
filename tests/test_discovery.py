"""Tests for knwler.discovery module."""

import pytest
from unittest.mock import AsyncMock, patch

from knwler.config import Config
from knwler.discovery import detect_language, discover_schema
from knwler.models import Schema


@pytest.fixture
def config():
    """Create a test config."""
    return Config(
        default_entity_types=["entity"],
        default_relation_types=["relation"],
    )


class TestDetectLanguage:
    @pytest.mark.asyncio
    async def test_detect_language_success(self, config):
        with (
            patch("knwler.discovery.llm_generate", new_callable=AsyncMock) as mock_llm,
            patch("knwler.discovery.parse_json_response") as mock_parse,
            patch("knwler.discovery.load_languages") as mock_langs,
        ):
            mock_llm.return_value = '{"language": "en"}'
            mock_parse.return_value = {"language": "en"}
            mock_langs.return_value = {"en": {}, "de": {}, "fr": {}}

            result = await detect_language("Hello world", config)
            assert result == "en"

    @pytest.mark.asyncio
    async def test_detect_language_with_sample_size(self, config):
        long_text = "a" * 5000
        with (
            patch("knwler.discovery.llm_generate", new_callable=AsyncMock) as mock_llm,
            patch("knwler.discovery.parse_json_response") as mock_parse,
            patch("knwler.discovery.load_languages") as mock_langs,
        ):
            mock_llm.return_value = '{"language": "en"}'
            mock_parse.return_value = {"language": "en"}
            mock_langs.return_value = {"en": {}}

            await detect_language(long_text, config, sample_size=1000)
            called_prompt = mock_llm.call_args[0][0]
            assert len(called_prompt) > 0

    @pytest.mark.asyncio
    async def test_detect_language_invalid_code(self, config):
        with (
            patch("knwler.discovery.llm_generate", new_callable=AsyncMock) as mock_llm,
            patch("knwler.discovery.parse_json_response") as mock_parse,
            patch("knwler.discovery.load_languages") as mock_langs,
        ):
            mock_llm.return_value = '{"language": "invalid"}'
            mock_parse.return_value = {"language": "invalid"}
            mock_langs.return_value = {"en": {}}

            result = await detect_language("text", config)
            assert result == "en"


class TestDiscoverSchema:
    @pytest.mark.asyncio
    async def test_discover_schema_success(self, config):
        text = "John works at Apple. IBM owns several subsidiaries."
        with (
            patch("knwler.discovery.get_prompt") as mock_prompt,
            patch("knwler.discovery.llm_generate", new_callable=AsyncMock) as mock_llm,
            patch("knwler.discovery.parse_json_response") as mock_parse,
        ):
            mock_prompt.return_value = "Test prompt"
            mock_llm.return_value = (
                '{"entity_types": ["person"], "relation_types": ["works_at"]}'
            )
            mock_parse.return_value = {
                "entity_types": ["person", "company"],
                "relation_types": ["works_at"],
                "reasoning": "Based on text content",
            }

            result = await discover_schema(text, config)
            assert isinstance(result, Schema)
            assert result.entity_types == ["person", "company"]
            assert result.relation_types == ["works_at"]

    @pytest.mark.asyncio
    async def test_discover_schema_respects_max_types(self, config):
        with (
            patch("knwler.discovery.get_prompt") as mock_prompt,
            patch("knwler.discovery.llm_generate", new_callable=AsyncMock) as mock_llm,
            patch("knwler.discovery.parse_json_response") as mock_parse,
        ):
            mock_prompt.return_value = "Test prompt"
            mock_llm.return_value = "{}"
            mock_parse.return_value = {
                "entity_types": ["e1", "e2", "e3", "e4", "e5"],
                "relation_types": ["r1", "r2", "r3"],
            }

            result = await discover_schema(
                "Sample text", config, max_entity_types=3, max_relation_types=2
            )
            assert len(result.entity_types) == 3
            assert len(result.relation_types) == 2

    @pytest.mark.asyncio
    async def test_discover_schema_fallback_on_failure(self, config):
        with (
            patch("knwler.discovery.get_prompt") as mock_prompt,
            patch("knwler.discovery.llm_generate", new_callable=AsyncMock) as mock_llm,
            patch("knwler.discovery.parse_json_response") as mock_parse,
        ):
            mock_prompt.return_value = "Test prompt"
            mock_llm.return_value = "{}"
            mock_parse.return_value = {"entity_types": None}

            result = await discover_schema("text", config)
            assert result.entity_types == config.default_entity_types
            assert result.relation_types == config.default_relation_types
            assert result.reasoning == "Discovery failed, using defaults"

    @pytest.mark.asyncio
    async def test_discover_schema_sampling(self, config):
        long_text = "a" * 10000
        with (
            patch("knwler.discovery.get_prompt") as mock_prompt,
            patch("knwler.discovery.llm_generate", new_callable=AsyncMock) as mock_llm,
            patch("knwler.discovery.parse_json_response") as mock_parse,
        ):
            mock_prompt.return_value = "Test prompt"
            mock_llm.return_value = '{"entity_types": ["e"], "relation_types": ["r"]}'
            mock_parse.return_value = {"entity_types": ["e"], "relation_types": ["r"]}

            await discover_schema(long_text, config, sample_size=4000)
            mock_prompt.assert_called_once()
            sample_arg = mock_prompt.call_args[1]["sample"]
            assert len(sample_arg) < len(long_text)
