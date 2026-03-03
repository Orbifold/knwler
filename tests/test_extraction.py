import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from knwler.config import Config
from knwler.models import ExtractionResult, Schema, Graph


"""
Tests for knwler.extraction module.
"""


from knwler.extraction import (
    extract_graph,
    extract_chunk,
    _save_partial_results,
    extract_all,
)


@pytest.fixture
def mock_schema():
    return Schema(
        entity_types=["Person", "Organization"],
        relation_types=["works_for", "knows"],
        reasoning="Test reasoning",
    )


@pytest.fixture
def mock_config():
    config = Mock(spec=Config)
    config.max_concurrent = 3
    return config


@pytest.fixture
def sample_text():
    return "John works at Acme Corporation."


class TestExtractGraph:
    @patch("knwler.extraction.llm_generate")
    @patch("knwler.extraction.parse_json_response")
    @patch("knwler.extraction.get_prompt")
    def test_extract_graph_success(
        self,
        mock_get_prompt,
        mock_parse,
        mock_llm,
        mock_schema,
        sample_text,
        mock_config,
    ):
        mock_get_prompt.return_value = "test prompt"
        mock_llm.return_value = '{"entities": [], "relations": []}'
        mock_parse.return_value = {
            "entities": [{"name": "John", "type": "Person"}],
            "relations": [],
        }

        result = extract_graph(sample_text, mock_schema, mock_config)

        assert "entities" in result
        assert "relations" in result
        mock_llm.assert_called_once()

    @patch("knwler.extraction.llm_generate")
    @patch("knwler.extraction.parse_json_response")
    @patch("knwler.extraction.get_prompt")
    def test_extract_graph_fallback_prompt(
        self,
        mock_get_prompt,
        mock_parse,
        mock_llm,
        mock_schema,
        sample_text,
        mock_config,
    ):
        mock_get_prompt.return_value = None
        mock_llm.return_value = '{"entities": [], "relations": []}'
        mock_parse.return_value = {"entities": [], "relations": []}

        result = extract_graph(sample_text, mock_schema, mock_config)

        assert isinstance(result, dict)
        mock_llm.assert_called_once()


class TestExtractChunk:
    @patch("knwler.extraction.extract_graph")
    @patch("knwler.extraction.get_encoder")
    def test_extract_chunk_success(
        self, mock_encoder, mock_extract, mock_schema, mock_config
    ):
        mock_encoder.return_value.encode.return_value = [1, 2, 3]
        mock_extract.return_value = {"entities": [{"name": "Test"}], "relations": []}

        result = extract_chunk("test chunk", 0, mock_schema, mock_config)

        assert isinstance(result, ExtractionResult)
        assert result.id is not None and isinstance(result.id, str)
        assert result.chunk_idx == 0
        assert result.chunk_tokens == 3
        assert result.chunk_time >= 0

    @patch("knwler.extraction.extract_graph")
    @patch("knwler.extraction.get_encoder")
    def test_extract_chunk_timing(
        self, mock_encoder, mock_extract, mock_schema, mock_config
    ):
        mock_encoder.return_value.encode.return_value = []
        mock_extract.return_value = {"entities": [], "relations": []}

        result = extract_chunk("chunk", 5, mock_schema, mock_config)

        assert result.chunk_idx == 5
        assert result.chunk_time >= 0


class TestSavePartialResults:
    def test_save_partial_results(self, tmp_path, mock_schema):
        output_path = tmp_path / "output.json"
        results = [
            ExtractionResult(
                entities=[{"name": "Entity1", "type": "Person"}],
                relations=[],
                chunk_idx=0,
                chunk_time=1.0,
                chunk_tokens=10,
            ),
        ]

        _save_partial_results(output_path, mock_schema, results, 1, 2)

        partial_path = output_path.with_suffix(".partial.json")
        assert partial_path.exists()

        data = json.loads(partial_path.read_text())
        assert data["status"] == "in_progress"
        assert data["progress"] == "1/2"
        assert len(data["results"]) == 1


class TestExtractAll:
    @pytest.mark.asyncio
    @patch("knwler.extraction.extract_chunk")
    @patch("knwler.extraction.get_console_msg")
    async def test_extract_all_success(
        self, mock_msg, mock_chunk, mock_schema, mock_config
    ):
        mock_msg.return_value = "Extracting..."
        mock_chunk.return_value = ExtractionResult(
            entities=[], relations=[], chunk_idx=0, chunk_time=0.1, chunk_tokens=10
        )
        chunks = ["chunk1", "chunk2"]

        results = await extract_all(chunks, mock_schema, mock_config)

        assert len(results) == 2
        assert all(isinstance(r, ExtractionResult) for r in results)

    @pytest.mark.asyncio
    @patch("knwler.extraction.extract_chunk")
    @patch("knwler.extraction.get_console_msg")
    async def test_extract_all_with_output_path(
        self, mock_msg, mock_chunk, tmp_path, mock_schema, mock_config
    ):
        mock_msg.return_value = "Extracting..."
        mock_chunk.return_value = ExtractionResult(
            entities=[], relations=[], chunk_idx=0, chunk_time=0.1, chunk_tokens=10
        )
        output_path = tmp_path / "output.json"
        chunks = ["chunk1"]

        results = await extract_all(chunks, mock_schema, mock_config, output_path)

        assert len(results) == 1
        partial_path = output_path.with_suffix(".partial.json")
        assert not partial_path.exists()
