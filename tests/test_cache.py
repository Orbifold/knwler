import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from knwler.cache import CACHE_DIR, cache_key, get_cached_response, save_to_cache

"""Tests for knwler.cache module."""


class TestCacheKey:
    """Tests for cache_key function."""

    def test_cache_key_generation(self):
        """Test that cache_key generates a valid SHA256 hash."""
        key = cache_key("test prompt", "gpt-4", 0.7, 100)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex digest length

    def test_cache_key_consistency(self):
        """Test that same inputs produce same cache key."""
        key1 = cache_key("prompt", "model", 0.5, 50)
        key2 = cache_key("prompt", "model", 0.5, 50)
        assert key1 == key2

    def test_cache_key_different_for_different_inputs(self):
        """Test that different inputs produce different keys."""
        key1 = cache_key("prompt1", "model", 0.5, 50)
        key2 = cache_key("prompt2", "model", 0.5, 50)
        assert key1 != key2


class TestGetCachedResponse:
    """Tests for get_cached_response function."""

    def test_get_cached_response_not_found(self):
        """Test retrieval when cache file doesn't exist."""
        result = get_cached_response("nonexistent_key")
        assert result is None

    def test_get_cached_response_found(self):
        """Test successful retrieval of cached response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("knwler.cache.CACHE_DIR", Path(tmpdir)):
                key = "test_key"
                expected_response = "test response content"
                cache_file = Path(tmpdir) / f"{key}.json"
                cache_file.write_text(
                    json.dumps(
                        {
                            "model": "test-model",
                            "response": expected_response,
                            "cached_at": "2024-01-01 12:00:00",
                        }
                    )
                )

                result = get_cached_response(key)
                assert result == expected_response

    def test_get_cached_response_invalid_json(self):
        """Test handling of corrupted JSON in cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("knwler.cache.CACHE_DIR", Path(tmpdir)):
                key = "corrupted_key"
                cache_file = Path(tmpdir) / f"{key}.json"
                cache_file.write_text("invalid json content{")

                result = get_cached_response(key)
                assert result is None


class TestSaveToCache:
    """Tests for save_to_cache function."""

    def test_save_to_cache_creates_file(self):
        """Test that save_to_cache creates a cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("knwler.cache.CACHE_DIR", Path(tmpdir)):
                key = "test_key"
                response = "test response"
                model = "test-model"

                save_to_cache(key, response, model)

                cache_file = Path(tmpdir) / f"{key}.json"
                assert cache_file.exists()

    def test_save_to_cache_correct_format(self):
        """Test that cached data has correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("knwler.cache.CACHE_DIR", Path(tmpdir)):
                key = "test_key"
                response = "test response"
                model = "test-model"

                save_to_cache(key, response, model)

                cache_file = Path(tmpdir) / f"{key}.json"
                data = json.loads(cache_file.read_text())

                assert data["response"] == response
                assert data["model"] == model
                assert "cached_at" in data

    def test_save_to_cache_creates_directory(self):
        """Test that save_to_cache creates cache directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "newcache"
            with patch("knwler.cache.CACHE_DIR", cache_path):
                save_to_cache("key", "response", "model")
                assert cache_path.exists()
