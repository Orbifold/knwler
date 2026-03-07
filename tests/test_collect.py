import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from urllib.parse import urlparse
import aiohttp
from bs4 import BeautifulSoup
from knwler.collect import *


class TestSanitizeUrl:
    """Tests for sanitize_url function."""

    def test_sanitize_url_with_valid_https(self):
        """Test that valid HTTPS URL is returned unchanged."""
        url = "https://example.com"
        assert WebpageCollector.sanitize_url(url) == url

    def test_sanitize_url_with_valid_http(self):
        """Test that valid HTTP URL is returned unchanged."""
        url = "http://example.com"
        assert WebpageCollector.sanitize_url(url) == url

    def test_sanitize_url_adds_https_when_missing(self):
        """Test that HTTPS is added when scheme is missing."""
        url = "example.com"
        result = WebpageCollector.sanitize_url(url)
        assert result == "https://example.com"

    def test_sanitize_url_with_path(self):
        """Test that URL with path is sanitized correctly."""
        url = "example.com/path/to/page"
        result = WebpageCollector.sanitize_url(url)
        assert result == "https://example.com/path/to/page"

    def test_sanitize_url_strips_whitespace(self):
        """Test that whitespace is stripped from URL."""
        url = "  https://example.com  "
        result = WebpageCollector.sanitize_url(url)
        assert result == "https://example.com"

    def test_sanitize_url_empty_raises_error(self):
        """Test that empty URL raises ValueError."""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            WebpageCollector.sanitize_url("")

    def test_sanitize_url_invalid_scheme_raises_error(self):
        """Test that invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            WebpageCollector.sanitize_url("ftp://example.com")

    def test_sanitize_url_missing_netloc_raises_error(self):
        """Test that missing host raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URL: missing host"):
            WebpageCollector.sanitize_url("https://")


class TestFetchPage:
    """Tests for fetch_page function."""

    @pytest.mark.asyncio
    async def test_fetch_page_success(self):
        """Test successful page fetch."""
        url = "https://example.com"
        mock_html = "<html><title>Test Page</title><body>Content</body></html>"

        with (
            patch("knwler.collect.webpage.get_cached_webpage", return_value=None),
            patch("knwler.collect.webpage.cache_webpage"),
            patch("aiohttp.ClientSession") as mock_session_class,
        ):
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=mock_html)

            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session.get = MagicMock(return_value=AsyncMock())
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session.get.return_value.__aexit__.return_value = None

            mock_session_class.return_value = mock_session

            result = await WebpageCollector.fetch_page(url)

            assert result["id"] == url
            assert result["name"] == "Test Page"
            assert "text" in result
            assert "content" in result
            assert result["description"] == "Webpage content"

    @pytest.mark.asyncio
    async def test_fetch_page_404_raises_error(self):
        """Test that 404 response raises ValueError."""
        url = "https://example.com/notfound"

        with (
            patch("knwler.collect.webpage.get_cached_webpage", return_value=None),
            patch("knwler.collect.webpage.cache_webpage"),
            patch("aiohttp.ClientSession") as mock_session_class,
        ):
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 404

            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session.get = MagicMock(return_value=AsyncMock())
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session.get.return_value.__aexit__.return_value = None

            mock_session_class.return_value = mock_session

            with pytest.raises(ValueError, match="Webpage not found"):
                await WebpageCollector.fetch_page(url)

    @pytest.mark.asyncio
    async def test_fetch_page_403_raises_error(self):
        """Test that 403 response raises ValueError."""
        url = "https://example.com/forbidden"

        with (
            patch("knwler.collect.webpage.get_cached_webpage", return_value=None),
            patch("knwler.collect.webpage.cache_webpage"),
            patch("aiohttp.ClientSession") as mock_session_class,
        ):
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 403

            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session.get = MagicMock(return_value=AsyncMock())
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session.get.return_value.__aexit__.return_value = None

            mock_session_class.return_value = mock_session

            with pytest.raises(ValueError, match="Access forbidden"):
                await WebpageCollector.fetch_page(url)

    @pytest.mark.asyncio
    async def test_fetch_page_other_status_raises_error(self):
        """Test that other error status codes raise ValueError."""
        url = "https://example.com"

        with (
            patch("knwler.collect.webpage.get_cached_webpage", return_value=None),
            patch("knwler.collect.webpage.cache_webpage"),
            patch("aiohttp.ClientSession") as mock_session_class,
        ):
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 500

            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session.get = MagicMock(return_value=AsyncMock())
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session.get.return_value.__aexit__.return_value = None

            mock_session_class.return_value = mock_session

            with pytest.raises(ValueError, match="Failed to fetch webpage"):
                await WebpageCollector.fetch_page(url)

    @pytest.mark.asyncio
    async def test_fetch_page_sanitizes_url(self):
        """Test that fetch_page sanitizes URL before fetching."""
        url = "example.com"
        mock_html = "<html><title>Test</title></html>"

        with (
            patch("knwler.collect.webpage.get_cached_webpage", return_value=None),
            patch("knwler.collect.webpage.cache_webpage"),
            patch("aiohttp.ClientSession") as mock_session_class,
        ):
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=mock_html)

            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session.get = MagicMock(return_value=AsyncMock())
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session.get.return_value.__aexit__.return_value = None

            mock_session_class.return_value = mock_session

            await WebpageCollector.fetch_page(url)

            mock_session.get.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_fetch_page_uses_default_title_when_missing(self):
        """Test that URL is used as title when title tag is missing."""
        url = "https://example.com"
        mock_html = "<html><body>No title</body></html>"

        with (
            patch("knwler.collect.webpage.get_cached_webpage", return_value=None),
            patch("knwler.collect.webpage.cache_webpage"),
            patch("aiohttp.ClientSession") as mock_session_class,
        ):
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=mock_html)

            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = None
            mock_session.get = MagicMock(return_value=AsyncMock())
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session.get.return_value.__aexit__.return_value = None

            mock_session_class.return_value = mock_session

            result = await WebpageCollector.fetch_page(url)

            assert result["name"] == url

    @pytest.mark.asyncio
    async def test_fetch_corp(self):
        found = await WebpageCollector.fetch_page("https://orbifold.net", no_cache=True)
        assert found["id"] == "https://orbifold.net"
        assert "Graph Consulting" in found["name"]
        assert "graph machine learning" in found["content"].lower()
        assert found["cached"] is False
        # Fetch again to test caching
        found = await WebpageCollector.fetch_page("https://orbifold.net")
        assert found["cached"] is True

    @pytest.mark.asyncio
    async def test_download_pdf(self):
        found, content = await WebpageCollector.fetch_document(
            "https://temprl.com/knwler/BurgerlijkBoek1.pdf", no_cache=True
        )
        assert found["id"] == "https://temprl.com/knwler/BurgerlijkBoek1.pdf"
        assert "Burgerlijk" in found["name"]
        assert "pdf" in found["file_path"]
        assert found["cached"] is False
        # Fetch again to test caching
        found, content = await WebpageCollector.fetch_document(
            "https://temprl.com/knwler/BurgerlijkBoek1.pdf"
        )
        assert found["cached"] is True

    @pytest.mark.asyncio
    async def test_get_wikipedia(self):
        found = await WikipediaCollector.fetch_article(
            "Artificial intelligence", no_cache=True
        )
        assert found["id"] == "https://en.wikipedia.org/wiki/Artificial_intelligence"
        assert "Artificial intelligence" in found["name"]
        assert "through the backpropagation algorithm" in found["text"].lower()
        assert found["cached"] is False
        # Fetch again to test caching
        found = await WikipediaCollector.fetch_article("Artificial intelligence")
        assert found["cached"] is True

        title = WikipediaCollector.take_random_wiki_title()
        assert isinstance(title, str)
        assert len(title) > 0