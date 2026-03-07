from urllib.parse import urlparse, urlunparse

import json

import aiohttp
import markdownify
from bs4 import BeautifulSoup

from knwler.cache import *

document_extensions = [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]


class WebpageCollector:
    DEFAULT_HEADERS = {
        "User-Agent": "Knwl/1.8.0 (https://knwler.com; info@orbifold.net)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    @staticmethod
    def sanitize_url(url: str) -> str:
        """
        Sanitizes and normalizes a URL.
        Ensures the URL has a valid protocol (defaults to https if missing).
        """
        url = url.strip()
        if not url:
            raise ValueError("URL cannot be empty")

        parsed = urlparse(url)

        if not parsed.scheme:
            url = f"https://{url}"
            parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

        if not parsed.netloc:
            raise ValueError(f"Invalid URL: missing host")

        return urlunparse(parsed)

    @staticmethod
    async def fetch_page(url: str, no_cache: bool = False) -> tuple[dict, bytes] | None:
        """
        Fetches the content of a webpage given its URL.
        Responses are cached on disk keyed by URL.
        """
        url = WebpageCollector.sanitize_url(url)
        cached = get_cached_webpage(url)
        if cached is not None and not no_cache:
            result = json.loads(cached)
            result["cached"] = True
            return result, result["content"].encode("utf-8")

        async with aiohttp.ClientSession(
            headers=WebpageCollector.DEFAULT_HEADERS
        ) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    html_content = markdownify.markdownify(text)
                    soup = BeautifulSoup(text, "html.parser")
                    html_title = soup.title.string if soup.title else url
                    result = {
                        "text": html_content,
                        "name": html_title,
                        "id": url,
                        "description": "Webpage content",
                        "content": html_content,
                        "cached": False,
                    }
                    cache_webpage(url, json.dumps(result))
                    return result, html_content.encode("utf-8")
                elif response.status == 404:
                    raise ValueError(f"Webpage not found: {url}")
                elif response.status == 403:
                    raise ValueError(f"Access forbidden to webpage: {url}")
                else:
                    raise ValueError(
                        f"Failed to fetch webpage: {url} with status code {response.status}"
                    )
        return None

    @staticmethod
    async def fetch_document(
        url: str, no_cache: bool = False
    ) -> tuple[dict, bytes] | None:
        """
        Fetches the pdf or document.
        The URL must end with a supported document extension. Supported extensions include .pdf, .doc, .docx, .xls, .xlsx, .ppt, and .pptx.
        The result is cached on disk keyed by URL. The cache includes both the document content and metadata such as filename and extension.
        """
        if not WebpageCollector.is_document_url(url):
            raise ValueError("URL does not point to a supported document type")
        # download the files and save it to ~/,knwler/documents/ with the filename as the hash of the url
        # return the path to the file
        key = hash_args(url)
        content, metadata = get_cached_document(url)
        if metadata is not None and not no_cache:
            metadata["cached"] = True
            return metadata, content

        async with aiohttp.ClientSession(
            headers=WebpageCollector.DEFAULT_HEADERS
        ) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    metadata = {
                        "id": url,
                        "name": url.split("/")[-1],
                    }
                    metadata = cache_document(url, content, metadata)
                    metadata["cached"] = False
                    return metadata, content
                elif response.status == 404:
                    raise ValueError(f"Document not found: {url}")
                elif response.status == 403:
                    raise ValueError(f"Access forbidden to document: {url}")
                else:
                    raise ValueError(
                        f"Failed to fetch document: {url} with status code {response.status}"
                    )
        return None

    @staticmethod
    async def fetch_url(url: str, no_cache: bool = False) -> tuple[dict, bytes] | None:
        """
        Fetches the content of a URL, which can be either a webpage or a document.
        If the URL points to a supported document type, it fetches the document; otherwise, it fetches the webpage.
        The result is cached on disk keyed by URL.
        """
        if WebpageCollector.is_document_url(url):
            return await WebpageCollector.fetch_document(url, no_cache=no_cache)
        else:
            return await WebpageCollector.fetch_page(url, no_cache=no_cache)

    @staticmethod
    def is_document_url(url: str) -> bool:
        """
        Checks if the URL points to a supported document type based on its extension.
        Supported extensions include .pdf, .doc, .docx, .xls, .xlsx, .ppt, and .pptx.
        """
        return any(url.lower().endswith(ext) for ext in document_extensions)
