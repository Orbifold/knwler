# ================================================================================================
# This is a wrapper around the various other modules in order to make Knwler
# as easy as possible to use in downstream applications. It provides a single entry point
# for parsing files, consolidating knowledge, and generating responses.
# In prticular, this does not render any Rich/Typer output but rather returns raw data strcutures.
# ================================================================================================
from pathlib import Path
import time
import json
import fitz  # PyMuPDF


async def parse_file(file_path: Path) -> tuple[str, dict]:
    """
    Parses the given file and returns its text content along with metadata such as file size and parsing time.
    - Supported file types include .txt, .md, and .pdf. For unsupported file types, a ValueError is raised.
    - If the file does not exist, a FileNotFoundError is raised.
    - The parsing time is measured and included in the metadata for performance monitoring.
    """
    if not file_path:
        raise ValueError("No file path provided.")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    start_time = time.perf_counter()
    if file_path.suffix.lower() in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        end_time = time.perf_counter()
        return text, {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "time": round(end_time - start_time, 2),
        }
    elif file_path.suffix.lower() in [".pdf"]:
        doc = fitz.open(file_path)
        text = "\n\n".join(page.get_text() for page in doc)
        end_time = time.perf_counter()
        return text, {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "time": round(end_time - start_time, 2),
        }
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


async def fetch_url(url: str, no_cache: bool = False) -> tuple[dict, bytes] | None:
    """
    Fetches the webpage content and metadata for the given URL. The result is cached on disk keyed by URL.
    - If no_cache is True, the cache is bypassed and the content is fetched directly from the web.
    - The metadata includes the URL, page title, and a description. If the URL points to a document (e.g. PDF), the content is returned as bytes along with metadata such as filename and extension.
    - If the URL points to a webpage, the content is returned as text along with metadata.
    - If the URL is not accessible or returns an error status code, a ValueError is raised with an appropriate message. If the URL is not valid, a ValueError is raised.
    """
    from knwler.collect.webpage import WebpageCollector

    return await WebpageCollector.fetch_url(url, no_cache=no_cache)

