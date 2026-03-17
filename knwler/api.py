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
from knwler.collect.webpage import WebpageCollector
from knwler.config import Config
from knwler.chunking import chunk_text
from knwler.extraction import extract_all
from knwler.discovery import discover_schema, detect_language
from knwler.models import *
from knwler.consolidation import consolidate_extracted_graphs
from knwler.clustering import cluster_graph as clustering
from knwler.language import set_language as _set_language
from knwler.extras import (
    extract_summary as _extract_summary,
    extract_title as _extract_title,
    rephrase_chunks as _rephrase_chunks,
)
from knwler.export import export_html


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

    return await WebpageCollector.fetch_url(url, no_cache=no_cache)


def is_document_url(url: str) -> bool:
    """
    Checks if the URL points to a supported document type based on its extension.
    Supported extensions include .pdf, .doc, .docx, .xls, .xlsx, .ppt, and .pptx.
    """
    return WebpageCollector.is_document_url(url)


async def extract_file(
    file_path: Path, output: Path | None = None, config: Config | None = Config()
) -> list[ExtractionResult]:
    if not file_path:
        raise ValueError("No file path provided.")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    # if not md or txt, parse  first
    if file_path.suffix.lower() not in [".txt", ".md"]:
        text, metadata = await parse_file(file_path)
    else:
        text = file_path.read_text(encoding="utf-8")
    if output is None:
        output_path = Path.cwd() / "knwler_output"
    else:
        output_path = output
    output_path.mkdir(parents=True, exist_ok=True)
    chunks = chunk_text(text, config)
    schema = await discover_schema(text, config)
    extraction_results = await extract_all(chunks, schema, config, output_path)
    return extraction_results


async def extract_chunks(
    chunks: list[str], schema: Schema | None = None, config: Config | None = Config()
) -> list[ExtractionResult]:
    if schema is None:
        schema = await discover_schema(" ".join(chunks), config)
    extraction_results = await extract_all(chunks, schema, config)
    return extraction_results


async def infer_schema(text: str, config: Config | None = Config()) -> Schema:
    return await discover_schema(text, config)


async def chunk(text: str, config: Config | None = Config()) -> list[str]:
    return chunk_text(text, config)


async def consolidate(
    extraction_results: list[ExtractionResult],
    summarize: bool = True,
    filter_low_importance: bool = True,
    config: Config | None = Config(),
) -> Graph:
    consolidation, timing = await consolidate_extracted_graphs(
        extraction_results,
        config=config,
        summarize=summarize,
        filter_low_importance=filter_low_importance,
    )
    return Graph(**consolidation)


async def cluster_graph(
    graph: Graph, config: Config | None = Config()
) -> ClusteredGraph:
    """
    Analyze the communities in the graph and return statistics about them.
    This can be used to identify clusters of related entities and relations in the graph, which can be useful for understanding the structure of the knowledge and for downstream applications like question answering or recommendation.
    """
    clustered_graph = await clustering(graph, config)
    return clustered_graph


async def set_language(lang_code: str):
    """Set the language for prompts and UI messages. This will affect all subsequent operations that involve language generation or UI output."""
    return  _set_language(lang_code)


async def discover_language(text: str, config: Config | None = Config()) -> str:
    """Detect the language of the given text. This can be used to automatically set the language for prompts and UI messages."""
    found = await detect_language(text, config)
    await set_language(found)
    return found


async def extract_summary(text: str, config: Config | None = Config()) -> str:
    """Extract a concise summary of the given text. This can be used to quickly understand the main points of a document or to generate a brief description for display purposes."""
    return await _extract_summary(text, config)


async def extract_title(chunks: list[str], config: Config | None = Config()) -> str:
    """
    Extract a title for the given chunks.
    This can be used to generate a title for a document or to assign a name to a graph.
    The extraction is not based on the whole text since the title is usually mentioned in the beginning, so we can save tokens by only looking at the first few chunks.
    If you really want to use the whole text simply use `[text]`.
    """
    return await _extract_title(chunks, config, max_chunks=3)


async def render_html(graph: KnowledgeGraph, template: str = "default") -> str:
    """
    Render the given graph as an HTML string. This can be used to display the graph in a web application or to export it for sharing.
    The rendering includes a visualization of the graph structure as well as a summary and title for context.
    """
    return export_html(graph, template=template)


async def rephrase_chunks(
    chunks: list[str], config: Config | None = Config()
) -> list[str]:
    """Rephrase each chunk in simple language for UI display. This can be used to make the content more accessible and easier to understand for a wider audience."""
    return await _rephrase_chunks(chunks, config)
