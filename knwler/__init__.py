"""
knwler — Fast and accurate graph extraction from text using LLMs.
"""

from knwler.config import Config, console
from knwler.models import ExtractionResult, Schema, Graph
from knwler.language import set_language, get_lang, get_prompt, get_ui, get_console_msg
from knwler.cache import CACHE_DIR
from knwler.llm import llm_generate, parse_json_response
from knwler.chunking import chunk_text, get_encoder
from knwler.discovery import discover_schema, detect_language
from knwler.extraction import extract_graph, extract_chunk, extract_all
from knwler.consolidation import consolidate_extracted_graphs, consolidate_graphs
from knwler.clustering import cluster_graph, create_network
from knwler.export import export_html
from knwler.stats import compute_stats, compute_community_stats, print_stats
from knwler.extras import rephrase_chunks, extract_title, extract_summary
from knwler.cli import app

__all__ = [
    # Core types
    "Config",
    "ExtractionResult",
    "Schema",
    "console",
    # Language
    "set_language",
    "get_lang",
    "get_prompt",
    "get_ui",
    "get_console_msg",
    # Cache
    "CACHE_DIR",
    # LLM
    "llm_generate",
    "parse_json_response",
    # Chunking
    "chunk_text",
    "get_encoder",
    # Discovery
    "discover_schema",
    "detect_language",
    # Extraction
    "extract_graph",
    "extract_chunk",
    "extract_all",
    # Consolidation
    "consolidate_extracted_graphs",
    "consolidate_graphs",
    # Community
    "analyze_communities",
    "create_network",
    # Export
    "export_html",
    # Stats
    "compute_stats",
    "compute_community_stats",
    "print_stats",
    # Extras
    "rephrase_chunks",
    "extract_title",
    "extract_summary",
    # models
    "Graph",
    # CLI
    "app",
]
