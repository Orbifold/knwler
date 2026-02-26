"""
CLI entry point — Typer application.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Annotated, Optional


import networkx as nx
import typer
from rich.panel import Panel
from rich.padding import Padding

from knwler.config import (
    DEFAULT_OLLAMA_EXTRACTION_MODEL,
    DEFAULT_OLLAMA_SCHEMA_MODEL,
    Config,
    ExtractionResult,
    Schema,
    console,
)
from knwler.language import (
    DEFAULT_LANGUAGE,
    get_console_msg,
    get_lang,
    set_language,
    get_current_language,
)
from knwler.cache import CACHE_DIR
from knwler.chunking import chunk_text
from knwler.community import analyze_communities, create_network
from knwler.consolidation import consolidate_graphs
from knwler.discovery import detect_language, discover_schema
from knwler.export import export_html
from knwler.extraction import extract_all
from knwler.extras import extract_summary, extract_title, rephrase_chunks
from knwler.stats import compute_community_stats, compute_stats, print_stats
from knwler.cli_extract import extract_app
from knwler.cli_info import info_app, show_version

app = typer.Typer(
    help="Extract knowledge graphs from text using Ollama or OpenAI.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool) -> None:
    if value:
        show_version()
        raise typer.Exit()


@app.callback()
def _main_callback(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    pass


app.add_typer(
    extract_app,
    name="extract",
    help="Run the extraction pipeline on a file or directory",
)

app.add_typer(
    info_app,
    name="info",
    help="View info about Knwler installation",
)


def main():   
    set_language(DEFAULT_LANGUAGE)
    app()
