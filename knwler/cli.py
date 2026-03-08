"""
CLI entry point — Typer application.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Optional


import networkx as nx
import typer
from rich.panel import Panel
from rich.padding import Padding

from knwler.config import (
    DEFAULT_OLLAMA_EXTRACTION_MODEL,
    DEFAULT_OLLAMA_DISCOVERY_MODEL,
    Config,
    console,
)
from knwler.models import ExtractionResult, Schema, Graph

from knwler.language import (
    DEFAULT_LANGUAGE,
    get_console_msg,
    get_lang,
    set_language,
    get_current_language,
)
from knwler.cache import CACHE_DIR
from knwler.chunking import chunk_text
from knwler.clustering import cluster_graph, create_network
from knwler.consolidation import consolidate_extracted_graphs
from knwler.discovery import detect_language, discover_schema
from knwler.export import export_html
from knwler.extraction import extract_all
from knwler.extras import extract_summary, extract_title, rephrase_chunks
from knwler.stats import compute_community_stats, compute_stats, print_stats
from knwler.cli_extract import extract_app
from knwler.cli_info import info_app, show_version
from knwler.cli_consolidate import cli_consolidate_graphs
from knwler.cli_fetch import fetch_app
from knwler.cli_cache import cache_app
from knwler.cli_demo import demo_app

app = typer.Typer(
    help="Turn documents into structured knowledge.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.command("consolidate", help="Consolidate extracted graphs into a single graph.")
def consolidate_graphs_command(
    directory: Annotated[
        Optional[Path],
        typer.Option(
            "--dir",
            "-D",
            help="Path to a directory containing graph JSON files to consolidate (defaults to 'results/'). The command will search recursively for all 'graph.json' files in the specified directory and its subdirectories.",
        ),
    ] = None,
    openai: Annotated[
        bool,
        typer.Option(
            "--openai",
            help="Use OpenAI models for consolidation (overrides --extraction-model and --discovery-model).",
        ),
    ] = False,
    anthropic: Annotated[
        bool,
        typer.Option(
            "--anthropic",
            help="Use Anthropic models for consolidation (overrides --extraction-model and --discovery-model).",
        ),
    ] = False,
    extraction_model: Annotated[
        Optional[str],
        typer.Option(
            "--extraction-model",
            help="Model to use for extraction during consolidation (overrides --openai and --anthropic).",
        ),
    ] = None,
    discovery_model: Annotated[
        Optional[str],
        typer.Option(
            "--discovery-model",
            help="Model to use for discovery during consolidation (overrides --openai and --anthropic).",
        ),
    ] = None,
    concurrent: Annotated[
        int,
        typer.Option(
            "--concurrent",
            "-C",
            help="Maximum number of concurrent API calls during consolidation.",
        ),
    ] = 5,
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            help="Maximum number of tokens to use in model responses during consolidation.",
        ),
    ] = 2048,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Disable caching of model responses during consolidation.",
        ),
    ] = False,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            "--base-url",
            help="Base URL for the API (useful for OpenAI-compatible APIs like Azure OpenAI).",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-O",
            help="Directory to save the consolidated graph (defaults to 'results/').",
        ),
    ] = None,
    include_chunks: Annotated[
        bool,
        typer.Option(
            "--include-chunks",
            help="Whether to include chunks in the consolidation process (useful when merging towards a vector database ingestion).",
        ),
    ] = False,
):
    """Consolidate extracted graphs into a single graph."""
    if openai and anthropic:
        typer.echo("Error: --openai and --anthropic are mutually exclusive.")
        return typer.Exit(1)
    backend = "openai" if openai else ("anthropic" if anthropic else "ollama")
    resolved_extraction = extraction_model or (
        DEFAULT_OPENAI_EXTRACTION_MODEL
        if openai
        else (
            DEFAULT_ANTHROPIC_EXTRACTION_MODEL
            if anthropic
            else DEFAULT_OLLAMA_EXTRACTION_MODEL
        )
    )
    resolved_discovery = discovery_model or (
        DEFAULT_OPENAI_DISCOVERY_MODEL
        if openai
        else (
            DEFAULT_ANTHROPIC_DISCOVERY_MODEL
            if anthropic
            else DEFAULT_OLLAMA_DISCOVERY_MODEL
        )
    )
    config = Config(
        backend=backend,
        extraction_model=resolved_extraction,
        discovery_model=resolved_discovery,
        max_concurrent=concurrent,
        max_tokens=max_tokens,
        use_cache=not no_cache,
        base_url=base_url,
    )
    asyncio.run(
        cli_consolidate_graphs(
            directory=directory,
            include_chunks=include_chunks,
            output=output,
            config=config,
        )
    )


def _version_callback(value: bool) -> None:
    if value:
        show_version()
        return typer.Exit()


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
app.add_typer(cache_app, name="cache", help="View and manage Knwler cache")
app.add_typer(demo_app, name="demo", help="Demo commands for Knwler")

app.add_typer(
    fetch_app,
    name="fetch",
    help="Fetch content from a URL or Wikipedia and save it locally",
)


def main():
    set_language(DEFAULT_LANGUAGE)

    _KNOWN_SUBCOMMANDS = {"extract", "info", "consolidate", "fetch"}
    _GLOBAL_FLAGS = {"--version", "-V", "--help", "-h"}
    args = sys.argv[1:]
    if (
        args
        and args[0] == "extract"
        and (len(args) == 1 or args[1].startswith("-"))
        and (len(args) == 1 or args[1] != "extract")
    ):
        # 'knwler extract [-f ...]' → 'knwler extract extract [-f ...]'
        sys.argv.insert(2, "extract")
    elif (
        args
        and args[0] not in _KNOWN_SUBCOMMANDS
        and args[0] not in _GLOBAL_FLAGS
        and args[0].startswith("-")
    ):
        # 'knwler -f ...' → 'knwler extract extract -f ...'
        sys.argv.insert(1, "extract")
        sys.argv.insert(2, "extract")

    app()
