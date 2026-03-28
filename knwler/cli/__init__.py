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
    DEFAULT_OPENAI_EXTRACTION_MODEL,
    DEFAULT_OPENAI_DISCOVERY_MODEL,
    DEFAULT_ANTHROPIC_EXTRACTION_MODEL,
    DEFAULT_ANTHROPIC_DISCOVERY_MODEL,
    DEFAULT_GITHUB_EXTRACTION_MODEL,
    DEFAULT_GITHUB_DISCOVERY_MODEL,
    DEFAULT_LMSTUDIO_EXTRACTION_MODEL,
    DEFAULT_LMSTUDIO_DISCOVERY_MODEL,
    DEFAULT_GEMINI_EXTRACTION_MODEL,
    DEFAULT_GEMINI_DISCOVERY_MODEL,
    Config,
    console,
)
from knwler.models import ChunkGraph, Schema, Graph

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
from knwler.consolidation import consolidate_chunk_graphs
from knwler.discovery import detect_language, discover_schema
from knwler.export import export_html
from knwler.extraction import extract_chunks
from knwler.extras import extract_summary, extract_title, rephrase_chunks
from knwler.stats import compute_cluster_stats, compute_stats, print_stats
from knwler.cli.extract import extract_app
from knwler.cli.info import info_app, show_version
from knwler.cli.consolidate import cli_consolidate_graphs
from knwler.cli.fetch import fetch_app
from knwler.cli.cache import cache_app
from knwler.cli.demo import demo_app
from knwler.cli.benchmark import benchmark_app
from knwler.cli.graph import graph_app
from knwler.cli.batch import batch_app
from knwler.cli.parse import parse_app
from knwler.cli.render import render_command
from knwler.cli.export import export_app

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
    github: Annotated[
        bool,
        typer.Option(
            "--github",
            help="Use GitHub Models API for consolidation (requires GITHUB_TOKEN env var). Experimental.",
        ),
    ] = False,
    lmstudio: Annotated[
        bool,
        typer.Option(
            "--lmstudio",
            help="Use LM Studio (local OpenAI-compatible server at http://localhost:1234/v1).",
        ),
    ] = False,
    gemini: Annotated[
        bool,
        typer.Option(
            "--gemini",
            help="Use Google Gemini API for consolidation (requires GEMINI_API_KEY env var).",
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
    backend_flags = sum([openai, anthropic, github, lmstudio, gemini])
    if backend_flags > 1:
        console.print(
            "[red]\u2717[/red] [bold red]Error:[/bold red] --openai, --anthropic, --github, --lmstudio, and --gemini are mutually exclusive."
        )
        raise typer.Exit(1)
    backend = (
        "openai"
        if openai
        else (
            "anthropic"
            if anthropic
            else (
                "github"
                if github
                else ("lmstudio" if lmstudio else ("gemini" if gemini else "ollama"))
            )
        )
    )
    resolved_extraction = extraction_model or (
        DEFAULT_OPENAI_EXTRACTION_MODEL
        if openai
        else (
            DEFAULT_ANTHROPIC_EXTRACTION_MODEL
            if anthropic
            else (
                DEFAULT_GITHUB_EXTRACTION_MODEL
                if github
                else (
                    DEFAULT_LMSTUDIO_EXTRACTION_MODEL
                    if lmstudio
                    else (
                        DEFAULT_GEMINI_EXTRACTION_MODEL
                        if gemini
                        else DEFAULT_OLLAMA_EXTRACTION_MODEL
                    )
                )
            )
        )
    )
    resolved_discovery = discovery_model or (
        DEFAULT_OPENAI_DISCOVERY_MODEL
        if openai
        else (
            DEFAULT_ANTHROPIC_DISCOVERY_MODEL
            if anthropic
            else (
                DEFAULT_GITHUB_DISCOVERY_MODEL
                if github
                else (
                    DEFAULT_LMSTUDIO_DISCOVERY_MODEL
                    if lmstudio
                    else (
                        DEFAULT_GEMINI_DISCOVERY_MODEL
                        if gemini
                        else DEFAULT_OLLAMA_DISCOVERY_MODEL
                    )
                )
            )
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
app.add_typer(cache_app, name="cache", help="View and manage Knwler cache")
app.add_typer(demo_app, name="demo", help="Demo commands for Knwler")

app.add_typer(
    fetch_app,
    name="fetch",
    help="Fetch content from a URL or Wikipedia and save it locally",
)

app.add_typer(
    benchmark_app,
    name="benchmark",
    help="Run the Knwler benchmark suite",
)

app.add_typer(
    graph_app,
    name="graph",
    help="Convert and analyse graph.json files",
)

app.add_typer(batch_app, name="batch", help="Run batch API pipeline (OpenAI / Gemini).")
app.add_typer(
    parse_app, name="parse", help="Parse documents and extract their text content."
)
app.command("render", help="Render a graph.json or all_data.json to HTML.")(
    render_command
)
app.add_typer(
    export_app, name="export", help="Export graph data to Neo4j, SurrealDB, or JSON-LD."
)


def main():
    set_language(DEFAULT_LANGUAGE)

    _KNOWN_SUBCOMMANDS = {
        "extract",
        "info",
        "consolidate",
        "fetch",
        "cache",
        "demo",
        "benchmark",
        "graph",
        "batch",
        "render",
        "export",
    }
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
    # app()
    try:
        app()
    except typer.Exit:
        pass
    except Exception as e:
        console.print(Panel.fit(f"[red]Error:[/red] {str(e)}"))
        sys.exit(1)
