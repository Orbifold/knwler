#!/usr/bin/env python3
"""
Unified CLI for Knwler Batch Processing.

Dispatches to the OpenAI or Gemini batch processor based on `--backend`.
"""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from knwler.config import (
    DEFAULT_GEMINI_DISCOVERY_MODEL,
    DEFAULT_GEMINI_EXTRACTION_MODEL,
    DEFAULT_OPENAI_DISCOVERY_MODEL,
    DEFAULT_OPENAI_EXTRACTION_MODEL,
)

console = Console()

_DEFAULT_MODELS = {
    "openai": (DEFAULT_OPENAI_DISCOVERY_MODEL, DEFAULT_OPENAI_EXTRACTION_MODEL),
    "gemini": (DEFAULT_GEMINI_DISCOVERY_MODEL, DEFAULT_GEMINI_EXTRACTION_MODEL),
}


def _resolve_models(
    backend: str,
    discovery_model: str | None,
    extraction_model: str | None,
) -> tuple[str, str]:
    """Return (discovery_model, extraction_model) with backend-aware defaults."""
    defaults = _DEFAULT_MODELS.get(backend, _DEFAULT_MODELS["openai"])
    return (discovery_model or defaults[0], extraction_model or defaults[1])


batch_app = typer.Typer(
    help="Process documents via Batch API (OpenAI or Gemini).",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@batch_app.command(
    "run",
    help="Run the full batch pipeline (optionally with cross-file consolidation).",
)
def cmd_run(
    input: Annotated[
        Path,
        typer.Option("--input", "-i", help="Directory containing files to process."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for results."),
    ],
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            "-b",
            help="Batch API backend to use: 'openai' or 'gemini'.",
        ),
    ] = "openai",
    discovery_model: Annotated[
        str | None,
        typer.Option(
            "--discovery-model",
            help="Model for schema discovery. Defaults depend on backend.",
        ),
    ] = None,
    extraction_model: Annotated[
        str | None,
        typer.Option(
            "--extraction-model",
            help="Model for extraction steps. Defaults depend on backend.",
        ),
    ] = None,
    template: Annotated[
        str, typer.Option("--template", help="HTML report template.")
    ] = "default",
    consolidate: Annotated[
        bool,
        typer.Option(
            "--consolidate",
            help="After per-file processing, merge all document graphs into a single "
            "consolidated_graph.json using a fourth batch round.",
        ),
    ] = False,
) -> None:
    """Run the Batch API pipeline (3 rounds + optional cross-file consolidation)."""
    backend = backend.lower()
    if backend not in ("openai", "gemini"):
        console.print(
            f"[red]✗[/red] Unknown backend [cyan]{backend}[/cyan]. "
            "Choose 'openai' or 'gemini'."
        )
        raise typer.Exit(1)
    if not input.is_dir():
        console.print(f"[red]✗[/red] Not a directory: [cyan]{input}[/cyan]")
        raise typer.Exit(1)

    disc, extr = _resolve_models(backend, discovery_model, extraction_model)

    if backend == "gemini":
        from knwler.cli_batch_gemini import GeminiBatchProcessor

        proc = GeminiBatchProcessor(
            input_dir=input,
            output_dir=output,
            discovery_model=disc,
            extraction_model=extr,
            template=template,
            consolidate=consolidate,
        )
        try:
            proc.run()
        finally:
            proc.close()
    else:
        from knwler.cli_batch_openai import BatchProcessor

        proc = BatchProcessor(
            input_dir=input,
            output_dir=output,
            discovery_model=disc,
            extraction_model=extr,
            template=template,
            consolidate=consolidate,
        )
        try:
            asyncio.run(proc.run())
        finally:
            proc.close()


@batch_app.command(
    "consolidate",
    help="Run cross-file consolidation on an already-processed output directory.",
)
def cmd_consolidate(
    input: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Original input directory.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o", help="Output directory produced by a previous 'run'."
        ),
    ],
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            "-b",
            help="Batch API backend to use: 'openai' or 'gemini'.",
        ),
    ] = "openai",
    discovery_model: Annotated[
        str | None,
        typer.Option("--discovery-model", help="Model for schema discovery."),
    ] = None,
    extraction_model: Annotated[
        str | None,
        typer.Option(
            "--extraction-model", help="Model used for summarization requests."
        ),
    ] = None,
    template: Annotated[
        str, typer.Option("--template", help="HTML report template.")
    ] = "default",
) -> None:
    """Merge all per-file graphs from a previous run into consolidated_graph.json."""
    backend = backend.lower()
    if backend not in ("openai", "gemini"):
        console.print(
            f"[red]✗[/red] Unknown backend [cyan]{backend}[/cyan]. "
            "Choose 'openai' or 'gemini'."
        )
        raise typer.Exit(1)
    if not input.is_dir():
        console.print(f"[red]✗[/red] Not a directory: [cyan]{input}[/cyan]")
        raise typer.Exit(1)
    if not output.is_dir():
        console.print(f"[red]✗[/red] Output directory not found: [cyan]{output}[/cyan]")
        raise typer.Exit(1)

    disc, extr = _resolve_models(backend, discovery_model, extraction_model)

    if backend == "gemini":
        from knwler.cli_batch_gemini import GeminiBatchProcessor

        proc = GeminiBatchProcessor(
            input_dir=input,
            output_dir=output,
            discovery_model=disc,
            extraction_model=extr,
            template=template,
            consolidate=True,
        )
        try:
            proc.round4()
        finally:
            proc.close()
    else:
        from knwler.cli_batch_openai import BatchProcessor

        proc = BatchProcessor(
            input_dir=input,
            output_dir=output,
            discovery_model=disc,
            extraction_model=extr,
            template=template,
            consolidate=True,
        )
        try:
            asyncio.run(proc.round4())
        finally:
            proc.close()


@batch_app.command("status", help="Show the current pipeline status.")
def cmd_status(
    input: Annotated[
        Path,
        typer.Option("--input", "-i", help="Directory containing files to process."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory (where batch DB lives)."),
    ],
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            "-b",
            help="Batch API backend to use: 'openai' or 'gemini'.",
        ),
    ] = "openai",
    discovery_model: Annotated[
        str | None,
        typer.Option("--discovery-model", help="Model for schema discovery."),
    ] = None,
    extraction_model: Annotated[
        str | None,
        typer.Option("--extraction-model", help="Model for extraction steps."),
    ] = None,
) -> None:
    """Show current pipeline status for an existing output directory."""
    backend = backend.lower()
    if backend not in ("openai", "gemini"):
        console.print(
            f"[red]✗[/red] Unknown backend [cyan]{backend}[/cyan]. "
            "Choose 'openai' or 'gemini'."
        )
        raise typer.Exit(1)
    if not input.is_dir():
        console.print(f"[red]✗[/red] Not a directory: [cyan]{input}[/cyan]")
        raise typer.Exit(1)

    disc, extr = _resolve_models(backend, discovery_model, extraction_model)

    if backend == "gemini":
        from knwler.cli_batch_gemini import GeminiBatchProcessor

        proc = GeminiBatchProcessor(
            input_dir=input,
            output_dir=output,
            discovery_model=disc,
            extraction_model=extr,
        )
        try:
            proc.print_status()
        finally:
            proc.close()
    else:
        from knwler.cli_batch_openai import BatchProcessor

        proc = BatchProcessor(
            input_dir=input,
            output_dir=output,
            discovery_model=disc,
            extraction_model=extr,
        )
        try:
            proc.print_status()
        finally:
            proc.close()
