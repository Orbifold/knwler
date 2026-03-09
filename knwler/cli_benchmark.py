"""
CLI sub-app for running the benchmark suite.
"""

import asyncio
import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from knwler.config import (
    DEFAULT_ANTHROPIC_DISCOVERY_MODEL,
    DEFAULT_ANTHROPIC_EXTRACTION_MODEL,
    DEFAULT_OLLAMA_DISCOVERY_MODEL,
    DEFAULT_OLLAMA_EXTRACTION_MODEL,
    DEFAULT_OPENAI_DISCOVERY_MODEL,
    DEFAULT_OPENAI_EXTRACTION_MODEL,
    Config,
    console,
)
from benchmark.run import run as benchmark_run

benchmark_app = typer.Typer(help="Run the Knwler benchmark suite.")


@benchmark_app.command("run", help="Run the full benchmark grid and generate a report.")
def run_benchmark(
    backend: Annotated[
        Optional[str],
        typer.Option(
            "--backend",
            "-b",
            help=(
                "Run a specific backend instead of the full grid. Must be one of 'openai', 'anthropic', or 'ollama'."
            ),
        ),
    ] = None,
    extraction_model: Annotated[
        Optional[str],
        typer.Option(
            "--extraction-model",
            help="Override the extraction model for a single custom run (requires --discovery-model and a backend flag).",
        ),
    ] = None,
    discovery_model: Annotated[
        Optional[str],
        typer.Option(
            "--discovery-model",
            help="Override the discovery model for a single custom run (requires --extraction-model and a backend flag).",
        ),
    ] = None,
    no_browser: Annotated[
        bool,
        typer.Option(
            "--no-browser",
            help="Do not open the HTML report in the browser after completion.",
        ),
    ] = False,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-O",
            help="Directory to write benchmark output (defaults to benchmark/output/).",
        ),
    ] = None,
) -> None:
    """Run the benchmark grid and generate an HTML report."""
    try:
        from benchmark.run import crunch, get_path, output_path as default_output_path
        from benchmark.report import generate_report
    except ImportError as exc:
        typer.echo(
            f"[red]Could not import benchmark module: {exc}[/red]\n"
            "Make sure you are running from the project root.",
            err=True,
        )
        raise typer.Exit(1)

    import time

    out_dir = output or default_output_path

    # --- Build the grid ---
    # If no backend flag is given, run all three backends with their defaults.
    custom_run = extraction_model and discovery_model
    if custom_run and not backend:
        typer.echo(
            "[red]Error: --extraction-model and --discovery-model require a --backend to be specified.[/red]"
        )
        raise typer.Exit(1)

    grid: dict[str, list[dict]] = {}

    # Custom single-run override
    if custom_run:

        grid.setdefault(backend, []).append(
            {
                "discovery_model": discovery_model,
                "extraction_model": extraction_model,
            }
        )

    # --- Execute ---
    async def _run() -> None:
        results = []
        if custom_run:
            console.print(
                f"[green]Running single benchmark with config:[/green] {backend} - discovery={discovery_model}, extraction={extraction_model}"
            )
            start_time = time.perf_counter()
            result = await crunch(
                config=Config(
                    backend=backend,
                    discovery_model=discovery_model,
                    extraction_model=extraction_model,
                    use_cache=False,  # for benchmarking we want to disable the cache
                )
            )
            consolidation_time = time.perf_counter() - start_time
            result["consolidation_time"] = round(consolidation_time, 2)
            results.append(result)

        else:
            console.print(f"[green]Running full benchmark grid...[/green]")
            await benchmark_run()

        results_path = out_dir / f"benchmark_results_{int(time.time())}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        console.print(f"[green]Results saved to[/green] {results_path}")

        report_path = out_dir / f"{results_path.stem}_report.html"
        console.print("[cyan]Generating HTML report…[/cyan]")
        generate_report(results, report_path=report_path, open_browser=not no_browser)
        console.print(f"[green]Report → {report_path}[/green]")

    asyncio.run(_run())


@benchmark_app.command(
    "report", help="Generate an HTML report from an existing results JSON file."
)
def report_from_file(
    results_file: Annotated[
        Path,
        typer.Option(
            "--results-file",
            "-f",
            help="Path to the results JSON file from a previous benchmark run.",
        ),
    ],
    report_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-O",
            help="Path to write the HTML report (defaults to <results_file>_report.html).",
        ),
    ] = None,
    no_browser: Annotated[
        bool,
        typer.Option(
            "--no-browser",
            help="Do not open the report in the browser.",
        ),
    ] = False,
) -> None:
    """Re-generate an HTML report from a previously saved results JSON."""
    try:
        from benchmark.report import generate_report
    except ImportError as exc:
        typer.echo(f"Could not import benchmark module: {exc}", err=True)
        raise typer.Exit(1)

    if not results_file.exists():
        typer.echo(f"Results file not found: {results_file}", err=True)
        raise typer.Exit(1)

    results = json.loads(results_file.read_text(encoding="utf-8"))
    out = report_path or results_file.with_name(f"{results_file.stem}_report.html")
    generate_report(results, report_path=out, open_browser=not no_browser)
    console.print(f"[green]Report written to[/green] {out}")
