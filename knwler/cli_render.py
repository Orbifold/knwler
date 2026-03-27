"""
CLI command for rendering an existing graph.json (or all_data.json) to HTML.
"""

import json
import os
import webbrowser
from pathlib import Path
from typing import Annotated, Optional

import typer

from knwler.config import console
from knwler.export import export_html

_TEMPLATES = ["default", "columns", "research", "cypher", "blank"]


def render_command(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Path to a graph.json file to render.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help=f"Template to use for rendering. One of: {', '.join(_TEMPLATES)}.",
        ),
    ] = "default",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Path for the output HTML file. Defaults to index.html next to the input file.",
        ),
    ] = None,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open/--no-open",
            help="Open the rendered HTML in the default browser after saving.",
        ),
    ] = True,
) -> None:
    """
    Render an existing graph.json or all_data.json to an HTML report.
    """
    if template not in _TEMPLATES:
        console.print(
            f"[red]Unknown template '[bold]{template}[/bold]'. "
            f"Choose one of: {', '.join(_TEMPLATES)}[/red]"
        )
        raise typer.Exit(1)

    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        console.print(f"[red]Failed to parse JSON from {input_file}:[/red] {exc}")
        raise typer.Exit(1)

    html_path = output or input_file.parent / "index.html"

    html_content = export_html(data, template=template)
    html_path.write_text(html_content, encoding="utf-8")
    console.print(
        f"[green]✔ HTML rendered with template '[bold]{template}[/bold]' → {html_path}[/green]"
    )

    if open_browser:
        if os.name == "nt":
            os.startfile(html_path)  # type: ignore[attr-defined]
        else:
            webbrowser.open(html_path.resolve().as_uri())
