import asyncio
import re
import os
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import urlparse

import typer
from knwler.api import fetch_url, parse_file, is_document_url
from knwler.config import console

fetch_app = typer.Typer(
    help="Fetch content from a URL or Wikipedia and save it locally.",
    no_args_is_help=True,
)


@fetch_app.command("url", help="Fetch content from a URL and save it locally.")
def fetch_command(
    url: Annotated[str, typer.Argument(help="The URL to fetch.")],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file or directory path. If a directory, the filename is derived from the URL. Defaults to the current directory.",
        ),
    ] = None,
    parse: Annotated[
        bool,
        typer.Option(
            "--parse",
            "-p",
            help="When fetching a pdf, also parse its text content and save it as a .md file alongside the pdf.",
        ),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache", help="Bypass the cache and fetch directly from the web."
        ),
    ] = False,
    open: Annotated[
        bool,
        typer.Option(
            "--open",
            "-O",
            help="Open the fetched content with the default application after saving.",
        ),
    ] = False,
) -> None:

    try:
        metadata, content = asyncio.run(fetch_url(url, no_cache=no_cache))
    except Exception as e:
        console.print(f"[red]Error fetching URL:[/red] {e}")
        raise typer.Exit(code=1)
    is_document = is_document_url(url)

    filename = metadata.get("name", "output") if metadata else "output"
    if not is_document:
        # sanitize filename for webpage content
        filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
        if not filename.endswith(".md") and not filename.endswith(".txt"):
            filename += ".md"
        if parse:
            console.print(
                "[yellow]\u26a0 Warning[/yellow]: --parse flag is only applicable for documents like PDFs. Ignoring it for webpage content."
            )

    # Resolve the final output path
    if output is None:
        output_path = Path.cwd() / filename
    elif not output.suffix:
        output_path = output / filename
    else:
        output_path = output

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_document:
        output_path.write_bytes(content)
        console.print(f"[green]Saved document to[/green] {output_path}")

        if parse:
            try:
                content, _meta = asyncio.run(parse_file(output_path))
                text_path = output_path.with_suffix(".md")
                text_path.write_text(content, encoding="utf-8")
                console.print(f"[green]Saved parsed text to[/green] {text_path}")
            except Exception as e:
                console.print(f"[red]Error parsing document:[/red] {e}")
                raise typer.Exit(code=1)
    else:
        if isinstance(content, bytes):
            output_path.write_bytes(content)
        else:
            output_path.write_text(content, encoding="utf-8")
        console.print(f"[green]Saved content to[/green] {output_path}")

    if open:
        (
            os.startfile(output_path)
            if os.name == "nt"
            else os.system(f"open '{output_path}'")
        )


@fetch_app.command(
    "wiki", help="Fetch a Wikipedia article by title and save it locally."
)
def wiki_command(
    title: Annotated[
        str, typer.Argument(help="Title of the Wikipedia article to fetch.")
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file or directory path. If a directory, the filename is derived from the article title. Defaults to the current directory.",
        ),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache", help="Bypass the cache and fetch directly from Wikipedia."
        ),
    ] = False,
    open: Annotated[
        bool,
        typer.Option(
            "--open",
            "-O",
            help="Open the fetched content with the default application after saving.",
        ),
    ] = False,
) -> None:
    from knwler.collect.wikipedia import WikipediaCollector

    try:
        doc = asyncio.run(WikipediaCollector.fetch_article(title, no_cache=no_cache))
    except Exception as e:
        console.print(f"[red]Error fetching Wikipedia article:[/red] {e}")
        raise typer.Exit(code=1)

    if doc is None:
        console.print(f"[red]Wikipedia article not found:[/red] {title!r}")
        raise typer.Exit(code=1)

    filename = re.sub(r'[\\/*?:"<>|]', "_", doc["name"]) + ".md"

    if output is None:
        output_path = Path.cwd() / filename
    elif not output.suffix:
        output_path = output / filename
    else:
        output_path = output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc["content"], encoding="utf-8")

    cached_note = " [dim](cached)[/dim]" if doc.get("cached") else ""
    console.print(
        f"[green]Saved Wikipedia article to[/green] {output_path}{cached_note}"
    )
    if open:
        (
            os.startfile(output_path)
            if os.name == "nt"
            else os.system(f"open '{output_path}'")
        )
