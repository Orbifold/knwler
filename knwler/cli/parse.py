import typer
from pathlib import Path
from typing import Annotated, Optional

from knwler.config import console
from knwler.parse import load_text

parse_app = typer.Typer(
    help="Parse documents and extract their text content.",
    no_args_is_help=True,
)


@parse_app.command("file", help="Extract text from a PDF, text, or markdown file.")
def parse_file_command(
    file: Annotated[
        Path,
        typer.Argument(help="Path to the file to parse."),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path. Defaults to <stem>.md next to the input file.",
        ),
    ] = None,
) -> None:
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(code=1)

    try:
        text = load_text(file)
    except Exception as e:
        console.print(f"[red]Error parsing file:[/red] {e}")
        raise typer.Exit(code=1)

    if output is None:
        if file.suffix.lower() == ".pdf":
            output = file.with_suffix(".md")
        else:
            console.print(text)
            return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    console.print(f"[green]Saved parsed text to[/green] {output}")
