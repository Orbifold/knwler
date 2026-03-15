"""
Knwler Graph Extraction Pipeline — thin wrapper around the knwler package.

All functionality has been moved into the ``knwler`` package.
This file is kept for backwards compatibility so that
``uv run main.py`` continues to work.
Things like `uv run main.py extract --help` are also rewired to work, and will show the help for the `extract` subcommand.
"""

import sys
from knwler.cli import app
from knwler.config import console
from rich.panel import Panel
from rich.padding import Padding
import typer

_KNOWN_SUBCOMMANDS = {"extract", "info", "consolidate", "fetch"}

if __name__ == "__main__":
    # Route bare flags (e.g. -f, --file) to the default 'extract extract' subcommand
    # so that these three forms are all equivalent:
    #   main.py -f file.pdf
    #   main.py extract -f file.pdf
    #   main.py extract extract -f file.pdf
    _GLOBAL_FLAGS = {"--version", "-V", "--help", "-h"}
    args = sys.argv[1:]
    if (
        args
        and args[0] == "extract"
        and (len(args) == 1 or args[1].startswith("-"))
        and (len(args) == 1 or args[1] != "extract")
    ):
        # 'extract [-f ...]' / 'extract --help' → 'extract extract [-f ...] / --help'
        sys.argv.insert(2, "extract")
    elif (
        args
        and args[0] not in _KNOWN_SUBCOMMANDS
        and args[0] not in _GLOBAL_FLAGS
        and args[0].startswith("-")
    ):
        # '-f ...' → 'extract extract -f ...'
        sys.argv.insert(1, "extract")
        sys.argv.insert(2, "extract")

    app()

    # try:
    #     app()
    # except typer.Exit:
    #     # Prevent typer from printing "Error: No command specified." when no subcommand is given,
    #     # since we route that case to the default demo.
    #     pass
    # except Exception as e:
    #     console.print(Panel.fit(f"[red]Error:[/red] {str(e)}"))
    #     sys.exit(1)
