import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.padding import Padding
from rich.markdown import Markdown
from knwler.config import console
from importlib.metadata import version, PackageNotFoundError
from knwler.cache import *

cache_app = typer.Typer(help="View info about Knwler installation.")


@cache_app.command(
    "clear",
    help="Clears the cache of Knwler. Optionally specify what to clear: documents, llm, wikipedia, webpages. Clears all if omitted.",
    epilog="Examples:\n  knwler cache clear\n  knwler cache clear llm\n  knwler cache clear wikipedia",
)
def clear_cache(
    what: Optional[str] = typer.Argument(
        None,
        help="What to clear: documents, llm, wikipedia, webpages. Clears all if omitted.",
    )
):
    """Clear the cache of Knwler."""
    options = {"documents", "llm", "wikipedia", "webpages"}
    if what is None:
        clear_all_cache()
        console.print("[bold green]All caches cleared successfully![/]")
    elif what == "documents":
        clear_document_cache()
        console.print("[bold green]Documents cache cleared successfully![/]")
    elif what == "llm":
        clear_llm_cache()
        console.print("[bold green]LLM cache cleared successfully![/]")
    elif what == "wikipedia":
        clear_wikipedia_cache()
        console.print("[bold green]Wikipedia cache cleared successfully![/]")
    elif what == "webpages":
        clear_webpage_cache()
        console.print("[bold green]Webpages cache cleared successfully![/]")
    else:
        console.print(
            f"[bold red]Unknown option '{what}'. Choose from: {', '.join(sorted(options))}.[/]"
        )
        raise typer.Exit(code=1)


@cache_app.callback(invoke_without_command=True)
def _app_callback(ctx: typer.Context):
    """
    Called in case no subcommand is given, ie. `knwler cache`.
    """
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
