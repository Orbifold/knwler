import typer
from rich.console import Console
from rich.panel import Panel
from rich.padding import Padding
from rich.markdown import Markdown
from knwler.config import console
from importlib.metadata import version, PackageNotFoundError

info_app = typer.Typer(help="View info about Knwler installation.")


def get_version():
    try:
        try:
            return version("knwler")
        except PackageNotFoundError:
            return "unknown"
    except Exception:
        return "unknown"


@info_app.command(
    "version",
    help="Show the version of Knwler.",
    epilog="Example:\n  knwler info version",
)
def show_version():
    """Show the version of Knwler."""
    console.print(f"[bold blue]Knwler [/][bold yellow]v{get_version()}[/]")


@info_app.callback(invoke_without_command=True)
def _app_callback(ctx: typer.Context):
    """
    Called in case no subcommand is given, ie. `knwler info`.
    """
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
