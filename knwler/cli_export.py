"""
CLI sub-app for exporting graph data to external targets (Neo4j, SurrealDB, JSON-LD).

Each subcommand delegates to the matching integration module.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer

export_app = typer.Typer(add_completion=False, rich_markup_mode="rich")


# ── Neo4j ──────────────────────────────────────────────────────────────────────


@export_app.command("neo4j")
def neo4j(
    files: Annotated[
        list[Path],
        typer.Argument(help="JSON graph files to import into Neo4j."),
    ],
    uri: Annotated[
        str,
        typer.Option("--uri", "-u", help="Neo4j bolt URI."),
    ] = "bolt://localhost:7687",
    user: Annotated[
        str,
        typer.Option("--user", help="Neo4j username."),
    ] = "neo4j",
    password: Annotated[
        str,
        typer.Option("--password", "-p", help="Neo4j password."),
    ] = "123456789",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Cypher batch size."),
    ] = 500,
) -> None:
    """Import knowledge-graph JSON files into Neo4j."""
    from knwler.integrations.neo4j_import import main as neo4j_main

    neo4j_main(
        files=files, uri=uri, user=user, password=password, batch_size=batch_size
    )


# ── SurrealDB ─────────────────────────────────────────────────────────────────


@export_app.command("surrealdb")
def surrealdb(
    files: Annotated[
        list[Path],
        typer.Argument(help="JSON graph files to import into SurrealDB."),
    ],
    url: Annotated[
        str,
        typer.Option("--url", help="SurrealDB WebSocket URL."),
    ] = "ws://localhost:8000/rpc",
    user: Annotated[
        str,
        typer.Option("--user", "-u", help="SurrealDB username."),
    ] = "root",
    password: Annotated[
        str,
        typer.Option("--password", "-p", help="SurrealDB password."),
    ] = "root",
    namespace: Annotated[
        str,
        typer.Option("--ns", help="SurrealDB namespace."),
    ] = "knwler",
    database: Annotated[
        str,
        typer.Option("--db", help="SurrealDB database."),
    ] = "knwler",
) -> None:
    """Import knowledge-graph JSON files into SurrealDB."""
    from knwler.integrations.surreal_import import main as surreal_main

    surreal_main(
        files=files,
        url=url,
        user=user,
        password=password,
        namespace=namespace,
        database=database,
    )


# ── JSON-LD ───────────────────────────────────────────────────────────────────


@export_app.command("jsonld")
def jsonld(
    files: Annotated[
        list[Path],
        typer.Argument(help="JSON graph files to export as JSON-LD."),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o", help="Output directory (default: same as input file)."
        ),
    ] = None,
    pretty: Annotated[
        bool,
        typer.Option("--pretty/--compact", help="Pretty-print JSON-LD output."),
    ] = True,
) -> None:
    """Export knowledge-graph JSON files to JSON-LD (RDF)."""
    from knwler.integrations.jsonld_export import main as jsonld_main

    jsonld_main(files=files, output=output, pretty=pretty)
