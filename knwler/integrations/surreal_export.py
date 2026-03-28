#!/usr/bin/env python3
"""
Import knowledge-graph JSON files into SurrealDB (v3.0).

Usage:
    uv run surrealdb_import.py graph.json [graph2.json ...]

Requires:
    uv add surrealdb typer rich

Data model:
    document  ──has_chunk──>      chunk
    document  ──has_cluster──>  cluster
    document  ──mentions──>       entity
    entity    ──extracted_from──> chunk
    entity    ──belongs_to──>     cluster
    entity    ──<dynamic_type>──> entity      # knowledge-graph relations

SurrealDB leverages multi-model storage: each record type is a table,
graph edges are TYPE RELATION tables. Entities that appear across
multiple documents are merged by name (longer description wins).

To start SurrealDB locally with a RocksDB store, run:
    surreal start --user root --pass root rocksdb:~/surreal/

To see the results in Surrealist, connect with root/root and select the
ns/db via the menu at the top. Via the SQL CLI:
    surreal sql -u root -p root
"""
import json
import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from surrealdb import RecordID, Surreal

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

BATCH_SIZE = 500


def sanitize_id(s: str) -> str:
    """Turn an arbitrary string into a safe SurrealDB record ID part."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", s).strip("_")[:200]


def sanitize_table(s: str) -> str:
    """Turn a relation type into a valid SurrealDB table name."""
    t = re.sub(r"[^a-zA-Z0-9_]", "_", s).lower().strip("_")
    if t and t[0].isdigit():
        t = "_" + t
    return t


def doc_rid(url: str) -> RecordID:
    return RecordID("document", sanitize_id(url))


def chunk_rid(url: str, idx: int) -> RecordID:
    return RecordID("chunk", f"{sanitize_id(url)}__{idx}")


def cluster_rid(url: str, idx: int) -> RecordID:
    return RecordID("cluster", f"{sanitize_id(url)}__{idx}")


def entity_rid(name: str) -> RecordID:
    return RecordID("entity", sanitize_id(name))


def define_schema(db):
    """Define tables and indexes."""
    queries = [
        # Record tables
        "DEFINE TABLE IF NOT EXISTS document SCHEMAFULL",
        "DEFINE FIELD IF NOT EXISTS url ON document TYPE string",
        "DEFINE FIELD IF NOT EXISTS title ON document TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS summary ON document TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS language ON document TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS entity_types ON document TYPE option<array>",
        "DEFINE FIELD IF NOT EXISTS relation_types ON document TYPE option<array>",
        "DEFINE FIELD IF NOT EXISTS schema_reasoning ON document TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS num_chunks ON document TYPE option<int>",
        "DEFINE FIELD IF NOT EXISTS extraction_model ON document TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS discovery_model ON document TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS total_time ON document TYPE option<float>",
        "DEFINE FIELD IF NOT EXISTS timestamp ON document TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS source_file ON document TYPE option<string>",
        "DEFINE INDEX IF NOT EXISTS doc_url_idx ON document FIELDS url UNIQUE",
        "DEFINE TABLE IF NOT EXISTS chunk SCHEMAFULL",
        "DEFINE FIELD IF NOT EXISTS chunk_idx ON chunk TYPE int",
        "DEFINE FIELD IF NOT EXISTS text ON chunk TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS rephrase ON chunk TYPE option<string>",
        "DEFINE INDEX IF NOT EXISTS chunk_idx_idx ON chunk FIELDS chunk_idx",
        "DEFINE TABLE IF NOT EXISTS cluster SCHEMAFULL",
        "DEFINE FIELD IF NOT EXISTS cluster_idx ON cluster TYPE int",
        "DEFINE FIELD IF NOT EXISTS topics ON cluster TYPE option<array>",
        "DEFINE FIELD IF NOT EXISTS description ON cluster TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS members ON cluster TYPE option<array>",
        "DEFINE TABLE IF NOT EXISTS entity SCHEMALESS",
        "DEFINE FIELD IF NOT EXISTS name ON entity TYPE string",
        "DEFINE FIELD IF NOT EXISTS type ON entity TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS description ON entity TYPE option<string>",
        "DEFINE INDEX IF NOT EXISTS entity_name_idx ON entity FIELDS name UNIQUE",
        "DEFINE INDEX IF NOT EXISTS entity_type_idx ON entity FIELDS type",
        # Structural relation tables
        "DEFINE TABLE IF NOT EXISTS has_chunk TYPE RELATION IN document OUT chunk",
        "DEFINE TABLE IF NOT EXISTS cluster TYPE RELATION IN document OUT cluster",
        "DEFINE TABLE IF NOT EXISTS mentions TYPE RELATION IN document OUT entity",
        "DEFINE TABLE IF NOT EXISTS extracted_from TYPE RELATION IN entity OUT chunk",
        "DEFINE TABLE IF NOT EXISTS belongs_to TYPE RELATION IN entity OUT cluster",
    ]
    for q in queries:
        db.query(q)


def import_document(db, doc: dict) -> None:
    """Create a single document record from a full extraction JSON."""
    stats = doc["stats"][0] if doc.get("stats") else {}
    run_info = stats.get("run", {})

    db.upsert(
        doc_rid(doc["url"]),
        {
            "url": doc["url"],
            "title": doc.get("title"),
            "summary": doc.get("summary"),
            "language": doc.get("language"),
            "entity_types": doc.get("schema", {}).get("entity_types", []),
            "relation_types": doc.get("schema", {}).get("relation_types", []),
            "schema_reasoning": doc.get("schema", {}).get("reasoning"),
            "num_chunks": stats.get("num_chunks"),
            "extraction_model": run_info.get("extraction_model"),
            "discovery_model": run_info.get("discovery_model"),
            "total_time": stats.get("total_time"),
            "timestamp": run_info.get("timestamp"),
            "source_file": run_info.get("file"),
        },
    )


def import_documents(db, documents: list) -> None:
    """Create document records from a consolidated JSON's documents list."""
    for d in documents:
        url = d.get("url") or d.get("id", "")
        db.upsert(
            doc_rid(url),
            {
                "url": url,
                "title": d.get("title"),
                "summary": d.get("summary"),
                "language": d.get("language"),
            },
        )


def import_chunks(db, doc_url: str, chunks: list, progress: Progress, task) -> None:
    """Create chunk records and relate them to the document."""
    d = doc_rid(doc_url)
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        for c in batch:
            cr = chunk_rid(doc_url, c["chunk_idx"])
            db.upsert(
                cr,
                {
                    "chunk_idx": c["chunk_idx"],
                    "text": c.get("text"),
                    "rephrase": c.get("rephrase"),
                },
            )
            db.query("RELATE $from->has_chunk->$to", {"from": d, "to": cr})
            progress.advance(task)


def import_clusters(db, doc_url: str, clusters: list, progress: Progress, task) -> None:
    """Create cluster records and relate them to the document."""
    d = doc_rid(doc_url)
    for comm in clusters:
        cr = cluster_rid(doc_url, comm["id"])
        db.upsert(
            cr,
            {
                "cluster_idx": comm["id"],
                "topics": comm.get("topics", []),
                "description": comm.get("description"),
                "members": comm.get("members", []),
            },
        )
        db.query("RELATE $from->cluster->$to", {"from": d, "to": cr})
        progress.advance(task)


def import_entities(db, doc_url: str, entities: list, progress: Progress, task) -> None:
    """Create entity records, link to document, chunks, and clusters."""
    d = doc_rid(doc_url)
    for ent in entities:
        er = entity_rid(ent["name"])
        db.query(
            """
            UPSERT $rid MERGE {
                name: $name,
                type: type ?? $type,
                description: IF (
                    $description IS NOT NONE
                    AND string::len($description) > string::len(description ?? '')
                ) THEN $description ELSE description ?? $description END
            }
            """,
            {
                "rid": er,
                "name": ent["name"],
                "type": ent.get("type"),
                "description": ent.get("description"),
            },
        )
        db.query("RELATE $from->mentions->$to", {"from": d, "to": er})
        for cid in ent.get("chunk_ids", []):
            cr = chunk_rid(doc_url, cid)
            db.query("RELATE $from->extracted_from->$to", {"from": er, "to": cr})
        if ent.get("cluster_id") is not None:
            cor = cluster_rid(doc_url, ent["cluster_id"])
            db.query("RELATE $from->belongs_to->$to", {"from": er, "to": cor})
        progress.advance(task)


def import_relations(
    db, doc_url: str, relations: list, progress: Progress, task
) -> None:
    """Create typed graph edges between entities."""
    rel_tables_seen: set[str] = set()
    for rel in relations:
        table = sanitize_table(rel["type"])
        if table not in rel_tables_seen:
            db.query(
                f"DEFINE TABLE IF NOT EXISTS `{table}` TYPE RELATION IN entity OUT entity"
            )
            db.query(
                f"DEFINE FIELD IF NOT EXISTS description ON `{table}` TYPE option<string>"
            )
            db.query(
                f"DEFINE FIELD IF NOT EXISTS strength ON `{table}` TYPE option<float>"
            )
            db.query(
                f"DEFINE FIELD IF NOT EXISTS chunk_ids ON `{table}` TYPE option<array>"
            )
            db.query(
                f"DEFINE FIELD IF NOT EXISTS doc_url ON `{table}` TYPE option<string>"
            )
            rel_tables_seen.add(table)
        src = entity_rid(rel["source"])
        tgt = entity_rid(rel["target"])
        chunk_refs = [str(chunk_rid(doc_url, c)) for c in rel.get("chunk_ids", [])]
        db.query(
            f"""
            RELATE $src->`{table}`->$tgt CONTENT {{
                description: $description,
                strength: $strength,
                chunk_ids: $chunk_ids,
                doc_url: $doc_url
            }}
            """,
            {
                "src": src,
                "tgt": tgt,
                "description": rel.get("description"),
                "strength": rel.get("strength"),
                "chunk_ids": chunk_refs,
                "doc_url": doc_url,
            },
        )
        progress.advance(task)


def import_file(db, path: Path) -> dict:
    """Import a single results.json file. Returns a stats dict."""
    console.print(f"\n[bold cyan]Importing[/bold cyan] [dim]{path}[/dim]")
    with open(path) as f:
        doc = json.load(f)

    graph = doc.get("graph", {})
    entities = graph.get("entities", [])
    relations = graph.get("relations", [])
    chunks = doc.get("chunks", [])
    clusters = graph.get("clusters", [])

    if doc.get("documents") is not None:
        # Consolidated format: multiple documents listed explicitly
        doc_url = doc["documents"][0].get("url", "") if doc["documents"] else ""
        import_documents(db, doc["documents"])
    else:
        # Single extraction format: document fields at the top level
        doc_url = doc["url"]
        import_document(db, doc)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        t_chunks = progress.add_task("[green]Chunks", total=len(chunks))
        import_chunks(db, doc_url, chunks, progress, t_chunks)

        t_clust = progress.add_task("[yellow]Clusters", total=max(len(clusters), 1))
        import_clusters(db, doc_url, clusters, progress, t_clust)

        t_ent = progress.add_task("[blue]Entities", total=max(len(entities), 1))
        import_entities(db, doc_url, entities, progress, t_ent)

        t_rel = progress.add_task("[magenta]Relations", total=max(len(relations), 1))
        import_relations(db, doc_url, relations, progress, t_rel)

    return {
        "file": path.name,
        "chunks": len(chunks),
        "clusters": len(clusters),
        "entities": len(entities),
        "relations": len(relations),
    }


@app.command()
def main(
    files: Annotated[
        list[Path], typer.Argument(help="Path(s) to graph JSON file(s) to import")
    ],
    url: Annotated[
        str, typer.Option("--url", help="SurrealDB WebSocket URL")
    ] = "ws://localhost:8000/rpc",
    user: Annotated[
        str, typer.Option("--user", "-u", help="SurrealDB username")
    ] = "root",
    password: Annotated[
        str, typer.Option("--password", "-p", help="SurrealDB password")
    ] = "root",
    namespace: Annotated[
        str, typer.Option("--ns", help="SurrealDB namespace")
    ] = "knwler",
    database: Annotated[
        str, typer.Option("--db", help="SurrealDB database")
    ] = "knwler",
) -> None:
    """Import knowledge-graph JSON file(s) into SurrealDB."""
    for f in files:
        if not f.exists():
            console.print(f"[bold red]Error:[/bold red] File not found: {f}")
            raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold]URL[/bold]        {url}\n"
            f"[bold]Namespace[/bold]  {namespace}\n"
            f"[bold]Database[/bold]   {database}\n"
            f"[bold]Files[/bold]      {len(files)}",
            title="[bold green]SurrealDB Import[/bold green]",
            border_style="green",
        )
    )

    results = []
    with Surreal(url) as db:
        db.signin({"username": user, "password": password})
        db.use(namespace, database)

        with console.status("[bold]Defining schema...[/bold]", spinner="dots"):
            define_schema(db)
        console.print("[dim]Schema ready.[/dim]")

        for f in files:
            stats = import_file(db, f)
            results.append(stats)

    # Summary table
    table = Table(title="Import Summary", show_header=True, header_style="bold cyan")
    table.add_column("File", style="dim")
    table.add_column("Chunks", justify="right")
    table.add_column("Clusters", justify="right")
    table.add_column("Entities", justify="right")
    table.add_column("Relations", justify="right")

    total = {"chunks": 0, "clusters": 0, "entities": 0, "relations": 0}
    for r in results:
        table.add_row(
            r["file"],
            str(r["chunks"]),
            str(r["clusters"]),
            str(r["entities"]),
            str(r["relations"]),
        )
        for key in total:
            total[key] += r[key]

    if len(results) > 1:
        table.add_section()
        table.add_row(
            Text(f"Total ({len(results)} files)", style="bold"),
            Text(str(total["chunks"]), style="bold"),
            Text(str(total["clusters"]), style="bold"),
            Text(str(total["entities"]), style="bold"),
            Text(str(total["relations"]), style="bold"),
        )

    console.print()
    console.print(table)
    console.print(f"\n[bold green]Done.[/bold green] Imported {len(files)} file(s).")


if __name__ == "__main__":
    app()
