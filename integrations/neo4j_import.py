#!/usr/bin/env python3
"""
Import consolidated knowledge-graph JSON files into Neo4j.

Expected JSON structure:
    {
      "id": "<graph-id>",
      "documents": [{"id": ..., "title": ..., "url": ..., "language": ..., "summary": ...}],
      "schema":    {"entity_types": [...], "relation_types": [...]},
      "graph":     {"entities": [...], "relations": [...], "communities": [...]}
    }

Graph model:
    (:Document)-[:CONTAINS]->(:Chunk)

    (:Entity)-[:<dynamic_type>]->(:Entity)
    (:Entity)-[:BELONGS_TO]->(:Community)
    (:Community)-[:PART_OF]->(:Graph)
"""

import json
import re
import unicodedata
from pathlib import Path
from uuid import uuid4
import typer
from neo4j import GraphDatabase
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

app = typer.Typer(help="Import knowledge-graph JSON files into Neo4j.")
console = Console()

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123456789"
BATCH_SIZE = 500


def sanitize_relationship_type(raw_type: str) -> str:
    """Return a safe Neo4j relationship token (without backticks)."""
    if raw_type is None:
        return "RELATED_TO"
    normalized = unicodedata.normalize("NFKD", str(raw_type))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.upper()
    normalized = re.sub(r"[\x00-\x1F\x7F]", "", normalized)
    normalized = normalized.replace("`", "")
    normalized = re.sub(r"[^A-Z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized or "RELATED_TO"


def import_documents(session, documents: list):
    """
    Create Document nodes.
    """
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        session.run(
            """
            UNWIND $batch AS doc
            MERGE (d:Document {id: doc.id})
            SET d.title    = doc.title,
                d.summary  = doc.summary,
                d.url      = doc.url,
                d.language = doc.language
            """,
            batch=[
                {
                    "id": d["id"],
                    "title": d.get("title"),
                    "summary": d.get("summary"),
                    "url": d.get("url"),
                    "language": d.get("language"),
                }
                for d in batch
            ],
        )


def import_chunks(session, chunks: list):
    """
    Create Chunk nodes and link to their Documents via CONTAINS relationships.
    """
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        session.run(
            """
            UNWIND $batch AS chunk
            MERGE (c:Chunk {id: chunk.id})
            SET c.text = chunk.text,
                c.rephrase = chunk.rephrase,
                c.chunkIdx = chunk.chunk_idx,
                c.document = chunk.document
                
            with c, c.document as docId
            MATCH (d:Document {id: docId})
            CREATE (d)-[:CONTAINS]->(c)
            
            """,
            batch=[
                {
                    "id": c["id"],
                    "text": c.get("text"),
                    "rephrase": c.get("rephrase"),
                    "chunk_idx": c.get("chunk_idx", -1),
                    "document": c.get("document"),
                }
                for c in batch
            ],
        )


def create_constraints(session):
    constraints = [
        "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (co:Community) REQUIRE co.id IS UNIQUE",
        "CREATE CONSTRAINT entity_name_type IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    ]
    for c in constraints:
        session.run(c)


def import_graph(session, graph_id: str, schema: dict):
    """Create the top-level Graph node."""
    session.run(
        """
        MERGE (g:Graph {id: $id})
        SET g.entityTypes   = $entity_types,
            g.relationTypes = $relation_types
        """,
        id=graph_id,
        entity_types=schema.get("entity_types", []),
        relation_types=schema.get("relation_types", []),
    )


def import_communities(session, graph_id: str, communities: list):
    """Create Community nodes, link to Graph, then link member entities."""
    for i in range(0, len(communities), BATCH_SIZE):
        batch = communities[i : i + BATCH_SIZE]
        session.run(
            """
            UNWIND $batch AS comm
            MERGE (co:Community {id: $graph_id + '::' + toString(comm.id)})
            SET co.communityIdx = comm.id,
                co.topics       = comm.topics,
                co.description  = comm.description,
                co.members      = comm.members
            WITH co
            MATCH (g:Graph {id: $graph_id})
            MERGE (co)-[:PART_OF]->(g)
            """,
            batch=[
                {
                    "id": c["id"],
                    "topics": c.get("topics", []),
                    "description": c.get("description"),
                    "members": c.get("members", []),
                }
                for c in batch
            ],
            graph_id=graph_id,
        )

    # Link member entities to their communities
    member_links = []
    for c in communities:
        comm_id = f"{graph_id}::{c['id']}"
        for member_name in c.get("members", []):
            member_links.append({"comm_id": comm_id, "name": member_name})

    for i in range(0, len(member_links), BATCH_SIZE):
        session.run(
            """
            UNWIND $links AS link
            MATCH (co:Community {id: link.comm_id})
            MATCH (e:Entity {name: link.name})
            MERGE (e)-[:BELONGS_TO]->(co)
            """,
            links=member_links[i : i + BATCH_SIZE],
        )


def import_entities(session, entities: list):
    """Create Entity nodes and link to the Graph."""
    for i in range(0, len(entities), BATCH_SIZE):
        batch = entities[i : i + BATCH_SIZE]
        session.run(
            """
            UNWIND $batch AS ent
            MERGE (e:Entity {id: ent.id})
            SET e.description = ent.description,
                e.name = ent.name,
                e.type = ent.type
            """,
            batch=[
                {
                    "id": e.get("id", str(uuid4())),
                    "name": e["name"],
                    "type": e.get("type"),
                    "description": e.get("description"),
                    "chunk_ids": e.get("chunk_ids", []),
                }
                for e in batch
            ],
        )

        # Link chunks to entities
        chunk_links = [
            {"chunk_id": cid, "entity_id": e["id"], "name": e["name"]}
            for e in batch
            for cid in e.get("chunk_ids", [])
        ]
        for j in range(0, len(chunk_links), BATCH_SIZE):
            session.run(
                """
                UNWIND $links AS link
                MATCH (c:Chunk {id: link.chunk_id})
                MATCH (e:Entity {id: link.entity_id})
                MERGE (c)-[:HAS_ENTITY]->(e)
                """,
                links=chunk_links[j : j + BATCH_SIZE],
            )


def import_relations(session, relations: list):
    """Create typed relationships between entities."""
    by_type: dict[str, list] = {}
    for rel in relations:
        rel_type = sanitize_relationship_type(rel.get("type"))
        by_type.setdefault(rel_type, []).append(rel)

    for rel_type, rels in by_type.items():
        for i in range(0, len(rels), BATCH_SIZE):
            batch = rels[i : i + BATCH_SIZE]
            query = f"""
                UNWIND $batch AS rel
                MATCH (src:Entity {{name: rel.source}})
                MATCH (tgt:Entity {{name: rel.target}})
                MERGE (src)-[r:`{rel_type}`]->(tgt)
                SET r.description = rel.description,
                    r.strength    = rel.strength,
                    r.chunks_ids    = rel.chunk_ids
            """
            session.run(
                query,
                batch=[
                    {
                        "source": r["source"],
                        "target": r["target"],
                        "description": r.get("description"),
                        "strength": r.get("strength"),
                        "chunk_ids": r.get("chunk_ids", []),
                    }
                    for r in batch
                ],
            )


def import_file(driver, path: Path, progress: Progress, file_task):
    """
    Import a single graph (JSON) file.
    """
    with open(path) as f:
        doc = json.load(f)
    console.print(
        f"Importing [cyan]{path.name}[/cyan] with {len(doc.get('documents', []))} documents, {len(doc.get('graph', {}).get('entities', []))} entities, and {len(doc.get('graph', {}).get('relations', []))} relations."
    )
    if doc.get("documents") is None:
        # single extraction
        documents = [
            {
                "id": doc.get("id", str(uuid.uuid4())),
                "title": doc.get(
                    "title", f"Document {doc.get('id', str(uuid.uuid4()))}"
                ),
                "url": doc.get("url", ""),
                "language": doc.get("language", "en"),
                "summary": doc.get("summary", ""),
            }
        ]
    else:
        # consolidated set
        documents = doc.get("documents", [])

    schema = doc.get("schema", {})
    graph = doc.get("graph", {})
    chunks = doc.get("chunks", {})
    entities = graph.get("entities", [])
    relations = graph.get("relations", [])
    communities = graph.get("communities", [])

    steps = [
        # (
        #     f"{len(documents)} documents",
        #     lambda session: import_documents(session, documents),
        # ),
        # (
        #     f"{len(chunks)} chunks",
        #     lambda session: import_chunks(session, chunks),
        # ),
        # (
        #     f"{len(entities)} entities",
        #     lambda session: import_entities(session, entities),
        # ),
        (
            f"{len(relations)} relations",
            lambda session: import_relations(session, relations),
        ),
        # ("graph node", lambda session: import_graph(session, graph_id, schema)),
        # (
        #     f"{len(communities)} communities",
        #     lambda session: import_communities(session, graph_id, communities),
        # ),
    ]

    with driver.session() as session:
        for label, fn in steps:
            console.print(f"  - Importing [green]{label}[/green]...")
            fn(session)

    progress.update(file_task, advance=1)


@app.command()
def main(
    files: list[Path] = typer.Argument(..., help="JSON graph files to import"),
    uri: str = typer.Option(NEO4J_URI, "--uri", "-u", help="Neo4j bolt URI"),
    user: str = typer.Option(NEO4J_USER, "--user", help="Neo4j username"),
    password: str = typer.Option(
        NEO4J_PASSWORD, "--password", "-p", help="Neo4j password"
    ),
    batch_size: int = typer.Option(
        BATCH_SIZE, "--batch-size", "-b", help="Cypher batch size"
    ),
):
    """Import knowledge-graph JSON files into Neo4j."""
    global BATCH_SIZE
    BATCH_SIZE = batch_size

    missing = [f for f in files if not f.exists()]
    if missing:
        for f in missing:
            console.print(f"[red]File not found:[/red] {f}")
        raise typer.Exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        create_constraints(session)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        file_task = progress.add_task("⏩ ", total=len(files))
        for f in files:
            import_file(driver, f, progress, file_task)

    driver.close()

    table = Table(title="Import Summary", show_header=False)
    table.add_row("Files imported", str(len(files)))
    for f in files:
        table.add_row("", f"[dim]{f.name}[/dim]")
    console.print(table)


if __name__ == "__main__":
    app()
