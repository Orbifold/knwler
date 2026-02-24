#!/usr/bin/env python3
"""
Import knowledge-graph JSON files into Neo4j.

Usage:
    python neo4j_import.py results1.json results2.json ...
    python neo4j_import.py examples/*/results.json

Requires:
    pip install neo4j

Graph model:
    (:Document)-[:HAS_CHUNK]->(:Chunk)
    (:Document)-[:HAS_COMMUNITY]->(:Community)
    (:Document)-[:MENTIONS]->(:Entity)
    (:Entity)-[:EXTRACTED_FROM]->(:Chunk)
    (:Entity)-[:BELONGS_TO]->(:Community)
    (:Entity)-[:<dynamic_type>]->(:Entity)   # knowledge-graph relations
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your-neo4j-password"  # change this
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


def create_constraints(session):
    constraints = [
        "CREATE CONSTRAINT doc_url IF NOT EXISTS FOR (d:Document) REQUIRE d.url IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (co:Community) REQUIRE co.id IS UNIQUE",
        "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    ]
    for c in constraints:
        session.run(c)


def import_document(session, doc: dict):
    """Create the Document node with metadata and stats."""
    stats = doc["stats"][0] if doc.get("stats") else {}
    run_info = stats.get("run", {})

    session.run(
        """
        MERGE (d:Document {url: $url})
        SET d.title           = $title,
            d.summary         = $summary,
            d.language        = $language,
            d.entityTypes     = $entity_types,
            d.relationTypes   = $relation_types,
            d.schemaReasoning = $reasoning,
            d.numChunks       = $num_chunks,
            d.extractionModel = $extraction_model,
            d.discoveryModel  = $discovery_model,
            d.totalTime       = $total_time,
            d.timestamp       = $timestamp,
            d.sourceFile      = $source_file
        """,
        url=doc["url"],
        title=doc.get("title"),
        summary=doc.get("summary"),
        language=doc.get("language"),
        entity_types=doc.get("schema", {}).get("entity_types", []),
        relation_types=doc.get("schema", {}).get("relation_types", []),
        reasoning=doc.get("schema", {}).get("reasoning"),
        num_chunks=stats.get("num_chunks"),
        extraction_model=run_info.get("extraction_model"),
        discovery_model=run_info.get("discovery_model"),
        total_time=stats.get("total_time"),
        timestamp=run_info.get("timestamp"),
        source_file=run_info.get("file"),
    )


def import_chunks(session, doc_url: str, chunks: list):
    """Create Chunk nodes and link them to the Document."""
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        session.run(
            """
            UNWIND $batch AS chunk
            MERGE (c:Chunk {id: $doc_url + '::' + toString(chunk.chunk_idx)})
            SET c.chunkIdx = chunk.chunk_idx,
                c.text     = chunk.text,
                c.rephrase = chunk.rephrase
            WITH c
            MATCH (d:Document {url: $doc_url})
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            batch=[
                {
                    "chunk_idx": c["chunk_idx"],
                    "text": c.get("text"),
                    "rephrase": c.get("rephrase"),
                }
                for c in batch
            ],
            doc_url=doc_url,
        )


def import_communities(session, doc_url: str, communities: list):
    """Create Community nodes and link them to the Document."""
    for i in range(0, len(communities), BATCH_SIZE):
        batch = communities[i : i + BATCH_SIZE]
        session.run(
            """
            UNWIND $batch AS comm
            MERGE (co:Community {id: $doc_url + '::' + toString(comm.id)})
            SET co.communityIdx = comm.id,
                co.topics       = comm.topics,
                co.description  = comm.description,
                co.members      = comm.members
            WITH co
            MATCH (d:Document {url: $doc_url})
            MERGE (d)-[:HAS_COMMUNITY]->(co)
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
            doc_url=doc_url,
        )


def import_entities(session, doc_url: str, entities: list):
    """Create Entity nodes, link to Document, Chunks, and Communities."""
    for i in range(0, len(entities), BATCH_SIZE):
        batch = entities[i : i + BATCH_SIZE]
        # Create/merge entities and link to document
        session.run(
            """
            UNWIND $batch AS ent
            MERGE (e:Entity {name: ent.name})
            ON CREATE SET e.type        = ent.type,
                          e.description = ent.description
            ON MATCH SET  e.description = CASE
                            WHEN ent.description IS NOT NULL
                             AND size(ent.description) > size(coalesce(e.description, ''))
                            THEN ent.description
                            ELSE e.description
                          END,
                          e.type = coalesce(e.type, ent.type)
            WITH e
            MATCH (d:Document {url: $doc_url})
            MERGE (d)-[:MENTIONS]->(e)
            """,
            batch=[
                {
                    "name": e["name"],
                    "type": e.get("type"),
                    "description": e.get("description"),
                }
                for e in batch
            ],
            doc_url=doc_url,
        )

        # Link entities to their source chunks
        chunk_links = []
        for e in batch:
            for cid in e.get("chunk_ids", []):
                chunk_links.append({"name": e["name"], "chunk_id": f"{doc_url}::{cid}"})
        if chunk_links:
            for j in range(0, len(chunk_links), BATCH_SIZE):
                session.run(
                    """
                    UNWIND $links AS link
                    MATCH (e:Entity {name: link.name})
                    MATCH (c:Chunk {id: link.chunk_id})
                    MERGE (e)-[:EXTRACTED_FROM]->(c)
                    """,
                    links=chunk_links[j : j + BATCH_SIZE],
                )

        # Link entities to their communities
        comm_links = [
            {"name": e["name"], "comm_id": f"{doc_url}::{e['community_id']}"}
            for e in batch
            if e.get("community_id") is not None
        ]
        if comm_links:
            session.run(
                """
                UNWIND $links AS link
                MATCH (e:Entity {name: link.name})
                MATCH (co:Community {id: link.comm_id})
                MERGE (e)-[:BELONGS_TO]->(co)
                """,
                links=comm_links,
            )


def import_relations(session, doc_url: str, relations: list):
    """Create typed relationships between entities."""
    # Group relations by type so we can use a single Cypher per type
    by_type: dict[str, list] = {}
    for rel in relations:
        rel_type = sanitize_relationship_type(rel.get("type"))
        by_type.setdefault(rel_type, []).append(rel)

    for rel_type, rels in by_type.items():
        for i in range(0, len(rels), BATCH_SIZE):
            batch = rels[i : i + BATCH_SIZE]
            # Dynamic relationship types require APOC or string interpolation.
            # Since we group by type, we can use a literal type per query.
            query = f"""
                UNWIND $batch AS rel
                MATCH (src:Entity {{name: rel.source}})
                MATCH (tgt:Entity {{name: rel.target}})
                MERGE (src)-[r:`{rel_type}`]->(tgt)
                SET r.description = rel.description,
                    r.strength    = rel.strength,
                    r.chunk_ids   = rel.chunk_ids,
                    r.doc_url     = $doc_url
            """
            session.run(
                query,
                batch=[
                    {
                        "source": r["source"],
                        "target": r["target"],
                        "description": r.get("description"),
                        "strength": r.get("strength"),
                        "chunk_ids": [
                            f"{doc_url}::{c}" for c in r.get("chunk_ids", [])
                        ],
                    }
                    for r in batch
                ],
                doc_url=doc_url,
            )


def import_file(driver, path: Path):
    """Import a single results.json file."""
    print(f"Loading {path} ...")
    with open(path) as f:
        doc = json.load(f)

    doc_url = doc["url"]
    graph = doc.get("consolidated", {})
    entities = graph.get("entities", [])
    relations = graph.get("relations", [])
    chunks = doc.get("chunks", [])
    communities = doc.get("communities", [])

    with driver.session() as session:
        import_document(session, doc)
        print(f"  {len(chunks)} chunks ...")
        import_chunks(session, doc_url, chunks)
        print(f"  {len(communities)} communities ...")
        import_communities(session, doc_url, communities)
        print(f"  {len(entities)} entities ...")
        import_entities(session, doc_url, entities)
        print(f"  {len(relations)} relations ...")
        import_relations(session, doc_url, relations)

    print(f"  Done: {path.name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python neo4j_import.py <results.json> [results2.json ...]")
        sys.exit(1)

    files = [Path(p) for p in sys.argv[1:]]
    for f in files:
        if not f.exists():
            print(f"File not found: {f}")
            sys.exit(1)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        create_constraints(session)

    for f in files:
        import_file(driver, f)

    driver.close()
    print(f"\nImported {len(files)} file(s).")


if __name__ == "__main__":
    main()
