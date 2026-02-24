#!/usr/bin/env python3
"""
Import knowledge-graph JSON files into SurrealDB (v3.0).

Usage:
    uv run surrealdb_import.py examples/Deloitte/results.json

Requires:
    uv add surrealdb

Data model:
    document  ──has_chunk──>     chunk
    document  ──has_community──> community
    document  ──mentions──>      entity
    entity    ──extracted_from──> chunk
    entity    ──belongs_to──>    community
    entity    ──<dynamic_type>──> entity      # knowledge-graph relations

SurrealDB leverages multi-model storage: each record type is a table,
graph edges are TYPE RELATION tables. Entities that appear across
multiple documents are merged by name (longer description wins).


To start SurrealDB locally with a RocksDB store, run:
    surreal start --user root --pass root rocksdb:~/surreal/

To see the results in Surrealist, connect with root/root and select the ns/db via the menu at the top.
Via the SQL CLI:

     surreal sql -u root -p root


"""

import json
import re
import sys
from pathlib import Path

from surrealdb import Surreal, RecordID

SURREAL_URL = "ws://localhost:8000/rpc"
SURREAL_USER = "root"
SURREAL_PASS = "root"  # change this
SURREAL_NS = "knwler"
SURREAL_DB = "knwler"
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


def community_rid(url: str, idx: int) -> RecordID:
    return RecordID("community", f"{sanitize_id(url)}__{idx}")


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
        "DEFINE TABLE IF NOT EXISTS community SCHEMAFULL",
        "DEFINE FIELD IF NOT EXISTS community_idx ON community TYPE int",
        "DEFINE FIELD IF NOT EXISTS topics ON community TYPE option<array>",
        "DEFINE FIELD IF NOT EXISTS description ON community TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS members ON community TYPE option<array>",
        "DEFINE TABLE IF NOT EXISTS entity SCHEMALESS",
        "DEFINE FIELD IF NOT EXISTS name ON entity TYPE string",
        "DEFINE FIELD IF NOT EXISTS type ON entity TYPE option<string>",
        "DEFINE FIELD IF NOT EXISTS description ON entity TYPE option<string>",
        "DEFINE INDEX IF NOT EXISTS entity_name_idx ON entity FIELDS name UNIQUE",
        "DEFINE INDEX IF NOT EXISTS entity_type_idx ON entity FIELDS type",
        # Structural relation tables
        "DEFINE TABLE IF NOT EXISTS has_chunk TYPE RELATION IN document OUT chunk",
        "DEFINE TABLE IF NOT EXISTS has_community TYPE RELATION IN document OUT community",
        "DEFINE TABLE IF NOT EXISTS mentions TYPE RELATION IN document OUT entity",
        "DEFINE TABLE IF NOT EXISTS extracted_from TYPE RELATION IN entity OUT chunk",
        "DEFINE TABLE IF NOT EXISTS belongs_to TYPE RELATION IN entity OUT community",
    ]
    for q in queries:
        db.query(q)


def import_document(db, doc: dict):
    """Create the document record."""
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


def import_chunks(db, doc_url: str, chunks: list):
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
            db.query(
                "RELATE $from->has_chunk->$to",
                {"from": d, "to": cr},
            )


def import_communities(db, doc_url: str, communities: list):
    """Create community records and relate them to the document."""
    d = doc_rid(doc_url)

    for comm in communities:
        cr = community_rid(doc_url, comm["id"])
        db.upsert(
            cr,
            {
                "community_idx": comm["id"],
                "topics": comm.get("topics", []),
                "description": comm.get("description"),
                "members": comm.get("members", []),
            },
        )
        db.query(
            "RELATE $from->has_community->$to",
            {"from": d, "to": cr},
        )


def import_entities(db, doc_url: str, entities: list):
    """Create entity records, link to document, chunks, and communities."""
    d = doc_rid(doc_url)

    for ent in entities:
        er = entity_rid(ent["name"])

        # Upsert entity — keep the longer description
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

        # Link entity → document
        db.query(
            "RELATE $from->mentions->$to",
            {"from": d, "to": er},
        )

        # Link entity → source chunks
        for cid in ent.get("chunk_ids", []):
            cr = chunk_rid(doc_url, cid)
            db.query(
                "RELATE $from->extracted_from->$to",
                {"from": er, "to": cr},
            )

        # Link entity → community
        if ent.get("community_id") is not None:
            cor = community_rid(doc_url, ent["community_id"])
            db.query(
                "RELATE $from->belongs_to->$to",
                {"from": er, "to": cor},
            )


def import_relations(db, doc_url: str, relations: list):
    """Create typed graph edges between entities."""
    rel_tables_seen: set[str] = set()

    for rel in relations:
        table = sanitize_table(rel["type"])

        # Define the relation table on first encounter
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


def import_file(db, path: Path):
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

    import_document(db, doc)
    print(f"  {len(chunks)} chunks ...")
    import_chunks(db, doc_url, chunks)
    print(f"  {len(communities)} communities ...")
    import_communities(db, doc_url, communities)
    print(f"  {len(entities)} entities ...")
    import_entities(db, doc_url, entities)
    print(f"  {len(relations)} relations ...")
    import_relations(db, doc_url, relations)

    print(f"  Done: {path.name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python surrealdb_import.py <results.json> [results2.json ...]")
        sys.exit(1)

    files = [Path(p) for p in sys.argv[1:]]
    for f in files:
        if not f.exists():
            print(f"File not found: {f}")
            sys.exit(1)

    with Surreal(SURREAL_URL) as db:
        db.signin({"username": SURREAL_USER, "password": SURREAL_PASS})
        db.use(SURREAL_NS, SURREAL_DB)

        define_schema(db)

        for f in files:
            import_file(db, f)

    print(f"\nImported {len(files)} file(s).")


if __name__ == "__main__":
    main()
