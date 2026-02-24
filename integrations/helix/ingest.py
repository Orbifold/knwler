#!/usr/bin/env python3
"""
Ingest a knwler graph.json into HelixDB.

The script reads the JSON produced by knwler and inserts every layer
into HelixDB using the schema and queries defined in schema.hx / queries.hx.

Vertex counts (typical single-document graph):
    1  Document
   41  Chunk nodes
  272  Entity nodes
   25  Community nodes
Edge counts:
   41  HAS_CHUNK        (Document → Chunk)
  ~600 MENTIONED_IN     (Entity → Chunk, via chunk_ids[])
   25  IN_COMMUNITY     (Entity → Community, via community_id)
  202  RELATES_TO       (Entity → Entity, from graph.relations)

Usage:
    # start HelixDB pointing at this directory as the config dir, then:
    uv run helix/ingest.py results/Key_Differences_Between/graph.json

    # custom config dir and port:
    uv run helix/ingest.py results/.../graph.json ./my_helix_dir 6969
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from helix import Client, Instance


# ── helpers ──────────────────────────────────────────────────────────────────

def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _first_id(result: list[dict]) -> str:
    """Extract the HelixDB-assigned ID from a single-item AddN response."""
    return result[0]["id"]


def _pct(done: int, total: int) -> str:
    return f"{done}/{total} ({100 * done // total if total else 0}%)"


# ── ingestion steps ───────────────────────────────────────────────────────────

def insert_document(db: Client, data: dict) -> str:
    result = db.query("AddDocument", {
        "title":    data["title"],
        "summary":  data.get("summary", ""),
        "url":      data.get("url", ""),
        "language": data.get("language", "en"),
    })
    doc_id = _first_id(result)
    print(f"  [document]    created  id={doc_id}")
    return doc_id


def insert_chunks(db: Client, data: dict, doc_id: str) -> dict[int, str]:
    """Returns {chunk_idx: helix_id}."""
    chunk_id_map: dict[int, str] = {}
    chunks = data.get("chunks", [])
    for i, chunk in enumerate(chunks):
        result = db.query("AddChunk", {
            "doc_id":      doc_id,
            "chunk_idx":   chunk["chunk_idx"],
            "text":        chunk["text"],
            "rephrase":    chunk.get("rephrase", ""),
            "chunk_tokens": chunk.get("chunk_tokens", 0),
            "source_file": chunk.get("source_file", ""),
        })
        chunk_id_map[chunk["chunk_idx"]] = _first_id(result)
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"  [chunks]      {_pct(i + 1, len(chunks))}")
    return chunk_id_map


def insert_entities(db: Client, data: dict) -> dict[str, str]:
    """Returns {name_lower: helix_id}."""
    entity_id_map: dict[str, str] = {}
    entities = data["graph"]["entities"]
    for i, entity in enumerate(entities):
        result = db.query("AddEntity", {
            "name":        entity["name"],
            "entity_type": entity.get("type", ""),
            "description": entity.get("description", ""),
        })
        # Store under lowercased name so relation lookups are case-insensitive
        entity_id_map[entity["name"].lower()] = _first_id(result)
        if (i + 1) % 50 == 0 or i == len(entities) - 1:
            print(f"  [entities]    {_pct(i + 1, len(entities))}")
    return entity_id_map


def insert_communities(db: Client, data: dict) -> dict[int, str]:
    """Returns {community_int_id: helix_id}."""
    community_id_map: dict[int, str] = {}
    communities = data.get("communities", [])
    for community in communities:
        result = db.query("AddCommunity", {
            "community_id": community["id"],
            "topics":       json.dumps(community.get("topics", [])),
            "description":  community.get("description", ""),
        })
        community_id_map[community["id"]] = _first_id(result)
    print(f"  [communities] created  {len(community_id_map)}")
    return community_id_map


def link_entity_communities(
    db: Client,
    data: dict,
    entity_id_map: dict[str, str],
    community_id_map: dict[int, str],
) -> int:
    count = 0
    for entity in data["graph"]["entities"]:
        if "community_id" not in entity:
            continue
        entity_helix_id   = entity_id_map.get(entity["name"].lower())
        community_helix_id = community_id_map.get(entity["community_id"])
        if entity_helix_id and community_helix_id:
            db.query("LinkEntityToCommunity", {
                "entity_id":    entity_helix_id,
                "community_id": community_helix_id,
            })
            count += 1
    print(f"  [IN_COMMUNITY] linked  {count}")
    return count


def link_entity_chunks(
    db: Client,
    data: dict,
    entity_id_map: dict[str, str],
    chunk_id_map: dict[int, str],
) -> int:
    count = 0
    for entity in data["graph"]["entities"]:
        entity_helix_id = entity_id_map.get(entity["name"].lower())
        if not entity_helix_id:
            continue
        for chunk_idx in entity.get("chunk_ids", []):
            chunk_helix_id = chunk_id_map.get(chunk_idx)
            if chunk_helix_id:
                db.query("LinkEntityToChunk", {
                    "entity_id": entity_helix_id,
                    "chunk_id":  chunk_helix_id,
                })
                count += 1
    print(f"  [MENTIONED_IN] linked  {count}")
    return count


def link_relations(
    db: Client,
    data: dict,
    entity_id_map: dict[str, str],
) -> tuple[int, int]:
    created = 0
    skipped = 0
    relations = data["graph"]["relations"]
    for relation in relations:
        src_id = entity_id_map.get(relation["source"].lower())
        tgt_id = entity_id_map.get(relation["target"].lower())
        if src_id and tgt_id:
            db.query("AddRelation", {
                "source_id":     src_id,
                "target_id":     tgt_id,
                "relation_type": relation.get("type", ""),
                "description":   relation.get("description", ""),
                "strength":      float(relation.get("strength", 0.0)),
            })
            created += 1
        else:
            skipped += 1
            missing = []
            if not src_id:
                missing.append(f"source='{relation['source']}'")
            if not tgt_id:
                missing.append(f"target='{relation['target']}'")
            print(f"    [RELATES_TO] skipped — entity not found: {', '.join(missing)}")
    print(f"  [RELATES_TO]   created {created}  skipped {skipped}")
    return created, skipped


# ── main ──────────────────────────────────────────────────────────────────────

def ingest(
    graph_path: str,
    helix_config_dir: str = str(Path(__file__).parent),
    port: int = 6969,
) -> None:
    data = load(graph_path)

    print(f"\nConnecting to HelixDB (config={helix_config_dir}, port={port}) …")
    instance = Instance(helix_config_dir, port, verbose=False)
    instance.deploy()
    instance.start()
    db = Client(local=True, port=port, verbose=False)

    print(f"\nIngesting: \"{data['title']}\"")
    print("─" * 60)

    doc_id           = insert_document(db, data)
    chunk_id_map     = insert_chunks(db, data, doc_id)
    entity_id_map    = insert_entities(db, data)
    community_id_map = insert_communities(db, data)

    print("─" * 60)
    link_entity_communities(db, data, entity_id_map, community_id_map)
    link_entity_chunks(db, data, entity_id_map, chunk_id_map)
    rel_created, rel_skipped = link_relations(db, data, entity_id_map)

    print("─" * 60)
    n_vertices = 1 + len(chunk_id_map) + len(entity_id_map) + len(community_id_map)
    n_edges    = (
        len(chunk_id_map)   # HAS_CHUNK
        + len(entity_id_map)  # approximate MENTIONED_IN (actual may differ)
        + len(community_id_map)  # IN_COMMUNITY
        + rel_created
    )
    print(f"Done. {n_vertices} vertices, ~{n_edges} edges written to HelixDB.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    graph_path  = sys.argv[1]
    config_dir  = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent)
    port        = int(sys.argv[3]) if len(sys.argv) > 3 else 6969

    ingest(graph_path, config_dir, port)
