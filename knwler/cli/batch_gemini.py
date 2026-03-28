#!/usr/bin/env python3
"""
Knwler Google Gemini Batch Processor
=====================================

Processes a directory of documents through the Knwler knowledge graph
extraction pipeline using Google Gemini's Batch API for cost-effective
bulk processing (50% cost savings).

Features:
    - Processes all supported files in a directory (.pdf, .txt, .md)
    - Uses Gemini Batch API (50% cost savings) via file-based requests
    - SQLite-based state management for disaster recovery
    - Automatic polling with exponential backoff
    - Organized output directory with per-file results

Usage:
    # Start or resume processing
    python main.py batch-gemini run --input ./documents --output ./results

    # Check pipeline status
    python main.py batch-gemini status --input ./documents --output ./results

    # Use specific models
    python main.py batch-gemini run -i ./docs -o ./out --discovery-model gemini-3-flash-preview --extraction-model gemini-3-flash-preview

Pipeline Rounds:
    Round 1 (Discovery):    Language detection + Schema discovery
    Round 2 (Processing):   Title + Summary + Rephrase + Graph extraction
    Round 3 (Finalization): Consolidation summarization + Cluster labeling

Requirements:
    GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set.
    The google-genai package must be installed: pip install google-genai
"""

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import networkx as nx
import typer
from networkx.algorithms.community import louvain_communities
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knwler.chunking import chunk_text, get_encoder
from knwler.config import (
    Config,
    DEFAULT_GEMINI_DISCOVERY_MODEL,
    DEFAULT_GEMINI_EXTRACTION_MODEL,
)
from knwler.export import export_html
from knwler.language import (
    DEFAULT_LANGUAGE,
    get_prompt,
    load_languages,
    set_language,
)
from knwler.llm import parse_json_response
from knwler.models import Schema

console = Console()

SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".text", ".markdown"}
INITIAL_POLL_INTERVAL = 30
MAX_POLL_INTERVAL = 300
SUMMARIZATION_BATCH_SIZE = 20

# Gemini Batch API limits
MAX_BATCH_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

# Terminal states for Gemini batch jobs
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


# ═══════════════════════════════════════════════════════════════════════════
# Database
# ═══════════════════════════════════════════════════════════════════════════


class Database:
    """SQLite state manager for the batch processing pipeline."""

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_id              TEXT PRIMARY KEY,
                file_path            TEXT NOT NULL UNIQUE,
                file_name            TEXT NOT NULL,
                text_content         TEXT,
                language             TEXT,
                title                TEXT,
                summary              TEXT,
                schema_json          TEXT,
                chunks_json          TEXT,
                rephrased_json       TEXT,
                extractions_json     TEXT,
                agg_entities_json    TEXT,
                agg_relations_json   TEXT,
                summaries_json       TEXT,
                clusters_json        TEXT,
                graph_output_json    TEXT
            );

            CREATE TABLE IF NOT EXISTS batches (
                round_name      TEXT PRIMARY KEY,
                job_name        TEXT,
                input_file_name TEXT,
                output_file_name TEXT,
                status          TEXT DEFAULT 'pending',
                request_count   INTEGER DEFAULT 0,
                created_at      TEXT,
                completed_at    TEXT,
                error_details   TEXT
            );
        """
        )
        self.conn.commit()

    def upsert_file(self, file_id: str, file_path: str, file_name: str):
        self.conn.execute(
            "INSERT OR IGNORE INTO files (file_id, file_path, file_name) VALUES (?, ?, ?)",
            (file_id, file_path, file_name),
        )
        self.conn.commit()

    def update_file(self, file_id: str, **kwargs):
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [file_id]
        self.conn.execute(f"UPDATE files SET {sets} WHERE file_id = ?", vals)
        self.conn.commit()

    def get_file(self, file_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_files(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM files").fetchall()]

    def get_batch(self, round_name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM batches WHERE round_name = ?", (round_name,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_batch(self, round_name: str, **kwargs):
        existing = self.get_batch(round_name)
        if existing:
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            vals = list(kwargs.values()) + [round_name]
            self.conn.execute(f"UPDATE batches SET {sets} WHERE round_name = ?", vals)
        else:
            kwargs["round_name"] = round_name
            cols = ", ".join(kwargs.keys())
            placeholders = ", ".join("?" for _ in kwargs)
            self.conn.execute(
                f"INSERT INTO batches ({cols}) VALUES ({placeholders})",
                list(kwargs.values()),
            )
        self.conn.commit()

    def close(self):
        self.conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# Gemini Batch API client
# ═══════════════════════════════════════════════════════════════════════════


class GeminiBatchClient:
    """Client for Google Gemini Batch API operations using google-genai SDK."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set. "
                "Export it before running this script."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "The google-genai package is required for Gemini batch processing. "
                "Install it with: pip install google-genai"
            )
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=api_key)

    def upload_jsonl(self, jsonl_path: Path) -> str:
        """Upload a JSONL file via the File API. Returns the file resource name."""
        uploaded = self.client.files.upload(
            file=str(jsonl_path),
            config=self.types.UploadFileConfig(
                display_name=jsonl_path.stem,
                mime_type="jsonl",
            ),
        )
        return uploaded.name

    def create_batch(self, model: str, file_name: str, display_name: str = None) -> str:
        """Create a batch job from an uploaded file. Returns the job name."""
        config = {}
        if display_name:
            config["display_name"] = display_name
        job = self.client.batches.create(
            model=model,
            src=file_name,
            config=config,
        )
        return job.name

    def get_batch(self, job_name: str):
        """Retrieve current batch job status."""
        return self.client.batches.get(name=job_name)

    def download_file(self, file_name: str) -> bytes:
        """Download result file content as bytes."""
        return self.client.files.download(file=file_name)

    def poll_batch(self, job_name: str):
        """Poll until the batch job reaches a terminal state. Returns the final job."""
        interval = INITIAL_POLL_INTERVAL
        while True:
            job = self.get_batch(job_name)
            state = job.state.name if hasattr(job.state, "name") else str(job.state)

            console.print(f"  [dim]{time.strftime('%H:%M:%S')} | state={state}[/dim]")

            if state in TERMINAL_STATES:
                return job

            time.sleep(interval)
            interval = min(interval * 1.5, MAX_POLL_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt builders  (mirror the prompts used by Knwler's pipeline)
# ═══════════════════════════════════════════════════════════════════════════


def _gemini_request(
    key: str,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> dict:
    """Build one JSONL line for the Gemini Batch API (file-based format).

    Format: {"key": "<id>", "request": {"contents": [...], "generation_config": {...}}}
    """
    return {
        "key": key,
        "request": {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                    "role": "user",
                }
            ],
            "generation_config": {
                "response_mime_type": "application/json",
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        },
    }


def _prompt_language(text: str, sample_size: int = 1000) -> str:
    sample = text[:sample_size] if len(text) > sample_size else text
    return (
        "Detect the language of the following text. "
        "Return only a JSON object with the ISO 639-1 language code "
        '(e.g., "en" for English, "de" for German, "fr" for French, etc.).\n\n'
        f'TEXT:\n"""{sample}"""\n\n'
        'Return JSON:\n{\n  "language": "en"\n}'
    )


def _prompt_schema(text: str, sample_size: int = 4000) -> str:
    if len(text) <= sample_size:
        sample = text
    else:
        c = sample_size // 3
        sample = "\n\n[...]\n\n".join(
            [
                text[:c],
                text[len(text) // 2 - c // 2 : len(text) // 2 + c // 2],
                text[-c:],
            ]
        )
    prompt = get_prompt(
        "schema_discovery", sample=sample, max_entity_types=20, max_relation_types=25
    )
    if not prompt:
        prompt = (
            "Analyze this text and identify the best entity types and relation types "
            "for a knowledge graph.\n\n"
            f'TEXT SAMPLE:\n"""{sample}"""\n\n'
            "Guidelines:\n"
            "- Use snake_case for all type names\n"
            '- Be specific but not too narrow (e.g., "person" not "male_scientist")\n'
            "- Focus on types that appear multiple times or are central to the text\n"
            "- Entity types: what kinds of things are mentioned?\n"
            "- Relation types: what relationships exist between them?\n\n"
            "Return JSON:\n{\n"
            '  "entity_types": ["type1", "type2", ...],\n'
            '  "relation_types": ["rel1", "rel2", ...],\n'
            '  "reasoning": "Brief explanation"\n}\n\n'
            "Max 20 entity types and 25 relation types."
        )
    return prompt


def _prompt_title(chunks: list[str], max_chunks: int = 3) -> str:
    sample = "\n\n".join(chunks[:max_chunks])
    prompt = get_prompt("extract_title", sample=sample)
    if not prompt:
        prompt = (
            "Create a short, clear document title based on the text below. "
            "Return only a JSON object.\n\n"
            f'TEXT:\n"""{sample}"""\n\n'
            'Return JSON:\n{\n  "title": "..."\n}'
        )
    return prompt


def _prompt_summary(chunks: list[str], max_chunks: int = 3) -> str:
    sample = "\n\n".join(chunks[:max_chunks])
    prompt = get_prompt("extract_summary", sample=sample)
    if not prompt:
        prompt = (
            "Summarize the document below in 3-5 sentences. "
            "Focus on the main topic and key points. Return only JSON.\n\n"
            f'TEXT:\n"""{sample}"""\n\n'
            'Return JSON:\n{\n  "summary": "..."\n}'
        )
    return prompt


def _prompt_rephrase(chunk: str) -> str:
    prompt = get_prompt("rephrase_chunk", chunk=chunk)
    if not prompt:
        prompt = (
            "Rewrite the following text so that it is easy to understand. "
            "Keep all the key facts and names. Convey the information "
            "clearly and concisely.\n\n"
            f'TEXT:\n"""{chunk}"""\n\n'
            'Return JSON:\n{\n  "rephrase": "..."\n}'
        )
    return prompt


def _prompt_extraction(text: str, schema: Schema) -> str:
    prompt = get_prompt(
        "extract_graph",
        entity_types=json.dumps(schema.entity_types),
        relation_types=json.dumps(schema.relation_types),
        text=text,
    )
    if not prompt:
        prompt = (
            "Extract a knowledge graph from the text below.\n\n"
            f"ENTITY TYPES: {json.dumps(schema.entity_types)}\n"
            f"RELATION TYPES: {json.dumps(schema.relation_types)}\n\n"
            "RULES:\n"
            "- Only extract entities and relations clearly stated in the text\n"
            "- Each entity: name, type, 1-2 sentence description\n"
            "- Each relation: source, source_type, target, target_type, type, "
            "brief description, strength (1-10)\n"
            "- IMPORTANT: source_type and target_type must match the type of "
            "the corresponding entity\n\n"
            f'TEXT:\n"""{text}"""\n\n'
            "Return JSON:\n{\n"
            '  "entities": [{"name": "...", "type": "...", "description": "..."}],\n'
            '  "relations": [{"source": "...", "source_type": "...", "target": "...", '
            '"target_type": "...", "type": "...", '
            '"description": "...", "strength": 8}]\n}'
        )
    return prompt


def _prompt_summarize(items: list[dict]) -> str:
    lines = []
    for item in items:
        descs = ", ".join(f'"{d}"' for d in item["descriptions"])
        lines.append(f'- "{item["id"]}": [{descs}]')
    items_text = "\n".join(lines)
    prompt = get_prompt("summarize_descriptions", items_text=items_text)
    if not prompt:
        prompt = (
            "Summarize multiple descriptions for each item into ONE concise "
            "description (1-2 sentences).\n\n"
            f"ITEMS:\n{items_text}\n\n"
            "Return JSON:\n{\n"
            '  "summaries": {\n'
            '    "<id>": "<merged description>",\n'
            "    ...\n  }\n}"
        )
    return prompt


def _prompt_cluster(clusters: list[dict]) -> str:
    lines = []
    for c in clusters:
        members = ", ".join(
            f'{{"name": "{m["name"]}", "type": "{m["type"]}", '
            f'"description": "{m["description"]}"}}'
            for m in c["members"]
        )
        lines.append(f'- "{c["id"]}": [{members}]')
    clusters_text = "\n".join(lines)
    prompt = get_prompt("cluster_labeling", clusters_text=clusters_text)
    if not prompt:
        prompt = (
            "You are labeling graph clusters. For each cluster, return "
            "1-3 short topics and a 1-2 sentence description.\n\n"
            f"CLUSTERS:\n{clusters_text}\n\n"
            "Return JSON:\n{\n"
            '  "clusters": {\n'
            '    "<id>": {"topics": ["topic1", "topic2"], "description": "..."},\n'
            "    ...\n  }\n}"
        )
    return prompt


# ═══════════════════════════════════════════════════════════════════════════
# Local processing helpers
# ═══════════════════════════════════════════════════════════════════════════


def _load_text(file_path: Path) -> str:
    """Extract text from a PDF or read a text file."""
    from knwler.parse import load_text

    return load_text(file_path)


def _file_id(file_path: Path) -> str:
    """Deterministic short ID derived from the absolute path."""
    return hashlib.sha256(str(file_path.resolve()).encode()).hexdigest()[:16]


def _aggregate(extractions: list[dict]) -> tuple[dict, dict]:
    """Phase 1 of consolidation: group entities/relations by key."""
    entity_map: dict[tuple, dict] = {}
    relation_map: dict[tuple, dict] = {}

    for ext in extractions:
        chunk_id = ext.get("id", str(uuid4()))

        for e in ext.get("entities", []):
            name = (e.get("name") or "").strip()
            etype = (e.get("type") or "").strip()
            desc = (e.get("description") or "").strip()
            if not name or not etype:
                continue
            key = (name.lower(), etype.lower())
            if key not in entity_map:
                entity_map[key] = {
                    "id": str(uuid4()),
                    "name": name,
                    "type": etype,
                    "descriptions": [],
                    "chunk_ids": set(),
                }
            entity_map[key]["chunk_ids"].add(chunk_id)
            if desc and desc not in entity_map[key]["descriptions"]:
                entity_map[key]["descriptions"].append(desc)

        for rel in ext.get("relations", []):
            if isinstance(rel, str):
                continue
            src = (rel.get("source") or "").strip()
            src_type = (rel.get("source_type") or "").strip()
            tgt = (rel.get("target") or "").strip()
            tgt_type = (rel.get("target_type") or "").strip()
            rtype = (rel.get("type") or "").strip()
            desc = (rel.get("description") or "").strip()
            strength = rel.get("strength", 5)
            if not src or not tgt or not rtype:
                continue
            key = (
                src.lower(),
                src_type.lower(),
                tgt.lower(),
                tgt_type.lower(),
                rtype.lower(),
            )
            if key not in relation_map:
                relation_map[key] = {
                    "id": str(uuid4()),
                    "source": src,
                    "source_type": src_type,
                    "target": tgt,
                    "target_type": tgt_type,
                    "type": rtype,
                    "descriptions": [],
                    "strengths": [],
                    "chunk_ids": set(),
                }
            relation_map[key]["chunk_ids"].add(chunk_id)
            relation_map[key]["strengths"].append(strength)
            if desc and desc not in relation_map[key]["descriptions"]:
                relation_map[key]["descriptions"].append(desc)

    return entity_map, relation_map


def _items_to_summarize(entity_map: dict, relation_map: dict) -> list[dict]:
    """Find entities/relations that have multiple descriptions."""
    items = []
    for key, e in entity_map.items():
        if len(e["descriptions"]) > 1:
            items.append(
                {
                    "id": f"entity:{key[0]}:{key[1]}",
                    "name": e["name"],
                    "type": e["type"],
                    "descriptions": e["descriptions"],
                }
            )
    for key, r in relation_map.items():
        if len(r["descriptions"]) > 1:
            items.append(
                {
                    "id": f"relation:{key[0]}:{key[1]}:{key[2]}",
                    "name": f"{r['source']} -> {r['target']}",
                    "type": r["type"],
                    "descriptions": r["descriptions"],
                }
            )
    return items


def _apply_summaries(
    entity_map: dict, relation_map: dict, summaries: dict
) -> tuple[dict, dict]:
    """Apply LLM-produced summaries back into the aggregated maps."""
    for key, e in entity_map.items():
        sid = f"entity:{key[0]}:{key[1]}"
        if sid in summaries:
            e["descriptions"] = [summaries[sid]]
        elif len(e["descriptions"]) > 1:
            e["descriptions"] = [" ".join(e["descriptions"])]
    for key, r in relation_map.items():
        sid = f"relation:{key[0]}:{key[1]}:{key[2]}"
        if sid in summaries:
            r["descriptions"] = [summaries[sid]]
        elif len(r["descriptions"]) > 1:
            r["descriptions"] = [" ".join(r["descriptions"])]
    return entity_map, relation_map


def _is_meaningful_name(name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    if re.match(r"^[\d.,\-+%$€£¥]+$", name):
        return False
    if len(name) == 1 and not name.isalpha():
        return False
    if re.match(r"^[\W_]+$", name):
        return False
    filler = {
        "the",
        "a",
        "an",
        "it",
        "this",
        "that",
        "these",
        "those",
        "etc",
        "n/a",
        "na",
        "none",
    }
    return name.lower() not in filler


def _build_graph(entity_map: dict, relation_map: dict) -> dict:
    """Build the final consolidated graph dict from aggregated maps."""
    entities = [
        {
            "name": e["name"],
            "type": e["type"],
            "description": e["descriptions"][0] if e["descriptions"] else "",
            "chunk_ids": sorted(e["chunk_ids"]),
            "id": e["id"],
        }
        for e in entity_map.values()
    ]
    relations = [
        {
            "source": r["source"],
            "source_type": r.get("source_type", ""),
            "target": r["target"],
            "target_type": r.get("target_type", ""),
            "type": r["type"],
            "description": r["descriptions"][0] if r["descriptions"] else "",
            "strength": round(sum(r["strengths"]) / len(r["strengths"]), 1),
            "chunk_ids": sorted(r["chunk_ids"]),
            "id": r["id"],
        }
        for r in relation_map.values()
    ]

    # Filter: remove meaningless or isolated entities
    connected = set()
    for r in relations:
        connected.add(r["source"].lower())
        connected.add(r["target"].lower())
    entities = [
        e
        for e in entities
        if _is_meaningful_name(e["name"]) and e["name"].lower() in connected
    ]

    return {"entities": entities, "relations": relations}


def _detect_clusters(consolidated: dict) -> tuple[list[set], list[dict]]:
    """Run Louvain and build the cluster payload for the labeling prompt."""
    g = nx.Graph()
    entity_idx = {}
    for e in consolidated.get("entities", []):
        nid = f"{e['name']}::{e['type']}" if e.get("type") else e["name"]
        g.add_node(nid)
        entity_idx[nid] = e
    for r in consolidated.get("relations", []):
        src = (
            f"{r['source']}::{r['source_type']}"
            if r.get("source_type")
            else r["source"]
        )
        tgt = (
            f"{r['target']}::{r['target_type']}"
            if r.get("target_type")
            else r["target"]
        )
        g.add_edge(src, tgt, weight=r.get("strength", 1))

    if g.number_of_nodes() == 0:
        return [], []

    clusters = louvain_communities(g, weight="weight", seed=0)
    payload = []
    for cid, members in enumerate(clusters):
        objs = [
            {
                "name": entity_idx.get(n, {}).get("name", n),
                "type": entity_idx.get(n, {}).get("type", ""),
                "description": entity_idx.get(n, {}).get("description", ""),
            }
            for n in sorted(members)
        ]
        payload.append({"id": str(cid), "members": objs})
    return clusters, payload


def _apply_clusters(consolidated: dict, clusters: list[set], labels: dict) -> dict:
    """Attach cluster info to the consolidated graph."""
    entity_idx = {}
    for e in consolidated.get("entities", []):
        nid = f"{e['name']}::{e['type']}" if e.get("type") else e["name"]
        entity_idx[nid] = e

    if not labels:
        labels = {}
        for cid, members in enumerate(clusters):
            type_counts: dict[str, int] = {}
            for nid in members:
                etype = (entity_idx.get(nid, {}).get("type") or "").strip().lower()
                if etype:
                    type_counts[etype] = type_counts.get(etype, 0) + 1
            topics = [
                t.replace("_", " ").capitalize()
                for t, _ in sorted(type_counts.items(), key=lambda x: -x[1])[:3]
            ]
            labels[str(cid)] = {"topics": topics or ["misc"], "description": ""}

    clusters_out = []
    for cid, members in enumerate(clusters):
        label = labels.get(str(cid), {"topics": ["misc"], "description": ""})
        clusters_out.append(
            {
                "id": cid,
                "topics": label.get("topics", ["misc"]),
                "description": label.get("description", ""),
                "members": sorted(members),
            }
        )
        for nid in members:
            if nid in entity_idx:
                entity_idx[nid]["cluster_id"] = cid

    consolidated["clusters"] = clusters_out
    return consolidated


# ═══════════════════════════════════════════════════════════════════════════
# Serialization helpers  (dict keys → JSON-safe)
# ═══════════════════════════════════════════════════════════════════════════


def _serialize_map(m: dict) -> dict:
    """Serialize a (tuple-keyed, set-valued) map to JSON-safe dict."""
    out = {}
    for k, v in m.items():
        sk = json.dumps(k)
        v2 = dict(v)
        if "chunk_ids" in v2 and isinstance(v2["chunk_ids"], set):
            v2["chunk_ids"] = sorted(v2["chunk_ids"])
        out[sk] = v2
    return out


def _deserialize_map(d: dict) -> dict:
    """Reverse of _serialize_map."""
    out = {}
    for sk, v in d.items():
        k = tuple(json.loads(sk))
        v2 = dict(v)
        if "chunk_ids" in v2:
            v2["chunk_ids"] = set(v2["chunk_ids"])
        out[k] = v2
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main processor
# ═══════════════════════════════════════════════════════════════════════════


class GeminiBatchProcessor:
    """Orchestrates the three-round Gemini batch pipeline."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        discovery_model: str,
        extraction_model: str,
        template: str = "default",
        consolidate: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batches_dir = output_dir / "batches"
        self.files_dir = output_dir / "files"
        self.discovery_model = discovery_model
        self.extraction_model = extraction_model
        self.template = template
        self.consolidate = consolidate

        self.config = Config(
            backend="gemini",
            extraction_model=extraction_model,
            discovery_model=discovery_model,
        )

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY", ""
        )
        self.client = GeminiBatchClient(api_key)

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(exist_ok=True)
        self.files_dir.mkdir(exist_ok=True)

        self.db = Database(output_dir / "batch_gemini.db")

    # ── Phase 0: scan & load ─────────────────────────────────────────────

    def scan_and_load(self):
        """Discover files, extract text, and chunk."""
        files = sorted(
            f
            for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )
        if not files:
            console.print(f"[red]No supported files in {self.input_dir}[/red]")
            sys.exit(1)

        console.print(f"Found [cyan]{len(files)}[/cyan] file(s)")

        for fp in files:
            fid = _file_id(fp)
            existing = self.db.get_file(fid)

            self.db.upsert_file(fid, str(fp), fp.name)

            if existing and existing.get("chunks_json"):
                n = len(json.loads(existing["chunks_json"]))
                console.print(f"  [dim]{fp.name} — cached ({n} chunks)[/dim]")
                continue

            console.print(f"  Loading [cyan]{fp.name}[/cyan] … ", end="")
            text = _load_text(fp)
            chunk_objs = chunk_text(text, self.config)
            console.print(f"[green]{len(chunk_objs)}[/green] chunks")

            (self.files_dir / fp.stem).mkdir(exist_ok=True)

            self.db.update_file(
                fid,
                text_content=text,
                chunks_json=json.dumps([c.text for c in chunk_objs]),
            )

    # ── Round status helpers ─────────────────────────────────────────────

    def _is_round_complete(self, round_name: str) -> bool:
        batch = self.db.get_batch(round_name)
        if not batch:
            return False
        return batch["status"] == "completed"

    def _has_round_started(self, round_name: str) -> bool:
        batch = self.db.get_batch(round_name)
        if not batch:
            return False
        return batch["status"] in ("submitted", "completed")

    # ── Phase 1: discovery ───────────────────────────────────────────────

    def round1(self):
        """Round 1 — language detection + schema discovery."""
        rnd = "round1_discovery"

        if self._is_round_complete(rnd):
            console.print("[green]✓[/green] Round 1 already completed")
            return
        if self._has_round_started(rnd):
            console.print("Resuming Round 1 polling …")
            self._poll_and_save(rnd)
            self._parse_round1()
            return

        files = self.db.get_all_files()
        reqs = []
        for f in files:
            fid, text = f["file_id"], f["text_content"]
            reqs.append(_gemini_request(f"{fid}|language", _prompt_language(text)))
            reqs.append(_gemini_request(f"{fid}|schema", _prompt_schema(text)))

        console.print(f"Submitting [cyan]{len(reqs)}[/cyan] requests …")
        self._submit(rnd, reqs, self.discovery_model)
        self._poll_and_save(rnd)
        self._parse_round1()

    def _parse_round1(self):
        responses = self._load_responses("round1_discovery")
        for f in self.db.get_all_files():
            fid = f["file_id"]

            lang_r = responses.get(f"{fid}|language")
            if lang_r:
                lang = (
                    parse_json_response(lang_r)
                    .get("language", DEFAULT_LANGUAGE)
                    .lower()
                    .strip()
                )
                if lang not in load_languages():
                    lang = DEFAULT_LANGUAGE
                self.db.update_file(fid, language=lang)

            schema_r = responses.get(f"{fid}|schema")
            if schema_r:
                p = parse_json_response(schema_r)
                if p.get("entity_types"):
                    schema = {
                        "entity_types": p["entity_types"][:20],
                        "relation_types": p.get("relation_types", [])[:25],
                        "reasoning": p.get("reasoning", ""),
                    }
                else:
                    schema = asdict(Schema.default())
                    del schema["discovery_time"]
                self.db.update_file(fid, schema_json=json.dumps(schema))

            console.print(
                f"  {f['file_name']}: lang=[cyan]{self.db.get_file(fid).get('language', '?')}[/cyan]  "
                f"schema types={len(json.loads(self.db.get_file(fid).get('schema_json', '{}') or '{}').get('entity_types', []))}"
            )

    # ── Phase 2: processing ──────────────────────────────────────────────

    def round2(self):
        """Round 2 — title, summary, rephrase, extraction."""
        rnd = "round2_processing"

        if self._is_round_complete(rnd):
            console.print("[green]✓[/green] Round 2 already completed")
            return
        if self._has_round_started(rnd):
            console.print("Resuming Round 2 polling …")
            self._poll_and_save(rnd)
            self._parse_round2()
            return

        files = self.db.get_all_files()
        reqs = []
        for f in files:
            fid = f["file_id"]
            chunks = json.loads(f["chunks_json"])
            sd = json.loads(f["schema_json"])
            schema = Schema(
                entity_types=sd["entity_types"],
                relation_types=sd["relation_types"],
                reasoning=sd.get("reasoning", ""),
            )

            # Set language so get_prompt returns localised templates
            set_language(f.get("language") or DEFAULT_LANGUAGE)

            reqs.append(_gemini_request(f"{fid}|title", _prompt_title(chunks)))
            reqs.append(_gemini_request(f"{fid}|summary", _prompt_summary(chunks)))
            for i, chunk in enumerate(chunks):
                reqs.append(
                    _gemini_request(f"{fid}|rephrase|{i}", _prompt_rephrase(chunk))
                )
            for i, chunk in enumerate(chunks):
                reqs.append(
                    _gemini_request(
                        f"{fid}|extract|{i}", _prompt_extraction(chunk, schema)
                    )
                )

        console.print(f"Submitting [cyan]{len(reqs)}[/cyan] requests …")
        self._submit(rnd, reqs, self.extraction_model)
        self._poll_and_save(rnd)
        self._parse_round2()

    def _parse_round2(self):
        responses = self._load_responses("round2_processing")
        encoder = get_encoder()

        for f in self.db.get_all_files():
            fid = f["file_id"]
            chunks = json.loads(f["chunks_json"])

            # Title
            tr = responses.get(f"{fid}|title")
            title = (
                (parse_json_response(tr).get("title") or f["file_name"]).strip()
                if tr
                else f["file_name"]
            )
            self.db.update_file(fid, title=title)

            # Summary
            sr = responses.get(f"{fid}|summary")
            summary = (
                (parse_json_response(sr).get("summary") or "").strip() if sr else ""
            )
            self.db.update_file(fid, summary=summary)

            # Rephrased chunks
            rephrased = []
            for i in range(len(chunks)):
                r = responses.get(f"{fid}|rephrase|{i}")
                rephrased.append(
                    parse_json_response(r).get("rephrase", chunks[i])
                    if r
                    else chunks[i]
                )
            self.db.update_file(fid, rephrased_json=json.dumps(rephrased))

            # Extractions
            extractions = []
            for i in range(len(chunks)):
                r = responses.get(f"{fid}|extract|{i}")
                p = parse_json_response(r) if r else {}
                extractions.append(
                    {
                        "id": str(uuid4()),
                        "chunk_idx": i,
                        "entities": p.get("entities", []),
                        "relations": p.get("relations", []),
                        "chunk_tokens": len(encoder.encode(chunks[i])),
                    }
                )
            self.db.update_file(fid, extractions_json=json.dumps(extractions))

            te = sum(len(e["entities"]) for e in extractions)
            tr_ = sum(len(e["relations"]) for e in extractions)
            console.print(
                f"  {f['file_name']}: title=[cyan]{title[:50]}[/cyan]  "
                f"entities={te}  relations={tr_}"
            )

    # ── Phase 3: consolidation + clusters ─────────────────────────────

    def round3(self):
        """Round 3 — summarisation + cluster labeling."""
        rnd = "round3_consolidation"

        if self._is_round_complete(rnd):
            console.print("[green]✓[/green] Round 3 already completed")
            self._finalise()
            return
        if self._has_round_started(rnd):
            console.print("Resuming Round 3 polling …")
            self._poll_and_save(rnd)
            self._parse_round3()
            return

        files = self.db.get_all_files()
        reqs: list[dict] = []

        for f in files:
            fid = f["file_id"]
            extractions = json.loads(f["extractions_json"])
            set_language(f.get("language") or DEFAULT_LANGUAGE)

            # Aggregate locally
            emap, rmap = _aggregate(extractions)
            self.db.update_file(
                fid,
                agg_entities_json=json.dumps(_serialize_map(emap)),
                agg_relations_json=json.dumps(_serialize_map(rmap)),
            )

            # Summarization requests
            to_summ = _items_to_summarize(emap, rmap)
            for bi in range(0, max(len(to_summ), 1), SUMMARIZATION_BATCH_SIZE):
                batch_items = to_summ[bi : bi + SUMMARIZATION_BATCH_SIZE]
                if batch_items:
                    idx = bi // SUMMARIZATION_BATCH_SIZE
                    reqs.append(
                        _gemini_request(
                            f"{fid}|summarize|{idx}",
                            _prompt_summarize(batch_items),
                        )
                    )

            # Cluster detection (local) + labeling request
            pre_graph = _build_graph(emap, rmap)
            clusters, cpayload = _detect_clusters(pre_graph)
            clusters_ser = [sorted(c) for c in clusters]
            self.db.update_file(fid, clusters_json=json.dumps(clusters_ser))

            if cpayload:
                reqs.append(
                    _gemini_request(
                        f"{fid}|cluster",
                        _prompt_cluster(cpayload),
                    )
                )

        if not reqs:
            console.print("[dim]No LLM requests needed for Round 3[/dim]")
            self.db.upsert_batch(rnd, status="completed")
            self._finalise()
            return

        console.print(f"Submitting [cyan]{len(reqs)}[/cyan] requests …")
        self._submit(rnd, reqs, self.extraction_model)
        self._poll_and_save(rnd)
        self._parse_round3()

    def _parse_round3(self):
        responses = self._load_responses("round3_consolidation")

        for f in self.db.get_all_files():
            fid = f["file_id"]

            # Collect summaries
            summaries: dict[str, str] = {}
            for key, val in responses.items():
                if key.startswith(f"{fid}|summarize|"):
                    summaries.update(parse_json_response(val).get("summaries", {}))
            self.db.update_file(fid, summaries_json=json.dumps(summaries))

        self._finalise()

    # ── Finalise ─────────────────────────────────────────────────────────

    def _finalise(self):
        """Assemble per-file graph.json + HTML."""
        responses = self._load_responses("round3_consolidation")

        for f in self.db.get_all_files():
            fid = f["file_id"]

            # Re-read latest data
            f = self.db.get_file(fid)

            chunks = json.loads(f["chunks_json"])
            rephrased = json.loads(f.get("rephrased_json") or "[]")
            extractions = json.loads(f["extractions_json"])
            schema_data = json.loads(f["schema_json"])
            summaries = json.loads(f.get("summaries_json") or "{}")

            # Deserialise aggregated maps
            emap = _deserialize_map(json.loads(f.get("agg_entities_json") or "{}"))
            rmap = _deserialize_map(json.loads(f.get("agg_relations_json") or "{}"))

            # Apply summaries
            emap, rmap = _apply_summaries(emap, rmap, summaries)

            # Build final consolidated graph
            consolidated = _build_graph(emap, rmap)

            # Apply cluster labels
            clusters_data = json.loads(f.get("clusters_json") or "[]")
            clusters = [set(c) for c in clusters_data]

            comm_resp = responses.get(f"{fid}|cluster")
            comm_labels = (
                parse_json_response(comm_resp).get("clusters", {})
                if comm_resp
                else {}
            )

            if clusters:
                consolidated = _apply_clusters(consolidated, clusters, comm_labels)
            else:
                consolidated["clusters"] = []

            # Assemble output (same schema as knwler's graph.json)
            doc_id = str(uuid4())
            final_chunks = [
                {
                    "id": ext["id"],
                    "chunk_idx": ext["chunk_idx"],
                    "text": (
                        chunks[ext["chunk_idx"]]
                        if ext["chunk_idx"] < len(chunks)
                        else ""
                    ),
                    "rephrase": (
                        rephrased[ext["chunk_idx"]]
                        if ext["chunk_idx"] < len(rephrased)
                        else ""
                    ),
                    "entities": ext["entities"],
                    "relations": ext["relations"],
                    "chunk_time": 0.0,
                    "chunk_tokens": ext.get("chunk_tokens", 0),
                    "document": doc_id,
                }
                for ext in extractions
            ]

            output = {
                "id": doc_id,
                "title": f.get("title") or f["file_name"],
                "summary": f.get("summary") or "",
                "url": None,
                "language": f.get("language") or DEFAULT_LANGUAGE,
                "schema": schema_data,
                "stats": [
                    {
                        "run": {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "file": f["file_path"],
                            "backend": "gemini_batch",
                            "extraction_model": self.extraction_model,
                            "discovery_model": self.discovery_model,
                        }
                    }
                ],
                "graph": consolidated,
                "chunks": final_chunks,
            }

            # Write files
            file_dir = self.files_dir / Path(f["file_name"]).stem
            file_dir.mkdir(exist_ok=True)

            graph_path = file_dir / "graph.json"
            graph_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

            try:
                html = export_html(output, template=self.template)
                (file_dir / "index.html").write_text(html, encoding="utf-8")
            except Exception as exc:
                console.print(
                    f"  [yellow]HTML export failed for {f['file_name']}: {exc}[/yellow]"
                )

            self.db.update_file(fid, graph_output_json="done")
            console.print(
                f"  [green]✓[/green] {f['file_name']} → [cyan]{graph_path}[/cyan]"
            )

    # ── Batch helpers ────────────────────────────────────────────────────

    def _submit(self, round_name: str, requests: list[dict], model: str):
        """Write JSONL, upload via File API, and create a Gemini batch job."""
        jsonl_path = self.batches_dir / f"{round_name}_input.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for r in requests:
                fh.write(json.dumps(r) + "\n")

        # Validate file size (2 GB limit)
        file_size = jsonl_path.stat().st_size
        if file_size > MAX_BATCH_FILE_SIZE_BYTES:
            raise ValueError(
                f"Batch input file {jsonl_path.name} is "
                f"{file_size / (1024 ** 3):.2f} GB, exceeding the "
                f"2 GB limit. Try processing fewer documents at once."
            )

        console.print(
            f"  Total: {len(requests):,} requests "
            f"({file_size / (1024 ** 2):.1f} MB)"
        )

        file_name = self.client.upload_jsonl(jsonl_path)
        console.print(f"  Uploaded → file [dim]{file_name}[/dim]")

        job_name = self.client.create_batch(
            model=model,
            file_name=file_name,
            display_name=round_name,
        )
        console.print(f"  Batch job created → [dim]{job_name}[/dim]")

        self.db.upsert_batch(
            round_name,
            job_name=job_name,
            input_file_name=file_name,
            status="submitted",
            request_count=len(requests),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _poll_and_save(self, round_name: str):
        """Poll a batch job until terminal and save outputs."""
        info = self.db.get_batch(round_name)
        job_name = info["job_name"]

        console.print(f"  Polling job [dim]{job_name}[/dim] …")
        result = self.client.poll_batch(job_name)
        state = (
            result.state.name if hasattr(result.state, "name") else str(result.state)
        )

        if state == "JOB_STATE_SUCCEEDED":
            # Download the result file
            if result.dest and result.dest.file_name:
                content_bytes = self.client.download_file(result.dest.file_name)
                content = (
                    content_bytes.decode("utf-8")
                    if isinstance(content_bytes, bytes)
                    else str(content_bytes)
                )
                (self.batches_dir / f"{round_name}_output.jsonl").write_text(
                    content, encoding="utf-8"
                )
                self.db.upsert_batch(
                    round_name,
                    status="completed",
                    output_file_name=result.dest.file_name,
                    completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                console.print(f"  [green]✓[/green] Batch {round_name} complete")
            else:
                # Inline responses (shouldn't happen with file-based, but handle gracefully)
                self.db.upsert_batch(
                    round_name,
                    status="completed",
                    completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                console.print(
                    f"  [green]✓[/green] Batch {round_name} complete (no output file)"
                )
        else:
            error_msg = ""
            if hasattr(result, "error") and result.error:
                error_msg = str(result.error)
            error_details = json.dumps(
                {
                    "state": state,
                    "error": error_msg,
                }
            )
            self.db.upsert_batch(
                round_name,
                status=state.lower(),
                error_details=error_details,
            )
            console.print(f"  [red]Batch {round_name} ended with state: {state}[/red]")
            if error_msg:
                console.print(f"  [red]{error_msg}[/red]")

            raise RuntimeError(
                f"Batch {round_name} ended with state '{state}'. "
                "Check the errors and re-run to retry."
            )

    def _load_responses(self, round_name: str) -> dict[str, str]:
        """Parse JSONL output file → {key: content_text}."""
        path = self.batches_dir / f"{round_name}_output.jsonl"
        if not path.exists():
            return {}
        out: dict[str, str] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key")
            if not key:
                continue
            # Gemini file-based output: {"key": "...", "response": {"candidates": [...]}}
            resp = obj.get("response")
            if resp and isinstance(resp, dict):
                candidates = resp.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        out[key] = parts[0].get("text", "")
                    else:
                        console.print(
                            f"  [yellow]Request {key}: no parts in response[/yellow]"
                        )
                else:
                    console.print(f"  [yellow]Request {key}: no candidates[/yellow]")
            elif "error" in obj:
                console.print(
                    f"  [yellow]Request {key} failed: {obj['error']}[/yellow]"
                )
        return out

    # ── Phase 4: cross-file consolidation ─────────────────────────────

    def _collect_file_graphs(self) -> list[dict]:
        """Return all written per-file graph dicts (from _finalise)."""
        graphs = []
        for f in self.db.get_all_files():
            graph_path = self.files_dir / Path(f["file_name"]).stem / "graph.json"
            if graph_path.exists():
                graphs.append(json.loads(graph_path.read_text()))
        return graphs

    def round4(self):
        """Round 4 — cross-file consolidation with batch-submitted summarization."""
        rnd = "round4_cross_consolidation"

        if self._is_round_complete(rnd):
            console.print("[green]✓[/green] Round 4 already completed")
            return
        if self._has_round_started(rnd):
            console.print("Resuming Round 4 polling …")
            self._poll_and_save(rnd)
            self._parse_round4()
            return

        graphs = self._collect_file_graphs()
        if not graphs:
            console.print(
                "[yellow]⚠ No per-file graphs found — skipping cross-consolidation.[/yellow]"
            )
            return
        if len(graphs) == 1:
            console.print(
                "[dim]Only one document — writing consolidated_graph.json directly.[/dim]"
            )
            self._write_consolidated(graphs, {}, {}, [], {})
            self.db.upsert_batch(
                rnd,
                status="completed",
                request_count=0,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return

        console.print(f"Aggregating [cyan]{len(graphs)}[/cyan] document graph(s) …")

        extractions = [
            {
                "id": g.get("id", str(uuid4())),
                "entities": g.get("graph", {}).get("entities", []),
                "relations": g.get("graph", {}).get("relations", []),
            }
            for g in graphs
        ]
        emap, rmap = _aggregate(extractions)

        (self.batches_dir / "cross_agg_entities.json").write_text(
            json.dumps(_serialize_map(emap)), encoding="utf-8"
        )
        (self.batches_dir / "cross_agg_relations.json").write_text(
            json.dumps(_serialize_map(rmap)), encoding="utf-8"
        )

        to_summ = _items_to_summarize(emap, rmap)
        reqs: list[dict] = []
        for bi in range(0, max(len(to_summ), 1), SUMMARIZATION_BATCH_SIZE):
            batch_items = to_summ[bi : bi + SUMMARIZATION_BATCH_SIZE]
            if batch_items:
                idx = bi // SUMMARIZATION_BATCH_SIZE
                reqs.append(
                    _gemini_request(
                        f"cross|summarize|{idx}",
                        _prompt_summarize(batch_items),
                    )
                )

        pre_graph = _build_graph(emap, rmap)
        clusters, cpayload = _detect_clusters(pre_graph)
        (self.batches_dir / "cross_clusters.json").write_text(
            json.dumps([sorted(c) for c in clusters]), encoding="utf-8"
        )
        if cpayload:
            reqs.append(
                _gemini_request(
                    "cross|cluster",
                    _prompt_cluster(cpayload),
                )
            )

        if not reqs:
            console.print("[dim]No LLM requests needed for Round 4[/dim]")
            emap, rmap = _apply_summaries(emap, rmap, {})
            self._write_consolidated(graphs, emap, rmap, clusters, {})
            self.db.upsert_batch(
                rnd,
                status="completed",
                request_count=0,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return

        console.print(f"Submitting [cyan]{len(reqs)}[/cyan] requests …")
        self._submit(rnd, reqs, self.extraction_model)
        self._poll_and_save(rnd)
        self._parse_round4()

    def _parse_round4(self):
        """Apply batch results and write consolidated_graph.json."""
        graphs = self._collect_file_graphs()

        emap = _deserialize_map(
            json.loads((self.batches_dir / "cross_agg_entities.json").read_text())
        )
        rmap = _deserialize_map(
            json.loads((self.batches_dir / "cross_agg_relations.json").read_text())
        )

        responses = self._load_responses("round4_cross_consolidation")
        summaries: dict[str, str] = {}
        for key, val in responses.items():
            if key.startswith("cross|summarize|"):
                summaries.update(parse_json_response(val).get("summaries", {}))

        emap, rmap = _apply_summaries(emap, rmap, summaries)

        clusters_data = json.loads(
            (self.batches_dir / "cross_clusters.json").read_text()
        )
        clusters = [set(c) for c in clusters_data]
        cluster_resp = responses.get("cross|cluster")
        cluster_labels = (
            parse_json_response(cluster_resp).get("clusters", {}) if cluster_resp else {}
        )

        self._write_consolidated(graphs, emap, rmap, clusters, cluster_labels)

    def _write_consolidated(
        self,
        graphs: list[dict],
        emap: dict,
        rmap: dict,
        clusters: list[set],
        cluster_labels: dict,
    ) -> None:
        """Build and write consolidated_graph.json (+ HTML report)."""
        consolidated = (
            _build_graph(emap, rmap) if (emap or rmap) else graphs[0].get("graph", {})
        )

        if clusters:
            consolidated = _apply_clusters(consolidated, clusters, cluster_labels)
        elif "clusters" not in consolidated:
            consolidated["clusters"] = []

        documents = [
            {
                "id": g.get("id"),
                "title": g.get("title"),
                "summary": g.get("summary"),
                "url": g.get("url"),
                "language": g.get("language"),
            }
            for g in graphs
        ]
        entity_types = sorted(
            {et for g in graphs for et in g.get("schema", {}).get("entity_types", [])}
        )
        relation_types = sorted(
            {rt for g in graphs for rt in g.get("schema", {}).get("relation_types", [])}
        )

        output = {
            "id": str(uuid4()),
            "documents": documents,
            "schema": {"entity_types": entity_types, "relation_types": relation_types},
            "graph": consolidated,
            "chunks": [],
        }

        out_path = self.output_dir / "consolidated_graph.json"
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        console.print(
            f"[green]✓[/green] Consolidated graph ({len(consolidated.get('entities', []))} entities, "
            f"{len(consolidated.get('relations', []))} relations) → [cyan]{out_path}[/cyan]"
        )

        try:
            html = export_html(output, template=self.template)
            html_path = self.output_dir / "consolidated_graph.html"
            html_path.write_text(html, encoding="utf-8")
            console.print(
                f"[green]✓[/green] Consolidated report → [cyan]{html_path}[/cyan]"
            )
        except Exception as exc:
            console.print(f"  [yellow]HTML export failed: {exc}[/yellow]")

    # ── Status ───────────────────────────────────────────────────────────

    def print_status(self):
        files = self.db.get_all_files()

        table = Table(title="Files", show_lines=True)
        table.add_column("File", style="cyan")
        table.add_column("Lang")
        table.add_column("Chunks", justify="right")
        table.add_column("Title", max_width=40)
        table.add_column("Done")

        for f in files:
            n = len(json.loads(f["chunks_json"])) if f.get("chunks_json") else 0
            done = "✓" if f.get("graph_output_json") else "…"
            table.add_row(
                f["file_name"],
                f.get("language") or "—",
                str(n),
                (f.get("title") or "—")[:40],
                done,
            )
        console.print(table)

        console.print()
        rounds = [
            "round1_discovery",
            "round2_processing",
            "round3_consolidation",
            "round4_cross_consolidation",
        ]
        for rnd in rounds:
            b = self.db.get_batch(rnd)
            if b:
                status = b["status"]
                color = "green" if status == "completed" else "yellow"
                console.print(
                    f"  {rnd}: [{color}]{status}[/] ({b.get('request_count', 0)} requests)"
                )
                if status not in ("completed", "submitted", "pending") and b.get(
                    "error_details"
                ):
                    details = json.loads(b["error_details"])
                    if details.get("error"):
                        console.print(f"    [red]{details['error']}[/red]")
            else:
                console.print(f"  {rnd}: [dim]not started[/dim]")

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self):
        console.print(
            Panel.fit(
                "[bold]Knwler — Gemini Batch Processor[/bold]\n"
                f"Input:      [cyan]{self.input_dir}[/cyan]\n"
                f"Output:     [cyan]{self.output_dir}[/cyan]\n"
                f"Discovery:  [green]{self.discovery_model}[/green]\n"
                f"Extraction: [green]{self.extraction_model}[/green]\n"
                f"Consolidate:[{'green' if self.consolidate else 'dim'}] {self.consolidate}[/]",
                border_style="blue",
            )
        )

        console.print()
        console.rule("[bold cyan]Phase 0 · Scan & Load[/bold cyan]")
        self.scan_and_load()

        console.print()
        console.rule("[bold cyan]Phase 1 · Discovery[/bold cyan]")
        self.round1()

        console.print()
        console.rule("[bold cyan]Phase 2 · Processing[/bold cyan]")
        self.round2()

        console.print()
        console.rule(
            "[bold cyan]Phase 3 · Per-file Consolidation & Clusters[/bold cyan]"
        )
        self.round3()

        if self.consolidate:
            console.print()
            console.rule("[bold cyan]Phase 4 · Cross-file Consolidation[/bold cyan]")
            self.round4()

        console.print()
        console.rule("[bold green]Pipeline Complete[/bold green]")
        self.print_status()

    def close(self):
        self.db.close()


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


batch_gemini_app = typer.Typer(
    help="Process documents with Knwler via Google Gemini Batch API.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@batch_gemini_app.command(
    "run",
    help="Run the full Gemini batch pipeline (optionally with cross-file consolidation).",
)
def cmd_run(
    input: Annotated[
        Path,
        typer.Option("--input", "-i", help="Directory containing files to process."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for results."),
    ],
    discovery_model: Annotated[
        str,
        typer.Option("--discovery-model", help="Model for schema discovery."),
    ] = DEFAULT_GEMINI_DISCOVERY_MODEL,
    extraction_model: Annotated[
        str,
        typer.Option("--extraction-model", help="Model for extraction steps."),
    ] = DEFAULT_GEMINI_EXTRACTION_MODEL,
    template: Annotated[
        str, typer.Option("--template", help="HTML report template.")
    ] = "default",
    consolidate: Annotated[
        bool,
        typer.Option(
            "--consolidate",
            help="After per-file processing, merge all document graphs into a single "
            "consolidated_graph.json using a fourth batch round for description summarization.",
        ),
    ] = False,
) -> None:
    """Run the Gemini Batch API pipeline (3 rounds + optional cross-file consolidation)."""
    if not input.is_dir():
        console.print(f"[red]\u2717[/red] Not a directory: [cyan]{input}[/cyan]")
        raise typer.Exit(1)

    proc = GeminiBatchProcessor(
        input_dir=input,
        output_dir=output,
        discovery_model=discovery_model,
        extraction_model=extraction_model,
        template=template,
        consolidate=consolidate,
    )
    try:
        proc.run()
    finally:
        proc.close()


@batch_gemini_app.command(
    "consolidate",
    help="Run cross-file consolidation on an already-processed output directory.",
)
def cmd_consolidate(
    input: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Original input directory (needed to locate batch_gemini.db).",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o", help="Output directory produced by a previous 'run'."
        ),
    ],
    discovery_model: Annotated[
        str,
        typer.Option("--discovery-model", help="Model for schema discovery."),
    ] = DEFAULT_GEMINI_DISCOVERY_MODEL,
    extraction_model: Annotated[
        str,
        typer.Option(
            "--extraction-model", help="Model used for summarization requests."
        ),
    ] = DEFAULT_GEMINI_EXTRACTION_MODEL,
    template: Annotated[
        str, typer.Option("--template", help="HTML report template.")
    ] = "default",
) -> None:
    """Merge all per-file graphs from a previous run into consolidated_graph.json."""
    if not input.is_dir():
        console.print(f"[red]\u2717[/red] Not a directory: [cyan]{input}[/cyan]")
        raise typer.Exit(1)
    if not output.is_dir():
        console.print(
            f"[red]\u2717[/red] Output directory not found: [cyan]{output}[/cyan]"
        )
        raise typer.Exit(1)

    proc = GeminiBatchProcessor(
        input_dir=input,
        output_dir=output,
        discovery_model=discovery_model,
        extraction_model=extraction_model,
        template=template,
        consolidate=True,
    )
    try:
        proc.round4()
    finally:
        proc.close()


@batch_gemini_app.command("status", help="Show the current pipeline status.")
def cmd_status(
    input: Annotated[
        Path,
        typer.Option("--input", "-i", help="Directory containing files to process."),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o", help="Output directory (where batch_gemini.db lives)."
        ),
    ],
    discovery_model: Annotated[
        str,
        typer.Option("--discovery-model", help="Model for schema discovery."),
    ] = DEFAULT_GEMINI_DISCOVERY_MODEL,
    extraction_model: Annotated[
        str,
        typer.Option("--extraction-model", help="Model for extraction steps."),
    ] = DEFAULT_GEMINI_EXTRACTION_MODEL,
) -> None:
    """Show current pipeline status for an existing output directory."""
    if not input.is_dir():
        console.print(f"[red]\u2717[/red] Not a directory: [cyan]{input}[/cyan]")
        raise typer.Exit(1)

    proc = GeminiBatchProcessor(
        input_dir=input,
        output_dir=output,
        discovery_model=discovery_model,
        extraction_model=extraction_model,
    )
    try:
        proc.print_status()
    finally:
        proc.close()


if __name__ == "__main__":
    batch_gemini_app()
