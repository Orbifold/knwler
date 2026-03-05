"""
Graph consolidation: merge chunk results, summarize descriptions, filter.
"""

import re
import time
from uuid import uuid4
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
import json
from knwler.config import Config, console
from knwler.models import ExtractionResult, Schema, Graph

from knwler.language import get_console_msg, get_prompt
from knwler.llm import llm_generate, parse_json_response
from knwler.models import Schema, Graph
from dataclasses import dataclass, asdict, fields, field


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def consolidate_graphs(
    graphs: list[dict],
    cluster: bool = False,
    include_chunks: bool = False,
    filter_low_importance: bool = True,
    config: Config = Config(),
) -> dict:
    """
    Consolidate multiple chunk graphs into one, with summarization and filtering.
    This reuses the same consolidation logic as the one to merge chunk graphs into a document graph, but at a higher level to merge multiple document graphs into a single consolidated graph.

    The clustering flag is False by default since it's assumed that the merge of a a large amount of graphs is towards a database ingestion where clustering is more scalable.
    The clustering is also a reuse of the same community detection and labeling logic as in the community analysis step, but applied at the consolidation level to group similar entities together across documents.

    The chunks can be consolidated as well but typically one would move the chunks into a vector database.
    """
    if not graphs or len(graphs) == 0:
        return None
    if len(graphs) == 1:
        return graphs[0]
    # ensure we have dicts
    for i in range(len(graphs)):
        if isinstance(graphs[i], Graph):
            graphs[i] = asdict(graphs[i])
    # concatenate the chunks
    all_chunks = []
    for g in graphs:
        chunks = g.get("chunks", [])
        console.print(
            f"[green]\u2713[/green] Graph [cyan]{g.get('id', 'unknown')}[/cyan]: "
            f"{len(chunks)} chunk(s), {len(g.get('graph', {}).get('entities', []))} entities, "
            f"{len(g.get('graph', {}).get('relations', []))} relations."
        )
        all_chunks.extend(chunks)
    all_documents = []
    for g in graphs:
        all_documents.append(
            {
                "id": g.get("id"),
                "title": g.get("title"),
                "summary": g.get("summary"),
                "url": g.get("url"),
                "language": g.get("language"),
            }
        )
    all_schema = {
        "entity_types": set(),
        "relation_types": set(),
    }
    for g in graphs:
        schema = g.get("schema", {})
        all_schema["entity_types"].update(schema.get("entity_types", []))
        all_schema["relation_types"].update(schema.get("relation_types", []))
    all_schema["entity_types"] = sorted(all_schema["entity_types"])
    all_schema["relation_types"] = sorted(all_schema["relation_types"])
    console.print(
        f"[green]\u2713[/green] Aggregated [cyan]{len(all_chunks)}[/cyan] chunks, [cyan]{len(all_documents)}[/cyan] documents, "
        f"[cyan]{len(all_schema['entity_types'])}[/cyan] entity types, [cyan]{len(all_schema['relation_types'])}[/cyan] relation types from all graphs."
    )
    consolidated, consolidation_time = consolidate_extracted_graphs(
        [
            Graph(
                entities=g.get("graph", {}).get("entities", []),
                relations=g.get("graph", {}).get("relations", []),
            )
            for g in graphs
        ],
        config,
        summarize=True,
        filter_low_importance=filter_low_importance,
    )
    if cluster:
        from knwler.community import analyze_communities

        consolidated = analyze_communities(consolidated, config)

    return {
        "id": str(uuid4()),
        "documents": all_documents,
        "schema": all_schema,
        "graph": consolidated,
        "chunks": all_chunks if include_chunks else [],
    }


def consolidate_extracted_graphs(
    little_graphs: list[Graph],
    config: Config = Config(),
    summarize: bool = True,
    filter_low_importance: bool = True,
) -> tuple[dict, float]:
    """Consolidate chunk graphs with unique (name, type) and summarized descriptions.

    Returns ``(consolidated_graph, consolidation_time)``.
    """
    t0 = time.perf_counter()

    # Phase 1: Aggregate by key
    entity_map: dict[tuple[str, str], dict] = {}
    relation_map: dict[tuple[str, str, str], dict] = {}

    for r in little_graphs:
        for e in r.entities:
            name = (e.get("name") or "").strip()
            etype = (e.get("type") or "").strip()
            desc = (e.get("description") or "").strip()
            id = e.get("id", str(uuid4()))
            if not name or not etype:
                continue
            # this is crucial for disambiguation: we consider that two entities with the same name but different types are different,
            # and two entities with the same type but different names are different,
            # but if both name and type are the same, then they are the same entity and we will merge their descriptions and chunk_ids.
            key = (name.lower(), etype.lower())
            if key not in entity_map:
                entity_map[key] = {
                    "id": id,
                    "name": name,
                    "type": etype,
                    "descriptions": [],
                    "chunk_ids": set(),
                }
            # ============================================================================================
            # This method is used both for consolidating chunks and consolidating document graphs.
            # In the first case, the chunk_ids are used to keep track of which chunks contributed to the consolidated entity,
            # and in the second case, the chunk_ids are used to keep track of which documents contributed to the consolidated entity (since we reuse the same consolidation logic at both levels).
            # ============================================================================================
            if e.get("chunk_ids"):
                entity_map[key]["chunk_ids"].update(e.get("chunk_ids", set()))
            else:
                if isinstance(r, ExtractionResult):
                    entity_map[key]["chunk_ids"].add(r.id)
            if desc and desc not in entity_map[key]["descriptions"]:
                entity_map[key]["descriptions"].append(desc)

        for rel in r.relations:
            if isinstance(rel, str):
                console.print(
                    f"[yellow]\u26a0 Skipping badly formatted relation string: {rel}[/yellow]"
                )
                continue
            src = (rel.get("source") or "").strip()
            src_type = (rel.get("source_type") or "").strip()
            tgt = (rel.get("target") or "").strip()
            tgt_type = (rel.get("target_type") or "").strip()
            rtype = (rel.get("type") or "").strip()
            desc = (rel.get("description") or "").strip()
            strength = rel.get("strength", 5)
            id = rel.get("id", str(uuid4()))

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
                    "id": id,
                    "source": src,
                    "source_type": src_type,
                    "target": tgt,
                    "target_type": tgt_type,
                    "type": rtype,
                    "descriptions": [],
                    "strengths": [],
                    "chunk_ids": set(),
                }
            if rel.get("chunk_ids"):
                relation_map[key]["chunk_ids"].update(rel.get("chunk_ids", set()))
            else:
                if isinstance(r, ExtractionResult):
                    relation_map[key]["chunk_ids"].add(r.id)
            relation_map[key]["strengths"].append(strength)
            if desc and desc not in relation_map[key]["descriptions"]:
                relation_map[key]["descriptions"].append(desc)

    # Phase 2: Summarize descriptions that need merging
    if summarize:
        entity_map, relation_map = _summarize_descriptions(
            entity_map, relation_map, config
        )
    else:
        # if not summarizing, just merge multiple descriptions into one string to avoid losing information, but without hitting the LLM
        for e in entity_map:
            if len(entity_map[e]["descriptions"]) > 1:
                entity_map[e]["descriptions"] = [
                    " ".join(entity_map[e]["descriptions"])
                ]
        for r in relation_map:
            if len(relation_map[r]["descriptions"]) > 1:
                relation_map[r]["descriptions"] = [
                    " ".join(relation_map[r]["descriptions"])
                ]

    # Phase 3: Build final output
    entities = [
        {
            "name": e["name"],
            "type": e["type"],
            "description": e["descriptions"][0] if e["descriptions"] else "",
            "chunk_ids": sorted(e["chunk_ids"]),
            "id": e.get("id", str(uuid4())),
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
            "id": r.get("id", str(uuid4())),
        }
        for r in relation_map.values()
    ]

    # Phase 4: Remove low-importance nodes
    if filter_low_importance:
        entities, relations = _filter_low_importance_nodes(entities, relations)

    elapsed = time.perf_counter() - t0
    return {"entities": entities, "relations": relations}, elapsed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _is_meaningful_name(name: str) -> bool:
    """Check if an entity name is meaningful (not just numbers, punctuation, etc.)."""
    name = name.strip()
    if not name:
        return False
    if re.match(r"^[\d.,\-+%$€£¥]+$", name):
        return False
    if len(name) == 1 and not name.isalpha():
        return False
    if re.match(r"^[\W_]+$", name):
        return False

    filler_words = {
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
    if name.lower() in filler_words:
        return False

    return True


def _filter_low_importance_nodes(
    entities: list[dict],
    relations: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Remove low-importance entities: meaningless names or zero degree."""
    connected = set()
    for r in relations:
        connected.add(r["source"].lower())
        connected.add(r["target"].lower())

    original_count = len(entities)
    filtered_entities = [
        e
        for e in entities
        if _is_meaningful_name(e["name"]) and e["name"].lower() in connected
    ]

    removed = original_count - len(filtered_entities)
    if removed > 0:
        msg = (
            get_console_msg("removed_entities", count=removed)
            or f"Removed {removed} low-importance entities (meaningless name or zero degree)"
        )
        console.print(f"[dim]{msg}[/dim]")

    return filtered_entities, relations


def _summarize_descriptions(
    entity_map: dict,
    relation_map: dict,
    config: Config,
) -> tuple[dict, dict]:
    """Batch summarize entities/relations with multiple descriptions."""

    to_summarize: list[dict] = []
    for key, e in entity_map.items():
        if len(e["descriptions"]) > 1:
            to_summarize.append(
                {
                    "id": f"entity:{key[0]}:{key[1]}",
                    "name": e["name"],
                    "type": e["type"],
                    "descriptions": e["descriptions"],
                }
            )
    for key, r in relation_map.items():
        if len(r["descriptions"]) > 1:
            to_summarize.append(
                {
                    "id": f"relation:{key[0]}:{key[1]}:{key[2]}",
                    "name": f"{r['source']} -> {r['target']}",
                    "type": r["type"],
                    "descriptions": r["descriptions"],
                }
            )

    if not to_summarize:
        return entity_map, relation_map

    batch_size = 20
    summaries: dict[str, str] = {}
    total_batches = (len(to_summarize) + batch_size - 1) // batch_size
    progress_msg = (
        get_console_msg("summarizing", count=len(to_summarize))
        or f"Summarizing {len(to_summarize)} items..."
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"[cyan]{progress_msg}", total=total_batches)

        for i in range(0, len(to_summarize), batch_size):
            batch = to_summarize[i : i + batch_size]
            prompt = _build_summarization_prompt(batch)
            response = llm_generate(prompt, config, model=config.extraction_model)
            result = parse_json_response(response)
            summaries.update(result.get("summaries", {}))
            progress.update(task, advance=1)

    # Apply summaries back
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


def _build_summarization_prompt(items: list[dict]) -> str:
    """Build prompt for batch description summarization."""
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
            "Return JSON:\n"
            "{\n"
            '  "summaries": {\n'
            '    "<id>": "<merged description>",\n'
            "    ...\n"
            "  }\n"
            "}"
        )
    return prompt
