#!/usr/bin/env python3
"""
Export knowledge-graph JSON files to JSON-LD (RDF).

Usage:
    uv run jsonld_export.py graph.json [graph2.json ...]
    uv run jsonld_export.py graph.json --output ./exports/

The exporter maps the knwler data model onto standard RDF vocabularies:
  - Documents  → schema:CreativeWork  (+ dcterms metadata)
  - Entities   → skos:Concept  (typed via knwler:<EntityClass>)
  - Relations  → reified knwler:Relation resources  (source/target are @id refs)
  - Chunks     → knwler:Chunk  (schema:text, schema:position)
  - Clusters   → skos:Collection  (skos:member links to Entity nodes)

Namespaces used:
  knwler:   https://knwler.com/ontology#
  schema:   https://schema.org/
  dcterms:  http://purl.org/dc/terms/
  skos:     http://www.w3.org/2004/02/skos/core#
  rdfs:     http://www.w3.org/2000/01/rdf-schema#
  xsd:      http://www.w3.org/2001/XMLSchema#

Relation types from the extraction schema become knwler predicates, e.g.
  "is_part_of"  → knwler:isPartOf
  "protects"    → knwler:protects

In addition to the reified Relation resource, a shorthand direct triple is
emitted on the source entity so RDF reasoners can traverse the graph with
standard SPARQL without unpacking reifications.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Annotated
from uuid import uuid4

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

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

# ---------------------------------------------------------------------------
# Namespace helpers
# ---------------------------------------------------------------------------

KNWLER_NS = "https://knwler.com/ontology#"
SCHEMA_NS = "https://schema.org/"
DCTERMS_NS = "http://purl.org/dc/terms/"
SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"

# ---------------------------------------------------------------------------
# JSON-LD @context
# ---------------------------------------------------------------------------

JSONLD_CONTEXT = {
    # Prefix declarations
    "knwler": KNWLER_NS,
    "schema": SCHEMA_NS,
    "dcterms": DCTERMS_NS,
    "skos": SKOS_NS,
    "rdfs": RDFS_NS,
    "xsd": XSD_NS,
    # ---- Document ----
    "Document": "knwler:Document",
    "title": {"@id": "dcterms:title"},
    "summary": {"@id": "dcterms:description"},
    "url": {"@id": "schema:url", "@type": "@id"},
    "language": {"@id": "dcterms:language"},
    "extractionModel": {"@id": "knwler:extractionModel"},
    "discoveryModel": {"@id": "knwler:discoveryModel"},
    "timestamp": {"@id": "dcterms:created", "@type": "xsd:dateTime"},
    "numChunks": {"@id": "knwler:numChunks", "@type": "xsd:integer"},
    # ---- Schema (ontology declaration) ----
    "KnowledgeSchema": "knwler:KnowledgeSchema",
    "entityTypes": {"@id": "knwler:entityTypes", "@container": "@list"},
    "relationTypes": {"@id": "knwler:relationTypes", "@container": "@list"},
    "schemaReasoning": {"@id": "knwler:schemaReasoning"},
    # ---- Chunk ----
    "Chunk": "knwler:Chunk",
    "text": {"@id": "schema:text"},
    "rephraseText": {"@id": "knwler:rephraseText"},
    "chunkIndex": {"@id": "schema:position", "@type": "xsd:integer"},
    # ---- Entity ----
    "Entity": "knwler:Entity",
    "name": {"@id": "schema:name"},
    "description": {"@id": "schema:description"},
    "entityType": {"@id": "knwler:entityType"},
    "extractedFromChunk": {
        "@id": "knwler:extractedFromChunk",
        "@type": "@id",
        "@container": "@set",
    },
    "belongsToCluster": {"@id": "knwler:belongsToCluster", "@type": "@id"},
    # ---- Relation (reified) ----
    "Relation": "knwler:Relation",
    "relationType": {"@id": "knwler:relationType"},
    "relationSource": {"@id": "knwler:source", "@type": "@id"},
    "relationTarget": {"@id": "knwler:target", "@type": "@id"},
    "strength": {"@id": "knwler:strength", "@type": "xsd:decimal"},
    "relationDescription": {"@id": "knwler:description"},
    # ---- Cluster ----
    "Cluster": "knwler:Cluster",
    "topics": {"@id": "skos:prefLabel", "@container": "@set"},
    "clusterDescription": {"@id": "schema:description"},
    "members": {"@id": "skos:member", "@type": "@id", "@container": "@set"},
    # ---- Document ↔ parts ----
    "hasChunk": {"@id": "knwler:hasChunk", "@type": "@id", "@container": "@set"},
    "hasCluster": {"@id": "knwler:hasCluster", "@type": "@id", "@container": "@set"},
    "mentionsEntity": {
        "@id": "knwler:mentionsEntity",
        "@type": "@id",
        "@container": "@set",
    },
    "hasRelation": {"@id": "knwler:hasRelation", "@type": "@id", "@container": "@set"},
    "hasSchema": {"@id": "knwler:hasSchema", "@type": "@id"},
}


# ---------------------------------------------------------------------------
# IRI minting
# ---------------------------------------------------------------------------


def _slug(s: str) -> str:
    """Percent-encode a string into a safe IRI path segment."""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "_", s).strip("_")
    s = re.sub(r"[^A-Za-z0-9_\-.]", lambda m: f"%{ord(m.group()):02X}", s)
    return s[:200]


def entity_iri(entity_id: str, name: str = "") -> str:
    seg = _slug(entity_id) if entity_id else _slug(name) or str(uuid4())
    return f"knwler:entity/{seg}"


def chunk_iri(chunk_id: str, doc_id: str = "", idx: int = 0) -> str:
    if chunk_id:
        return f"knwler:chunk/{_slug(chunk_id)}"
    return f"knwler:chunk/{_slug(doc_id)}__{idx}"


def cluster_iri(doc_id: str, cluster_id) -> str:
    return f"knwler:cluster/{_slug(doc_id)}__{_slug(str(cluster_id))}"


def relation_iri(rel_id: str) -> str:
    return f"knwler:relation/{_slug(rel_id) if rel_id else str(uuid4())}"


def document_iri(doc_id: str, url: str = "") -> str:
    seg = _slug(doc_id) if doc_id else _slug(url) or str(uuid4())
    return f"knwler:document/{seg}"


def schema_iri(doc_id: str) -> str:
    return f"knwler:schema/{_slug(doc_id)}"


# ---------------------------------------------------------------------------
# Relation type → knwler predicate (camelCase)
# ---------------------------------------------------------------------------


def relation_predicate(raw_type: str) -> str:
    """Convert a snake_case relation type to a knwler: predicate IRI."""
    if not raw_type:
        return "knwler:relatedTo"
    parts = re.split(r"[_\s]+", raw_type.strip().lower())
    camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
    camel = re.sub(r"[^A-Za-z0-9]", "", camel)
    return f"knwler:{camel}" if camel else "knwler:relatedTo"


def entity_class(raw_type: str) -> str:
    """Convert a snake_case entity type to a knwler: class IRI."""
    if not raw_type:
        return "knwler:Entity"
    parts = re.split(r"[_\s]+", raw_type.strip().lower())
    pascal = "".join(p.capitalize() for p in parts if p)
    pascal = re.sub(r"[^A-Za-z0-9]", "", pascal)
    return f"knwler:{pascal}" if pascal else "knwler:Entity"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_schema_node(doc_id: str, schema: dict) -> dict | None:
    if not schema:
        return None
    node = {
        "@id": schema_iri(doc_id),
        "@type": "KnowledgeSchema",
    }
    if schema.get("entity_types"):
        node["entityTypes"] = schema["entity_types"]
    if schema.get("relation_types"):
        node["relationTypes"] = schema["relation_types"]
    if schema.get("reasoning"):
        node["schemaReasoning"] = schema["reasoning"]
    return node


def build_document_node(doc_id: str, doc: dict, stats: dict, run_info: dict) -> dict:
    node: dict = {
        "@id": document_iri(doc_id, doc.get("url", "")),
        "@type": ["Document", "schema:CreativeWork"],
    }
    if doc.get("title"):
        node["title"] = doc["title"]
    if doc.get("summary"):
        node["summary"] = doc["summary"]
    if doc.get("url"):
        node["url"] = doc["url"]
    if doc.get("language"):
        node["language"] = doc["language"]
    if run_info.get("extraction_model"):
        node["extractionModel"] = run_info["extraction_model"]
    if run_info.get("discovery_model"):
        node["discoveryModel"] = run_info["discovery_model"]
    if run_info.get("timestamp"):
        node["timestamp"] = run_info["timestamp"]
    if stats.get("num_chunks") is not None:
        node["numChunks"] = stats["num_chunks"]
    return node


def build_chunk_nodes(doc_id: str, doc_iri_val: str, chunks: list) -> tuple[list, list]:
    """Return (chunk_nodes, chunk_iris)."""
    nodes = []
    iris = []
    for c in chunks:
        cid = c.get("id", "")
        idx = c.get("chunk_idx", 0)
        iri = chunk_iri(cid, doc_id, idx)
        iris.append(iri)
        node: dict = {
            "@id": iri,
            "@type": "Chunk",
            "chunkIndex": idx,
        }
        if c.get("text"):
            node["text"] = c["text"]
        if c.get("rephrase"):
            node["rephraseText"] = c["rephrase"]
        # Back-link to the document
        node["schema:isPartOf"] = {"@id": doc_iri_val}
        nodes.append(node)
    return nodes, iris


def build_entity_nodes(
    doc_id: str, entities: list, chunk_id_map: dict
) -> tuple[list, dict]:
    """
    Return (entity_nodes, entity_iri_map{name -> iri}).

    The name map stores both the original casing and the lower-cased key so
    that relation endpoints differing only in case still resolve correctly.
    chunk_id_map: maps chunk uuid → chunk IRI string
    """
    nodes = []
    name_to_iri: dict[str, str] = {}
    for e in entities:
        ent_id = e.get("id", "")
        iri = entity_iri(ent_id, e.get("name", ""))
        name_to_iri[e["name"]] = iri
        name_to_iri[e["name"].lower()] = iri

        # Build @type list: always Entity + specific class
        types = ["Entity", "skos:Concept"]
        if e.get("type"):
            types.append(entity_class(e["type"]))

        node: dict = {
            "@id": iri,
            "@type": types,
            "name": e["name"],
        }
        if e.get("description"):
            node["description"] = e["description"]
        if e.get("type"):
            node["entityType"] = e["type"]

        # Provenance: which chunks was this entity extracted from?
        chunk_refs = []
        for cid in e.get("chunk_ids", []):
            mapped = chunk_id_map.get(cid) or chunk_iri(cid, doc_id)
            chunk_refs.append(mapped)
        if chunk_refs:
            node["extractedFromChunk"] = chunk_refs

        # Cluster membership — resolved later after clusters are built
        if e.get("community_id") is not None:
            node["_community_id"] = e[
                "community_id"
            ]  # temporary, stripped before output

        nodes.append(node)
    return nodes, name_to_iri


def build_cluster_nodes(
    doc_id: str, clusters: list, entity_name_to_iri: dict
) -> tuple[list, dict]:
    """Return (cluster_nodes, cluster_id_map{int->iri})."""
    nodes = []
    id_to_iri: dict = {}
    for c in clusters:
        iri = cluster_iri(doc_id, c["id"])
        id_to_iri[c["id"]] = iri

        node: dict = {
            "@id": iri,
            "@type": ["Cluster", "skos:Collection"],
        }
        if c.get("topics"):
            node["topics"] = c["topics"]
        if c.get("description"):
            node["clusterDescription"] = c["description"]

        # Resolve member IRIs
        member_iris = []
        for m in c.get("members", []):
            entity_name = m.split("::")[0] if "::" in m else m
            if entity_name in entity_name_to_iri:
                member_iris.append(entity_name_to_iri[entity_name])
        if member_iris:
            node["members"] = member_iris

        nodes.append(node)
    return nodes, id_to_iri


def build_relation_nodes(
    relations: list,
    entity_name_to_iri: dict,
    chunk_id_map: dict,
    doc_id: str,
) -> tuple[list, dict, list]:
    """
    Return (relation_nodes, predicate_shorthand_map, stub_entity_nodes).

    Each relation gets a reified knwler:Relation resource.
    Additionally we collect {source_iri -> [(predicate, target_iri)]} for
    shorthand direct triples that are added to entity nodes.
    Entities referenced in relations but absent from the entity list become
    lightweight stub nodes so all IRI references remain resolvable.
    """
    nodes = []
    stubs: list = []
    # {source_iri -> [(predicate, target_iri)]}
    shorthand: dict[str, list] = {}

    def _resolve_entity(name: str) -> str:
        """Return IRI for entity name, creating a stub node if needed."""
        iri = entity_name_to_iri.get(name) or entity_name_to_iri.get(name.lower())
        if iri:
            return iri
        stub_iri = entity_iri("", name)
        stubs.append(
            {
                "@id": stub_iri,
                "@type": ["Entity", "skos:Concept"],
                "name": name,
                "knwler:stubEntity": True,
            }
        )
        entity_name_to_iri[name] = stub_iri
        entity_name_to_iri[name.lower()] = stub_iri
        return stub_iri

    for rel in relations:
        rel_id = rel.get("id", str(uuid4()))
        iri = relation_iri(rel_id)

        src_name = rel.get("source", "")
        tgt_name = rel.get("target", "")
        src_iri = _resolve_entity(src_name)
        tgt_iri = _resolve_entity(tgt_name)

        predicate = relation_predicate(rel.get("type", ""))

        node: dict = {
            "@id": iri,
            "@type": "Relation",
            "relationType": rel.get("type", ""),
            "relationSource": src_iri,
            "relationTarget": tgt_iri,
        }
        if predicate:
            node["knwler:predicate"] = predicate
        if rel.get("description"):
            node["relationDescription"] = rel["description"]
        if rel.get("strength") is not None:
            node["strength"] = rel["strength"]

        chunk_refs = []
        for cid in rel.get("chunk_ids", []):
            mapped = chunk_id_map.get(cid) or chunk_iri(cid, doc_id)
            chunk_refs.append(mapped)
        if chunk_refs:
            node["extractedFromChunk"] = chunk_refs

        nodes.append(node)

        # Accumulate shorthand
        shorthand.setdefault(src_iri, []).append((predicate, tgt_iri))

    return nodes, shorthand, stubs


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def convert_graph(doc: dict) -> dict:
    """Convert a single knwler JSON (single or consolidated) to a JSON-LD document."""

    # ---- Determine top-level doc identity ----
    is_consolidated = doc.get("documents") is not None
    if is_consolidated:
        doc_list = doc["documents"]
        primary = doc_list[0] if doc_list else {}
        doc_id = doc.get("id") or primary.get("id") or str(uuid4())
    else:
        primary = doc
        doc_id = doc.get("id") or str(uuid4())

    stats_raw = doc.get("stats", [{}])
    stats = (
        stats_raw[0] if isinstance(stats_raw, list) and stats_raw else (stats_raw or {})
    )
    run_info = stats.get("run", {})

    graph_section = doc.get("graph", {})
    entities_raw = graph_section.get("entities", [])
    relations_raw = graph_section.get("relations", [])
    clusters_raw = graph_section.get("communities", [])
    chunks_raw = doc.get("chunks", [])

    # ---- Schema node ----
    schema_node = build_schema_node(doc_id, doc.get("schema", {}))

    # ---- Document node(s) ----
    if is_consolidated:
        doc_nodes = [
            build_document_node(d.get("id") or str(uuid4()), d, stats, run_info)
            for d in doc_list
        ]
        # Use the primary doc IRI as anchor
        primary_doc_iri = doc_nodes[0]["@id"] if doc_nodes else document_iri(doc_id)
    else:
        doc_node = build_document_node(doc_id, primary, stats, run_info)
        doc_nodes = [doc_node]
        primary_doc_iri = doc_node["@id"]

    # Attach schema reference
    if schema_node:
        for dn in doc_nodes:
            dn["hasSchema"] = schema_node["@id"]

    # ---- Chunks ----
    chunk_nodes, chunk_iris = build_chunk_nodes(doc_id, primary_doc_iri, chunks_raw)
    # Build chunk uuid → IRI lookup for provenance links
    chunk_id_map: dict[str, str] = {}
    for raw, iri in zip(chunks_raw, chunk_iris):
        cid = raw.get("id", "")
        if cid:
            chunk_id_map[cid] = iri
        chunk_id_map[str(raw.get("chunk_idx", ""))] = iri

    # ---- Entities ----
    entity_nodes, name_to_iri = build_entity_nodes(doc_id, entities_raw, chunk_id_map)

    # ---- Clusters ----
    cluster_nodes, cluster_id_to_iri = build_cluster_nodes(
        doc_id, clusters_raw, name_to_iri
    )

    # ---- Back-fill cluster membership on entity nodes ----
    for enode in entity_nodes:
        comm_id = enode.pop("_community_id", None)
        if comm_id is not None and comm_id in cluster_id_to_iri:
            enode["belongsToCluster"] = cluster_id_to_iri[comm_id]

    # ---- Relations ----
    relation_nodes, shorthand, stub_nodes = build_relation_nodes(
        relations_raw, name_to_iri, chunk_id_map, doc_id
    )
    # Stub entities are added to entity_nodes so they appear in @graph and
    # are included in the document's mentionsEntity list
    entity_nodes.extend(stub_nodes)

    # ---- Add shorthand direct triples to entity nodes ----
    iri_to_entity_node = {en["@id"]: en for en in entity_nodes}
    for src_iri, pairs in shorthand.items():
        if src_iri not in iri_to_entity_node:
            continue
        enode = iri_to_entity_node[src_iri]
        for predicate, tgt_iri in pairs:
            existing = enode.get(predicate, [])
            if not isinstance(existing, list):
                existing = [existing]
            ref = {"@id": tgt_iri}
            if ref not in existing:
                existing.append(ref)
            enode[predicate] = existing

    # ---- Wire document ↔ parts ----
    for dn in doc_nodes:
        if chunk_iris:
            dn["hasChunk"] = chunk_iris
        if cluster_nodes:
            dn["hasCluster"] = [cn["@id"] for cn in cluster_nodes]
        if entity_nodes:
            dn["mentionsEntity"] = [en["@id"] for en in entity_nodes]
        if relation_nodes:
            dn["hasRelation"] = [rn["@id"] for rn in relation_nodes]

    # ---- Assemble @graph ----
    graph: list = []
    graph.extend(doc_nodes)
    if schema_node:
        graph.append(schema_node)
    graph.extend(chunk_nodes)
    graph.extend(entity_nodes)
    graph.extend(cluster_nodes)
    graph.extend(relation_nodes)

    return {
        "@context": JSONLD_CONTEXT,
        "@graph": graph,
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def export_file(path: Path, output_dir: Path | None) -> dict:
    """Export a single graph JSON to JSON-LD. Returns a stats dict."""
    with open(path) as f:
        doc = json.load(f)

    console.print(f"\n[bold cyan]Exporting[/bold cyan] [dim]{path}[/dim]")

    graph_section = doc.get("graph", {})
    entities = graph_section.get("entities", [])
    relations = graph_section.get("relations", [])
    chunks = doc.get("chunks", [])
    clusters = graph_section.get("communities", [])

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
        t = progress.add_task(
            "[green]Converting nodes",
            total=len(chunks) + len(entities) + len(relations) + len(clusters),
        )
        result = convert_graph(doc)
        progress.update(
            t, advance=len(chunks) + len(entities) + len(relations) + len(clusters)
        )

    # Determine output path
    dest_dir = output_dir if output_dir else path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / (path.stem + ".jsonld")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    console.print(f"  [green]✓[/green] Written → [bold]{out_path}[/bold]")
    console.print(
        f"    [dim]{len(entities)} entities · "
        f"{len(relations)} relations · "
        f"{len(chunks)} chunks · "
        f"{len(clusters)} clusters · "
        f"{len(result['@graph'])} total nodes[/dim]"
    )

    return {
        "file": path.name,
        "output": str(out_path),
        "chunks": len(chunks),
        "clusters": len(clusters),
        "entities": len(entities),
        "relations": len(relations),
        "total_nodes": len(result["@graph"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    files: Annotated[
        list[Path],
        typer.Argument(help="Path(s) to knwler graph JSON file(s) to export"),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output directory (default: same as input file)"
        ),
    ] = None,
    pretty: Annotated[
        bool,
        typer.Option(
            "--pretty/--compact", help="Pretty-print JSON-LD output (default: pretty)"
        ),
    ] = True,
) -> None:
    """Export knowledge-graph JSON file(s) to JSON-LD (RDF)."""

    for f in files:
        if not f.exists():
            console.print(f"[bold red]Error:[/bold red] File not found: {f}")
            raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold]Format[/bold]      JSON-LD (RDF)\n"
            f"[bold]Ontologies[/bold]  knwler · schema.org · dcterms · skos\n"
            f"[bold]Output[/bold]      {output or '(same directory as input)'}\n"
            f"[bold]Files[/bold]       {len(files)}",
            title="[bold green]knwler → JSON-LD Export[/bold green]",
            border_style="green",
        )
    )

    results = []
    for f in files:
        stats = export_file(f, output)
        results.append(stats)

    table = Table(title="Export Summary", show_header=True, header_style="bold cyan")
    table.add_column("File", style="dim")
    table.add_column("Entities", justify="right")
    table.add_column("Relations", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Clusters", justify="right")
    table.add_column("Total nodes", justify="right")
    table.add_column("Output", style="green")
    for r in results:
        table.add_row(
            r["file"],
            str(r["entities"]),
            str(r["relations"]),
            str(r["chunks"]),
            str(r["clusters"]),
            str(r["total_nodes"]),
            r["output"],
        )
    console.print(table)


if __name__ == "__main__":
    app()
