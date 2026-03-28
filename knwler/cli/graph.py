"""CLI graph commands: convert and analyze."""

import json
import webbrowser
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import networkx as nx
import typer
from jinja2 import Environment, FileSystemLoader, select_autoescape

from knwler.config import console

graph_app = typer.Typer(
    help="Graph conversion and analysis tools.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

SUPPORTED_FORMATS = ["gml", "graphml", "jsonld", "cytoscape", "edgelist"]
_FORMAT_EXT = {
    "gml": ".gml",
    "graphml": ".graphml",
    "jsonld": ".jsonld",
    "cytoscape": ".cytoscape.json",
    "edgelist": ".edgelist.txt",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_graph_data(path: Path) -> dict:
    """Load graph.json or consolidated_graph.json into a normalised dict."""
    data = json.loads(path.read_text())
    # Support both wrapped {"graph": {...}} and flat {"entities": [...], ...}
    if "graph" in data and isinstance(data["graph"], dict):
        graph = data["graph"]
    else:
        graph = data
    docs = data.get("documents", [])
    title = data.get("title") or (docs[0].get("title") if docs else None) or path.stem
    return {
        "entities": graph.get("entities", []),
        "relations": graph.get("relations", []),
        "clusters": graph.get("clusters", []),
        "chunks": data.get("chunks", []),
        "title": title,
        "schema": data.get("schema", {}),
        "raw": data,
    }


def _build_nx_graph(entities: list, relations: list) -> nx.DiGraph:
    """Build a NetworkX directed graph from entity/relation lists."""
    G = nx.DiGraph()
    for e in entities:
        name = e.get("name", "")
        if not name:
            continue
        attrs = {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in e.items()
            if k not in ("name", "chunk_ids")
        }
        attrs["chunk_ids"] = ",".join(str(c) for c in e.get("chunk_ids", []))
        G.add_node(name, **attrs)
    for r in relations:
        src = r.get("source", "")
        tgt = r.get("target", "")
        if not src or not tgt:
            continue
        G.add_edge(
            src,
            tgt,
            type=r.get("type", ""),
            description=r.get("description", ""),
            strength=float(r.get("strength", 5)),
        )
    return G


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------


def _to_gml(G: nx.DiGraph, output_path: Path) -> None:
    nx.write_gml(G, str(output_path))


def _to_graphml(G: nx.DiGraph, output_path: Path) -> None:
    nx.write_graphml(G, str(output_path))


def _to_cytoscape(entities: list, relations: list, output_path: Path) -> None:
    nodes = [
        {
            "data": {
                "id": e["name"],
                "label": e["name"],
                "type": e.get("type", ""),
                "description": e.get("description", ""),
            }
        }
        for e in entities
        if e.get("name")
    ]
    edges = [
        {
            "data": {
                "id": f"e{i}",
                "source": r["source"],
                "target": r["target"],
                "type": r.get("type", ""),
                "description": r.get("description", ""),
                "strength": r.get("strength", 5),
            }
        }
        for i, r in enumerate(relations)
        if r.get("source") and r.get("target")
    ]
    output_path.write_text(
        json.dumps({"elements": {"nodes": nodes, "edges": edges}}, indent=2),
        encoding="utf-8",
    )


def _to_edgelist(relations: list, output_path: Path) -> None:
    lines = [
        f"{r['source']}\t{r['target']}\t{r.get('type', '')}"
        for r in relations
        if r.get("source") and r.get("target")
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _to_jsonld(entities: list, relations: list, title: str, output_path: Path) -> None:
    def slug(s: str) -> str:
        return s.lower().replace(" ", "-").replace("/", "-")

    context = {
        "knwler": "https://knwler.com/ontology#",
        "schema": "https://schema.org/",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "name": "rdfs:label",
        "description": "rdfs:comment",
    }
    entity_iris: dict[str, str] = {}
    graph_nodes = []

    for e in entities:
        name = e.get("name", "")
        if not name:
            continue
        iri = f"knwler:entity/{slug(e.get('type', 'entity'))}/{slug(name)}"
        entity_iris[name] = iri
        graph_nodes.append(
            {
                "@id": iri,
                "@type": [
                    "knwler:Entity",
                    f"knwler:{e.get('type','Entity').replace(' ','').title()}",
                ],
                "rdfs:label": name,
                "rdfs:comment": e.get("description", ""),
                "knwler:entityType": e.get("type", ""),
            }
        )

    for i, r in enumerate(relations):
        src, tgt = r.get("source", ""), r.get("target", "")
        if not src or not tgt:
            continue
        rel_type = r.get("type", "relatedTo")
        graph_nodes.append(
            {
                "@id": f"knwler:relation/{i}/{slug(rel_type)}",
                "@type": "knwler:Relation",
                "knwler:source": {
                    "@id": entity_iris.get(src, f"knwler:entity/unknown/{slug(src)}")
                },
                "knwler:target": {
                    "@id": entity_iris.get(tgt, f"knwler:entity/unknown/{slug(tgt)}")
                },
                "knwler:relationType": rel_type,
                "rdfs:comment": r.get("description", ""),
                "knwler:strength": r.get("strength", 5),
            }
        )

    output_path.write_text(
        json.dumps({"@context": context, "@graph": graph_nodes}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# convert command
# ---------------------------------------------------------------------------


@graph_app.command("convert")
def convert_graph(
    input: Annotated[
        Path, typer.Argument(help="Path to graph.json or consolidated_graph.json")
    ],
    format: Annotated[
        str,
        typer.Option(
            "--format", "-F", help=f"Output format: {', '.join(SUPPORTED_FORMATS)}"
        ),
    ] = "graphml",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-O", help="Output file path (default: same directory as input)"
        ),
    ] = None,
):
    """Convert a graph.json to GML, GraphML, JSON-LD, Cytoscape JSON, or edge list."""
    if not input.exists():
        console.print(f"[red]✗[/red] File not found: [cyan]{input}[/cyan]")
        raise typer.Exit(1)

    fmt = format.lower().strip()
    if fmt not in SUPPORTED_FORMATS:
        console.print(
            f"[red]✗[/red] Unsupported format: [bold]{fmt}[/bold]. "
            f"Choose from: {', '.join(SUPPORTED_FORMATS)}"
        )
        raise typer.Exit(1)

    console.rule(f"[bold magenta]Converting graph → {fmt.upper()}[/bold magenta]")
    data = _load_graph_data(input)
    entities = data["entities"]
    relations = data["relations"]

    if output is None:
        stem = input.stem if not input.stem.endswith(".json") else input.stem[:-5]
        output = input.parent / (stem + _FORMAT_EXT[fmt])

    if fmt == "gml":
        G = _build_nx_graph(entities, relations)
        _to_gml(G, output)
    elif fmt == "graphml":
        G = _build_nx_graph(entities, relations)
        _to_graphml(G, output)
    elif fmt == "cytoscape":
        _to_cytoscape(entities, relations, output)
    elif fmt == "edgelist":
        _to_edgelist(relations, output)
    elif fmt == "jsonld":
        _to_jsonld(entities, relations, data["title"], output)

    console.print(
        f"[green]✓[/green] Saved [cyan]{len(entities)}[/cyan] entities and "
        f"[cyan]{len(relations)}[/cyan] relations → [cyan]{output}[/cyan]"
    )


# ---------------------------------------------------------------------------
# Analysis computation
# ---------------------------------------------------------------------------


def _compute_analysis(data: dict, top_n: int) -> dict:
    entities = data["entities"]
    relations = data["relations"]
    clusters = data["clusters"]
    chunks = data["chunks"]

    G = _build_nx_graph(entities, relations)
    UG = G.to_undirected()

    num_entities = len(entities)
    num_relations = len(relations)
    density = round(nx.density(G), 6) if num_entities > 1 else 0.0
    avg_degree = round(2 * num_relations / num_entities, 2) if num_entities > 0 else 0.0

    if num_entities > 0:
        comps = list(nx.connected_components(UG))
        n_components = len(comps)
        largest_cc_size = max(len(c) for c in comps)
    else:
        n_components = 0
        largest_cc_size = 0
    largest_cc_pct = (
        round(100 * largest_cc_size / num_entities, 1) if num_entities > 0 else 0
    )

    # Degree
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    total_deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in G.nodes()}
    top_by_degree = sorted(total_deg.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # PageRank — fall back to degree-normalised score if scipy is unavailable
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
        except Exception:
            total_deg_sum = sum(total_deg.values()) or 1
            pagerank = {nd: d / total_deg_sum for nd, d in total_deg.items()}
    else:
        _n = G.number_of_nodes()
        pagerank = {nd: 1.0 / _n for nd in G.nodes()} if _n else {}
    top_by_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Betweenness centrality
    try:
        betweenness = (
            nx.betweenness_centrality(G, normalized=True)
            if G.number_of_nodes() > 1
            else {}
        )
    except Exception:
        betweenness = {}
    top_by_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]

    # Most referenced chunks
    chunk_ref: Counter = Counter()
    for e in entities:
        for cid in e.get("chunk_ids", []):
            chunk_ref[cid] += 1
    for r in relations:
        for cid in r.get("chunk_ids", []):
            chunk_ref[cid] += 1

    chunk_by_id = {c["id"]: c for c in chunks if "id" in c}
    top_chunks = []
    for cid, count in chunk_ref.most_common(top_n):
        c = chunk_by_id.get(cid, {})
        text = c.get("rephrase") or c.get("text", "")
        preview = text[:250] + "…" if len(text) > 250 else text
        top_chunks.append(
            {
                "id": cid,
                "chunk_idx": c.get("chunk_idx"),
                "references": count,
                "preview": preview,
            }
        )

    # Type distributions
    entity_type_counts = dict(
        Counter(e.get("type", "unknown") for e in entities).most_common(20)
    )
    relation_type_counts = dict(
        Counter(r.get("type", "unknown") for r in relations).most_common(20)
    )

    # Cluster summaries
    cluster_stats = [
        {
            "id": c.get("id"),
            "topics": c.get("topics", []),
            "description": c.get("description", ""),
            "size": len(c.get("members", [])),
            "members": sorted(c.get("members", []))[:12],
        }
        for c in sorted(
            clusters, key=lambda x: len(x.get("members", [])), reverse=True
        )
    ]

    entity_info = {e["name"]: e for e in entities if e.get("name")}

    def _enrich(name: str, score: float) -> dict:
        e = entity_info.get(name, {})
        desc = e.get("description") or ""
        return {
            "name": name,
            "type": e.get("type", ""),
            "description": desc[:160] + "…" if len(desc) > 160 else desc,
            "score": round(score, 5),
            "degree": total_deg.get(name, 0),
        }

    return {
        "title": data["title"],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "stats": {
            "num_entities": num_entities,
            "num_relations": num_relations,
            "density": density,
            "avg_degree": avg_degree,
            "num_clusters": len(clusters),
            "num_chunks": len(chunks),
            "n_components": n_components,
            "largest_cc_size": largest_cc_size,
            "largest_cc_pct": largest_cc_pct,
        },
        "top_by_degree": [_enrich(n, s) for n, s in top_by_degree],
        "top_by_pagerank": [_enrich(n, s) for n, s in top_by_pagerank],
        "top_by_betweenness": [_enrich(n, s) for n, s in top_by_betweenness],
        "top_chunks": top_chunks,
        "entity_type_counts": entity_type_counts,
        "relation_type_counts": relation_type_counts,
        "clusters": cluster_stats,
    }


def _render_analysis_html(analysis: dict) -> str:
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
    )
    tpl = env.get_template("graph_analysis.html")
    et = analysis["entity_type_counts"]
    rt = analysis["relation_type_counts"]
    return tpl.render(
        **analysis,
        entity_type_labels=list(et.keys()),
        entity_type_data=list(et.values()),
        relation_type_labels=list(rt.keys()),
        relation_type_data=list(rt.values()),
    )


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------


@graph_app.command("analyze")
def analyze_graph(
    input: Annotated[
        Path, typer.Argument(help="Path to graph.json or consolidated_graph.json")
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-O", help="Output directory (default: same directory as input)"
        ),
    ] = None,
    top: Annotated[
        int,
        typer.Option("--top", "-N", help="Number of top entities/chunks to include"),
    ] = 10,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open the HTML report in the browser"),
    ] = True,
):
    """Analyse a graph.json and save results as JSON + an interactive HTML report."""
    if not input.exists():
        console.print(f"[red]✗[/red] File not found: [cyan]{input}[/cyan]")
        raise typer.Exit(1)

    console.rule("[bold magenta]Graph Analysis[/bold magenta]")

    out_dir = output if output else input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input.stem.replace("_graph", "").replace(".json", "")
    json_out = out_dir / f"{stem}_analysis.json"
    html_out = out_dir / f"{stem}_analysis.html"

    with console.status("[cyan]Loading graph data…"):
        data = _load_graph_data(input)

    console.print(
        f"[green]✓[/green] Loaded [cyan]{len(data['entities'])}[/cyan] entities, "
        f"[cyan]{len(data['relations'])}[/cyan] relations"
    )

    with console.status("[cyan]Computing graph metrics…"):
        analysis = _compute_analysis(data, top_n=top)

    # Save JSON
    json_out.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    console.print(f"[green]✓[/green] Analysis JSON → [cyan]{json_out}[/cyan]")

    # Render and save HTML
    with console.status("[cyan]Rendering HTML report…"):
        html = _render_analysis_html(analysis)
    html_out.write_text(html, encoding="utf-8")
    console.print(f"[green]✓[/green] Analysis report → [cyan]{html_out}[/cyan]")

    stats = analysis["stats"]
    from rich.table import Table

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("Entities", str(stats["num_entities"]))
    t.add_row("Relations", str(stats["num_relations"]))
    t.add_row("Clusters", str(stats["num_clusters"]))
    t.add_row("Avg degree", str(stats["avg_degree"]))
    t.add_row("Density", str(stats["density"]))
    t.add_row("Components", str(stats["n_components"]))
    t.add_row("Largest component", f"{stats['largest_cc_pct']}% of nodes")
    console.print(t)

    if analysis["top_by_pagerank"]:
        top_entity = analysis["top_by_pagerank"][0]
        console.print(
            f"\n[bold]Most important entity:[/bold] [cyan]{top_entity['name']}[/cyan] "
            f"([dim]{top_entity['type']}[/dim], PageRank={top_entity['score']})"
        )
    if analysis["top_chunks"]:
        tc = analysis["top_chunks"][0]
        label = (
            f"Part {tc['chunk_idx'] + 1}"
            if tc["chunk_idx"] is not None
            else tc["id"][:8]
        )
        console.print(
            f"[bold]Most referenced chunk:[/bold] [cyan]{label}[/cyan] "
            f"([dim]{tc['references']} references[/dim])"
        )

    if open_browser:
        webbrowser.open(html_out.as_uri())
