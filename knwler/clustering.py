"""
Community detection and labeling.
"""

import networkx as nx
from networkx.algorithms.community import louvain_communities

from knwler.config import Config, console, null_console
from knwler.language import get_console_msg, get_prompt
from knwler.llm import llm_generate, parse_json_response
from knwler.models import Graph, ClusteredGraph
from dataclasses import asdict


# ---------------------------------------------------------------------------
# Network creation
# ---------------------------------------------------------------------------
def create_network(
    consolidated: dict | Graph,
    title: str = None,
    url: str = None,
    language: str = None,
    summary: str = None,
) -> nx.MultiDiGraph:
    """Create a NetworkX graph from consolidated data."""
    g = nx.MultiDiGraph()
    from knwler.cache import hash_args

    if isinstance(consolidated, Graph):
        consolidated = asdict(consolidated)  # type: ignore
    # add a document node so we have a reference from the nodes to the source document
    doc_hash = hash_args(consolidated.get("title", ""))
    g.add_node(
        doc_hash,
        type="document",
        title=title,
        summary=summary,
        url=consolidated.get("url", ""),
        language=language,
    )
    for e in consolidated["entities"]:
        node_id = f"{e['name']}::{e['type']}" if e.get("type") else e["name"]
        g.add_node(
            node_id,
            name=e["name"],
            type=e["type"],
            description=e["description"],
            community_id=e.get("community_id"),
            document=doc_hash,
        )
    for r in consolidated["relations"]:
        src_id = (
            f"{r['source']}::{r['source_type']}"
            if r.get("source_type")
            else r["source"]
        )
        tgt_id = (
            f"{r['target']}::{r['target_type']}"
            if r.get("target_type")
            else r["target"]
        )
        g.add_edge(src_id, tgt_id, type=r["type"], description=r["description"])
    return g


# ---------------------------------------------------------------------------
# Community analysis
# ---------------------------------------------------------------------------
async def cluster_graph(
    graph: dict | Graph, config: Config = Config(), _console=None
) -> ClusteredGraph:
    """Detect communities and label them with topics and descriptions."""
    if _console is None:
        _console = console
    if not graph:
        raise ValueError("No graph provided for clustering analysis.")
    analyzing_msg = (
        get_console_msg("analyzing_communities") or "Analyzing communities..."
    )
    if isinstance(graph, Graph):
        graph = asdict(graph)
    with _console.status(f"[cyan]{analyzing_msg}"):
        g = nx.Graph()
        for e in graph.get("entities", []):
            node_id = f"{e['name']}::{e['type']}" if e.get("type") else e["name"]
            g.add_node(node_id)
        for r in graph.get("relations", []):
            weight = r.get("strength", 1)
            src_id = (
                f"{r['source']}::{r['source_type']}"
                if r.get("source_type")
                else r["source"]
            )
            tgt_id = (
                f"{r['target']}::{r['target_type']}"
                if r.get("target_type")
                else r["target"]
            )
            g.add_edge(src_id, tgt_id, weight=weight)

        if g.number_of_nodes() == 0:
            graph["communities"] = []
            return ClusteredGraph(**graph)

        clusters = louvain_communities(g, weight="weight", seed=0)
        # Map composite node IDs (name::type) to entity dicts
        entity_map = {}
        for e in graph.get("entities", []):
            node_id = f"{e['name']}::{e['type']}" if e.get("type") else e["name"]
            entity_map[node_id] = e

        community_payload: list[dict] = []
        for cid, members in enumerate(clusters):
            member_objs = [
                {
                    "name": entity_map.get(node_id, {}).get("name", node_id),
                    "type": entity_map.get(node_id, {}).get("type", ""),
                    "description": entity_map.get(node_id, {}).get("description", ""),
                }
                for node_id in sorted(members)
            ]
            community_payload.append({"id": str(cid), "members": member_objs})

        labels: dict = {}
        if community_payload:
            prompt = _build_community_prompt(community_payload)
            response = await llm_generate(prompt, config, model=config.extraction_model)
            parsed = parse_json_response(response)
            labels = parsed.get("communities", {})

        if not labels:
            labels = _fallback_community_labels(community_payload)

        communities_out: list[dict] = []
        for cid, members in enumerate(clusters):
            label = labels.get(str(cid), {"topics": ["misc"], "description": ""})
            communities_out.append(
                {
                    "id": cid,
                    "topics": label.get("topics", ["misc"]),
                    "description": label.get("description", ""),
                    "members": sorted(members),
                }
            )
            for node_id in members:
                if node_id in entity_map:
                    entity_map[node_id]["community_id"] = cid

        graph["communities"] = communities_out
        detected_msg = (
            get_console_msg("detected_communities", count=len(clusters))
            or f"Detected {len(clusters)} communities"
        )
        _console.print(f"[white]{detected_msg}[/]")
    return ClusteredGraph(**graph)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_community_prompt(communities: list[dict]) -> str:
    """Build prompt for community labeling."""
    lines = []
    for c in communities:
        members = ", ".join(
            f'{{"name": "{m["name"]}", "type": "{m["type"]}", "description": "{m["description"]}"}}'
            for m in c["members"]
        )
        lines.append(f'- "{c["id"]}": [{members}]')
    communities_text = chr(10).join(lines)

    prompt = get_prompt("community_labeling", communities_text=communities_text)

    if not prompt:
        prompt = (
            "You are labeling graph communities. For each community, return "
            "1-3 short topics and a 1-2 sentence description.\n\n"
            f"COMMUNITIES:\n{communities_text}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "communities": {\n'
            '    "<id>": {"topics": ["topic1", "topic2"], "description": "..."},\n'
            "    ...\n"
            "  }\n"
            "}"
        )
    return prompt


def _fallback_community_labels(communities: list[dict]) -> dict:
    """Fallback labels using most common entity types."""
    labels: dict = {}
    for c in communities:
        type_counts: dict[str, int] = {}
        for m in c["members"]:
            etype = (m.get("type") or "").strip().lower()
            if not etype:
                continue
            type_counts[etype] = type_counts.get(etype, 0) + 1
        topics = [
            t.replace("_", " ").capitalize()
            for t, _ in sorted(type_counts.items(), key=lambda x: -x[1])
        ][:3]
        if not topics:
            topics = ["misc"]

        labels[c["id"]] = {
            "topics": topics,
            "description": f"Around {', '.join(topics)}.",
        }
    return labels
