"""
Community detection and labeling.
"""

import networkx as nx
from networkx.algorithms.community import louvain_communities

from knwler.config import Config, console
from knwler.language import get_console_msg, get_prompt
from knwler.llm import llm_generate, parse_json_response


# ---------------------------------------------------------------------------
# Network creation
# ---------------------------------------------------------------------------
def create_network(consolidated: dict) -> nx.MultiDiGraph:
    """Create a NetworkX graph from consolidated data."""
    g = nx.MultiDiGraph()
    for e in consolidated["entities"]:
        g.add_node(
            e["name"],
            type=e["type"],
            description=e["description"],
            community_id=e.get("community_id"),
        )
    for r in consolidated["relations"]:
        g.add_edge(
            r["source"], r["target"], type=r["type"], description=r["description"]
        )
    return g


# ---------------------------------------------------------------------------
# Community analysis
# ---------------------------------------------------------------------------
def analyze_communities(consolidated: dict, config: Config) -> dict:
    """Detect communities and label them with topics and descriptions."""
    analyzing_msg = (
        get_console_msg("analyzing_communities") or "Analyzing communities..."
    )
    with console.status(f"[cyan]{analyzing_msg}"):
        g = nx.Graph()
        for e in consolidated.get("entities", []):
            g.add_node(e["name"])
        for r in consolidated.get("relations", []):
            weight = r.get("strength", 1)
            g.add_edge(r["source"], r["target"], weight=weight)

        if g.number_of_nodes() == 0:
            consolidated["communities"] = []
            return consolidated

        communities = louvain_communities(g, weight="weight", seed=0)
        entity_map = {e["name"]: e for e in consolidated.get("entities", [])}

        community_payload: list[dict] = []
        for cid, members in enumerate(communities):
            member_objs = [
                {
                    "name": name,
                    "type": entity_map.get(name, {}).get("type", ""),
                    "description": entity_map.get(name, {}).get("description", ""),
                }
                for name in sorted(members)
            ]
            community_payload.append({"id": str(cid), "members": member_objs})

        labels: dict = {}
        if community_payload:
            prompt = _build_community_prompt(community_payload)
            response = llm_generate(prompt, config, model=config.extraction_model)
            parsed = parse_json_response(response)
            labels = parsed.get("communities", {})

        if not labels:
            labels = _fallback_community_labels(community_payload)

        communities_out: list[dict] = []
        for cid, members in enumerate(communities):
            label = labels.get(str(cid), {"topics": ["misc"], "description": ""})
            communities_out.append(
                {
                    "id": cid,
                    "topics": label.get("topics", ["misc"]),
                    "description": label.get("description", ""),
                    "members": sorted(members),
                }
            )
            for name in members:
                if name in entity_map:
                    entity_map[name]["community_id"] = cid

        consolidated["communities"] = communities_out
        detected_msg = (
            get_console_msg("detected_communities", count=len(communities))
            or f"Detected {len(communities)} communities"
        )
        console.print(f"[white]{detected_msg}[/]")
    return consolidated


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
