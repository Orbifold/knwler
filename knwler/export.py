"""
HTML report export.
"""

import html as html_mod
import json
import re
from datetime import datetime
from pathlib import Path

import markdown as md_lib

from jinja2 import Environment, FileSystemLoader, select_autoescape

from knwler.config import console
from knwler.language import get_current_language, get_ui
from typing import Tuple, List, Dict, Any
from knwler.models import KnowledgeGraph
from dataclasses import asdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _md_to_html(text: str) -> str:
    """Convert a markdown string to an HTML fragment."""
    return md_lib.markdown(text, extensions=["nl2br", "sane_lists"])


def _linkify_entities(
    text: str, entities: list[dict], already_html: bool = False
) -> str:
    """Replace known entity names in text with hyperlinks to ``#entity-<name>::<type>``.

    When *already_html* is ``True`` the input is treated as an HTML fragment
    (e.g. produced by :func:`_md_to_html`) and initial HTML-escaping is skipped.
    The regex is anchored so it won't match inside HTML tag attributes.
    """
    esc = text if already_html else html_mod.escape(text)
    if not entities:
        return esc

    esc_to_anchor: dict[str, str] = {}
    esc_names: list[str] = []
    for entity in entities:
        name = entity.get("name", "")
        type = entity.get("type", "")
        esc_name = html_mod.escape(name)
        key = esc_name.lower()
        if key in esc_to_anchor:
            continue
        esc_to_anchor[key] = (
            f"entity-{name.lower().replace(' ', '-')}::{type.lower().replace(' ', '-')}"
        )
        esc_names.append(esc_name)

    esc_names.sort(key=len, reverse=True)
    # The negative lookahead ``(?![^<]*>)`` prevents matching inside HTML tags.
    pattern = r"(?i)(?![^<]*>)\b(" + "|".join(re.escape(n) for n in esc_names) + r")\b"

    def repl(match: re.Match) -> str:
        matched = match.group(1)
        anchor = esc_to_anchor.get(matched.lower())
        if not anchor:
            return matched
        return f'<a href="#{anchor}" class="entity-link">{matched}</a>'

    return re.sub(pattern, repl, esc)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_html(
    results_data: KnowledgeGraph | dict,
    template: str = "default",
) -> str:
    """Export results.json data to an HTML report using Jinja2 template."""
    if isinstance(results_data, KnowledgeGraph):
        results_data = asdict(results_data)
    title = results_data.get("title") or "Knowledge Graph"
    summary = results_data.get("summary", "")
    url = results_data.get("url", "")
    chunks = results_data.get("chunks", [])
    graph = results_data.get("graph", {})
    entities = graph.get("entities", [])
    relations = graph.get("relations", [])
    communities = graph.get("communities", [])

    entity_names = {e["name"] for e in entities}
    chunk_mapping = {
        c["id"]: c["chunk_idx"] for c in chunks if "id" in c and "chunk_idx" in c
    }
    # Build relation lookup: entity -> list of (other, type, description, direction)
    rel_map: dict[Tuple[str, str], list[dict]] = {}
    for r in relations:
        src = (r.get("source", ""), r.get("source_type", ""))
        tgt = (r.get("target", ""), r.get("target_type", ""))
        rtype = r.get("type", "")
        desc = r.get("description", "")
        rel_map.setdefault(src, []).append(
            {"other": tgt, "type": rtype, "description": desc, "dir": "out"}
        )
        rel_map.setdefault(tgt, []).append(
            {"other": src, "type": rtype, "description": desc, "dir": "in"}
        )

    node_elements = [
        {
            "id": e.get("name", ""),
            "label": e.get("name", ""),
            "type": e.get("type", ""),
            "description": e.get("description", ""),
            "chunk_ids": e.get("chunk_ids", []),
            "community_id": e.get("community_id"),
        }
        for e in entities
    ]
    edge_elements = [
        {
            "id": f"e{idx}",
            "source": r.get("source", ""),
            "source_type": r.get("source_type", ""),
            "target": r.get("target", ""),
            "target_type": r.get("target_type", ""),
            "type": r.get("type", ""),
            "description": r.get("description", ""),
        }
        for idx, r in enumerate(relations)
        if r.get("source", "") in entity_names and r.get("target", "") in entity_names
    ]

    # Prepare communities display data
    clusters = []
    for c in communities:
        topics = c.get("topics", [])
        members = c.get("members", [])
        member_links = ", ".join(
            f'<a href="#entity-{m.lower().replace(" ", "-")}" class="entity-link">'
            f"{html_mod.escape(m)}</a>"
            for m in sorted(members)
        )
        clusters.append(
            {
                "topics": topics,
                "description": c.get("description", ""),
                "member_links": member_links,
            }
        )

    # Prepare chunks display data
    chunks_display = []
    for i, chunk in enumerate(chunks):
        rephrase = chunk.get("rephrase", "")
        original = chunk.get("text", "")
        display = rephrase if rephrase else original
        display_html = _md_to_html(display)
        original_html = _md_to_html(original)
        chunks_display.append(
            {
                "id": f"chunk-{chunk.get('id')}",
                "index": i + 1,
                "text": original,
                "linkified": _linkify_entities(
                    display_html, chunk.get("entities", []), already_html=True
                ),
                "original": original_html,
            }
        )

    # Prepare entities display data
    chunk_label = get_ui("chunk_label") or "Part"
    entities_display = []
    for e in sorted(entities, key=lambda x: x.get("name", "").lower()):
        name = e.get("name", "")
        entity_type = e.get("type", "")
        chunk_ids = e.get("chunk_ids", [])
        rels = rel_map.get((name, entity_type), [])

        rel_html = []
        for rel in rels:
            other = rel["other"]
            other_anchor = f"entity-{'::'.join(other).lower().replace(' ', '-')}"
            arrow = "\u2192" if rel["dir"] == "out" else "\u2190"
            label = (
                f'{arrow} <a href="#{other_anchor}" class="entity-link">'
                f"{html_mod.escape(other[0])}</a>"
            )
            description = rel["description"]
            if isinstance(description, list):
                description = ", ".join(str(d) for d in description)
            if description:
                label += f": {html_mod.escape(description)}"
            rel_html.append(label)

        chunk_links = " \u2022 ".join(
            f'<a href="#chunk-{cid}-rephrase">{chunk_label} {chunk_mapping.get(cid)+1}</a>'  # index is 0-based, so add 1 for display
            for cid in chunk_ids
            if cid is not None
        )

        entities_display.append(
            {
                "name": name,
                "type": e.get("type", ""),
                "anchor": f"entity-{name.lower().replace(' ', '-')}::{entity_type.lower().replace(' ', '-')}",
                "description": e.get("description", ""),
                "relations": rel_html,
                "chunk_links": chunk_links,
            }
        )

    # Get localized labels
    date_info = datetime.now().strftime("%b %d, %Y")
    metadata = get_ui(
        "metadata",
        entities=len(entities),
        relations=len(relations),
        communities=len(communities),
        date=date_info,
    ) or (
        f"Extracted {len(entities)} entities, {len(relations)} relations "
        f"and {len(communities)} topics on {date_info}"
    )

    labels = {
        "source": get_ui("source_label") or "Source",
        "nav_chunks": get_ui("nav_chunks") or "Text Chunks",
        "nav_topics": get_ui("nav_topics") or "Topics",
        "nav_entities": get_ui("nav_entities") or "Entities",
        "network_title": get_ui("network_title") or "Network",
        "topics_title": get_ui("topics_title") or "Topics",
        "content_title": get_ui("content_title") or "Content",
        "entities_title": get_ui("entities_title") or "Entities",
        "show_original": get_ui("show_original") or "Show original text",
        "show_rephrased": get_ui("show_rephrased") or "Show rephrased text",
        "appears_in": get_ui("appears_in") or "Appears in",
        "chunk_label": chunk_label,
    }

    js_labels = {
        "type": get_ui("type_label") or "Type",
        "description": get_ui("description_label") or "Description",
        "community": get_ui("community_label") or "Community",
        "chunks": get_ui("chunks_label") or "Chunks",
        "chunk": get_ui("chunk_label") or "Chunk",
        "showOriginal": get_ui("show_original") or "Show original text",
        "showRephrased": get_ui("show_rephrased") or "Show rephrased text",
    }

    disclaimer = get_ui("disclaimer") or (
        "The information contained in this document has been extracted and processed "
        "using automated artificial intelligence tools (Knwl.AI). While we strive for "
        "accuracy, this data is generated via machine learning algorithms and natural "
        "language processing, which may result in errors, omissions, or misinterpretations "
        'of the original source material. This document is provided "as is" and for '
        "informational purposes only. Orbifold Consulting makes no warranties, express or "
        "implied, regarding the accuracy, completeness, or reliability of this information. "
        "Users are advised to independently verify any critical data against original source "
        "documents before making business, legal, or financial decisions. Orbifold Consulting "
        "assumes no liability for any actions taken in reliance upon this information."
    )


    # Setup Jinja2 environment
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(f"{template}.html")

    html_content = template.render(
        lang=get_current_language() or "en",
        title=title,
        summary=summary,
        url=url,
        metadata=metadata,
        labels=labels,
        clusters=clusters,
        chunks=chunks_display,
        entities_display=entities_display,
        disclaimer=disclaimer,
        node_elements=node_elements,
        edge_elements=edge_elements,
        js_labels=js_labels,
        rawData=json.dumps(results_data, indent=2),
    )

    return html_content
