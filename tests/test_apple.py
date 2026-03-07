import pytest
from knwler.chunking import chunk_text
from knwler.config import Config
from knwler.extraction import extract_all
from knwler.models import Schema, ExtractionResult, Graph
from knwler.consolidation import consolidate_extracted_graphs
from dataclasses import asdict
import json
from knwler.export import export_html
from pathlib import Path


# ============================================================================================
# An entity is uniquely defined by its name and type. If the same name appears
# with different types, they are different entities.
# ============================================================================================
@pytest.mark.asyncio
async def test_disambiguation():
    with open("tests/data/apple.md", "r") as f:
        text = f.read()

    chunks = text.split("\n\n")  # simple chunking by paragraphs for this test
    assert len(chunks) == 2  # one about corp, one about fruit
    # couldn't be easier
    schema = Schema(
        entity_types=["Company", "Fruit", "Person"],
        relation_types=["founded_by", "is_a"],
    )
    # if you use a small model it typically fails to disambiguate and merges the two entities into one, which is a common error.
    config = Config(extraction_model="qwen3.5:9b")
    extraction_results = await extract_all(chunks, schema, config=config)
    assert len(extraction_results) == 2
    for g in extraction_results:
        assert g is not None
        assert g.entities is not None
        assert g.relations is not None
        print("\n-------Graph-------\n")
        print(f"\nEntities:")
        for e in g.entities:
            print(f"- {e['name']} ({e['type']}): {e['description']}")
        print(f"\nRelations:")
        for r in g.relations:
            print(
                f"- {r['source']}::{r['source_type']} -> {r['target']}::{r['target_type']} ({r['type']}): {r['description']} [Strength: {r['strength']}]"
            )
    g, _ = await consolidate_extracted_graphs(
        extraction_results,
        config=config,
    )
    print("\n-------Consolidated-------\n")
    print(f"\nEntities:")
    for e in g["entities"]:
        print(f"- {e['name']} ({e['type']}): {e['description']}")
    print(f"\nRelations:")
    for r in g["relations"]:
        print(
            f"- {r['source']}::{r['source_type']} - [{r['type']}] -> {r['target']}::{r['target_type']}: {r['description']} [Strength: {r['strength']}]"
        )

    # assert that apple is correctly disambiguated into two entities with the same name but different types
    assert any(e["name"] == "Apple" and e["type"] == "Company" for e in g["entities"])
    assert any(e["name"] == "Apple" and e["type"] == "Fruit" for e in g["entities"])
    # assert that the relation is correctly extracted with the correct source and target types
    assert any(
        r["source"] == "Apple"
        and r["source_type"] == "Company"
        and r["target"] == "Steve Jobs"
        and r["target_type"] == "Person"
        and r["type"] == "founded_by"
        for r in g["relations"]
    )
    # assert that Gala is an apple variety and not a company
    assert any(
        e["name"] == "Gala"
        and e["type"] == "Fruit"
        and "varieties" in e["description"].lower()
        for e in g["entities"]
    )
    assert any(
        r["source"] == "Apple"
        and r["source_type"] == "Fruit"
        and r["target"] == "Gala"
        and r["target_type"] == "Fruit"
        and r["type"] == "is_a"
        for r in g["relations"]
    )
    extracted_chunks = [
        {
            "id": r.id,
            "chunk_idx": r.chunk_idx,
            "text": chunks[r.chunk_idx],
            "rephrase": chunks[r.chunk_idx],
            "entities": r.entities,
            "relations": r.relations,
            "chunk_time": r.chunk_time,
            "chunk_tokens": r.chunk_tokens,
            "document": "Apples",
        }
        for r in extraction_results
    ]
    output = {
        "id": "Apples",
        "title": "Apple Test",
        "summary": "This graph tests disambiguation of entities with the same name but different types (Apple as a company and Apple as a fruit). It should correctly identify two separate entities for Apple and extract the relation that Apple Inc. was founded by Steve Jobs, while Gala is a variety of Apple fruit.",
        "graph": g,
        "chunks": extracted_chunks,
    }
    # save the output for inspection and debugging
    with open("tests/graph.json", "w") as f:
        json.dump(output, f, indent=2)
    # save the output as HTML for inspection and debugging
    export_html(
        output,
        output_path=Path("tests/graph.html"),
        title="Apple Test",
        template="columns",
    )
