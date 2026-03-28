import pytest
import json
from dataclasses import asdict
from knwler import (
    consolidate_document_graphs,
    Graph,
    consolidate_chunk_graphs,
    Chunk,
    DocumentGraph,
    ConsolidatedGraph,
)


@pytest.mark.asyncio
async def test_graph_consolidation():
    with open("tests/data/doc1.json", "r") as f:
        doc1 = DocumentGraph(**json.loads(f.read()))
    with open("tests/data/doc2.json", "r") as f:
        doc2 = DocumentGraph(**json.loads(f.read()))
    result = await consolidate_document_graphs([doc1, doc2], True)
    with open("tests/data/consolidated.json", "w") as f:
        json.dump(asdict(result), f, indent=2)


@pytest.mark.asyncio
async def test_no_summary():
    g1 = Graph(
        entities=[
            {"name": "Alice", "type": "Person", "description": "A1"},
            {"name": "Bob", "type": "Person", "description": "B"},
        ],
        relations=[],
    )
    g2 = Graph(
        entities=[
            {"name": "Alice", "type": "Person", "description": "A2"},
        ],
        relations=[],
    )
    # if low importance filtering is on, you will get empty
    consolidated, consolidation_time = await consolidate_chunk_graphs(
        [g1, g2], summarize=False, filter_low_importance=False
    )
    assert isinstance(consolidated, Graph)
    assert hasattr(consolidated, "entities") and hasattr(consolidated, "relations")
    assert len(consolidated.entities) == 2
    print(json.dumps(asdict(consolidated), indent=2))
    assert consolidated.entities[0]["name"] == "Alice"
    assert consolidated.entities[0]["type"] == "Person"
    # not summary means concatenating descriptions, so we expect both A1 and A2 to be in the description of Alice
    assert consolidated.entities[0]["description"] == "A1 A2"
