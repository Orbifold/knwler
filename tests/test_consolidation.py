import pytest
import json

from knwler import consolidate_graphs, Graph


def test_graph_consolidation():
    with open("tests/data/doc1.json", "r") as f:
        doc1 = json.loads(f.read())
    with open("tests/data/doc2.json", "r") as f:
        doc2 = json.loads(f.read())
    result = consolidate_graphs([doc1, doc2], True)
    with open("tests/data/consolidated.json", "w") as f:
        json.dump(result, f, indent=2)
