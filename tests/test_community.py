import pytest
import networkx as nx
from knwler.clustering import create_network
from knwler.models import *
"""
Tests for clustering.py
"""


def test_create_network_with_entities_and_relations():
    """Test creating a network with entities and relations."""
    consolidated = {
        "entities": [
            {"name": "Alice", "type": "person", "description": "A person"},
            {"name": "Bob", "type": "person", "description": "Another person"},
        ],
        "relations": [
            {
                "source": "Alice",
                "source_type": "person",
                "target": "Bob",
                "target_type": "person",
                "type": "knows",
                "description": "knows each other",
            },
        ],
    }

    g = create_network(consolidated)

    assert isinstance(g, nx.MultiDiGraph)
    assert g.number_of_nodes() == 2 + 1  # 2 entities + 1 document node
    assert g.number_of_edges() == 1
    assert "Alice::person" in g.nodes()
    assert "Bob::person" in g.nodes()
    assert g.nodes["Alice::person"]["type"] == "person"
    assert g.nodes["Bob::person"]["description"] == "Another person"
    assert g.has_edge("Alice::person", "Bob::person")


def test_create_network_with_cluster_id():
    """Test that cluster_id is preserved when present."""
    consolidated = {
        "entities": [
            {
                "name": "Entity1",
                "type": "type1",
                "description": "desc1",
                "cluster_id": 0,
            },
        ],
        "relations": [],
    }

    g = create_network(consolidated)

    assert g.nodes["Entity1::type1"]["cluster_id"] == 0


def test_create_network_without_cluster_id():
    """Test that cluster_id defaults to None when not present."""
    consolidated = {
        "entities": [
            {"name": "Entity1", "type": "type1", "description": "desc1"},
        ],
        "relations": [],
    }

    g = create_network(consolidated)

    assert g.nodes["Entity1::type1"]["cluster_id"] is None


def test_create_network_empty():
    """Test creating a network with no entities or relations."""
    consolidated = {"entities": [], "relations": []}

    g = create_network(consolidated)

    assert isinstance(g, nx.MultiDiGraph)
    assert g.number_of_nodes() == 1  # 0 entities + 1 document node
    assert g.number_of_edges() == 0


def test_create_network_edge_attributes():
    """Test that edge attributes are correctly set."""
    consolidated = {
        "entities": [
            {"name": "A", "type": "t1", "description": "d1"},
            {"name": "B", "type": "t2", "description": "d2"},
        ],
        "relations": [
            {
                "source": "A",
                "source_type": "t1",
                "target": "B",
                "target_type": "t2",
                "type": "connected",
                "description": "edge desc",
            },
        ],
    }

    g = create_network(consolidated)

    edge_data = g.get_edge_data("A::t1", "B::t2")
    assert edge_data[0]["type"] == "connected"
    assert edge_data[0]["description"] == "edge desc"


def test_create_network_multiple_edges():
    """Test creating a network with multiple edges."""
    consolidated = {
        "entities": [
            {"name": "X", "type": "t1", "description": "d1"},
            {"name": "Y", "type": "t2", "description": "d2"},
            {"name": "Z", "type": "t3", "description": "d3"},
        ],
        "relations": [
            {
                "source": "X",
                "source_type": "t1",
                "target": "Y",
                "target_type": "t2",
                "type": "rel1",
                "description": "d1",
            },
            {
                "source": "Y",
                "source_type": "t2",
                "target": "Z",
                "target_type": "t3",
                "type": "rel2",
                "description": "d2",
            },
            {
                "source": "X",
                "source_type": "t1",
                "target": "Z",
                "target_type": "t3",
                "type": "rel3",
                "description": "d3",
            },
        ],
    }

    g = create_network(consolidated)

    assert g.number_of_nodes() == 3 + 1  # 3 entities + 1 document node
    assert g.number_of_edges() == 3
