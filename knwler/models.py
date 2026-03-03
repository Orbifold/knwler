from dataclasses import dataclass, asdict, fields, field
from collections.abc import Mapping
from uuid import uuid4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    """Extracted graph for a single chunk."""

    entities: list[dict]
    relations: list[dict]


@dataclass
class ExtractionResult(Graph):
    """Result from extracting a single chunk."""

    chunk_idx: int
    chunk_time: float
    chunk_tokens: int
    id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def entities_count(self) -> int:
        return len(self.entities)

    @property
    def relations_count(self) -> int:
        return len(self.relations)


@dataclass
class Schema:
    """Discovered or default entity/relation schema."""

    entity_types: list[str] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)
    reasoning: str = ""
    discovery_time: float = 0.0
