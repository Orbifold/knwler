from dataclasses import dataclass, asdict, fields, field
from collections.abc import Mapping


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Result from extracting a single chunk."""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    chunk_idx: int = -1
    chunk_time: float = 0.0
    chunk_tokens: int = 0

    @property
    def entities_count(self) -> int:
        return len(self.entities)

    @property
    def relations_count(self) -> int:
        return len(self.relations)


@dataclass
class Schema:
    """Discovered or default entity/relation schema."""

    entity_types: list[str]
    relation_types: list[str]
    reasoning: str = ""
    discovery_time: float = 0.0
