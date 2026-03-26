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
class Chunk:
    """Text chunk with associated metadata."""

    text: str
    chunk_idx: int = 0
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class AugmentedChunk(Chunk):
    """Chunk with associated entities and relations."""

    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)


@dataclass
class ClusteredGraph(Graph):
    """Graph with community/cluster information added."""

    clusters: dict = field(default_factory=dict)


@dataclass
class ChunkGraph(Graph):
    """
    Result from extracting a single chunk.
    """

    chunk: Chunk
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

    @staticmethod
    def default() -> "Schema":
        return Schema(
            entity_types=[
                "person",
                "organization",
                "technology",
                "location",
                "project",
                "concept",
                "event",
            ],
            relation_types=[
                "works_at",
                "created",
                "lives_in",
                "located_in",
                "uses",
                "partners_with",
                "supports",
                "integrates_with",
                "related_to",
                "requires",
                "leads_to",
            ],
        )


@dataclass
class KnowledgeGraph:
    """Consolidated insights from a single document."""

    graph: ClusteredGraph
    schema: Schema
    title: str
    summary: str
    url: str | None = None
    chunks: list[AugmentedChunk] = field(default_factory=list)
    language: str | None = None


@dataclass
class DocumentGraph:
    """Result from extracting a document, including the consolidated graph and metadata."""

    graph: ChunkGraph = field(default_factory=ChunkGraph)
    chunks: list[Chunk] = field(default_factory=list)
    schema: Schema = field(default_factory=Schema.default)
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    url: str | None = None
    content: str | None = None
    language: str | None = None
    summary: str | None = None


@dataclass
class ConsolidatedGraph:
    """Result from consolidating multiple document graphs."""

    graph: ClusteredGraph = field(default_factory=ClusteredGraph)
    id: str = field(default_factory=lambda: str(uuid4()))
    documents: list[dict] = field(default_factory=list)
    schema: dict = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
