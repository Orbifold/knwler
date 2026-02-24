"""
Configuration, data classes, and shared constants.
"""

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

# ---------------------------------------------------------------------------
# Shared console
# ---------------------------------------------------------------------------
console = Console(record=True)

# ---------------------------------------------------------------------------
# Project root (parent of the knwler package directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "knwler"

# ---------------------------------------------------------------------------
# Default model names
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_SCHEMA_MODEL = "qwen2.5:14b"
DEFAULT_OLLAMA_EXTRACTION_MODEL = "qwen2.5:3b"
DEFAULT_OPENAI_DISCOVERY_MODEL = "gpt-4o"
DEFAULT_OPENAI_EXTRACTION_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Pipeline configuration."""

    # Backend selection
    use_openai: bool = False

    # Ollama settings
    ollama_url: str = "http://localhost:11434/api/generate"

    # OpenAI settings
    openai_api_key: str = None
    openai_base_url: str = "https://api.openai.com/v1"

    # Model settings
    ollama_extraction_model: str = DEFAULT_OLLAMA_EXTRACTION_MODEL
    ollama_discovery_model: str = DEFAULT_OLLAMA_SCHEMA_MODEL
    openai_extraction_model: str = DEFAULT_OPENAI_EXTRACTION_MODEL
    openai_discovery_model: str = DEFAULT_OPENAI_DISCOVERY_MODEL
    max_tokens: int = 400
    overlap_tokens: int = 50
    max_concurrent: int = 8
    num_predict: int = 1024
    temperature: float = 0.1
    use_cache: bool = True

    # Default schema (used if discovery is skipped or fails)
    default_entity_types: list[str] = field(
        default_factory=lambda: [
            "person",
            "organization",
            "technology",
            "location",
            "project",
            "concept",
            "event",
        ]
    )
    default_relation_types: list[str] = field(
        default_factory=lambda: [
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
        ]
    )

    @property
    def extraction_model(self) -> str:
        """Return the extraction model for the active backend."""
        return (
            self.openai_extraction_model
            if self.use_openai
            else self.ollama_extraction_model
        )

    @property
    def discovery_model(self) -> str:
        """Return the discovery model for the active backend."""
        return (
            self.openai_discovery_model
            if self.use_openai
            else self.ollama_discovery_model
        )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """Result from extracting a single chunk."""

    entities: list[dict]
    relations: list[dict]
    chunk_idx: int
    chunk_time: float
    chunk_tokens: int

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
