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
DEFAULT_ANTHROPIC_DISCOVERY_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_EXTRACTION_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Default backend URLs
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com/v1"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Pipeline configuration."""

    # Backend selection: "ollama" | "openai" | "anthropic"
    backend: str = "ollama"

    api_key: str = None
    base_url: str = (
        DEFAULT_OPENAI_BASE_URL
        if backend == "openai"
        else (DEFAULT_ANTHROPIC_URL if backend == "anthropic" else DEFAULT_OLLAMA_URL)
    )

    # Model settings
    extraction_model: str = (
        DEFAULT_OPENAI_EXTRACTION_MODEL
        if backend == "openai"
        else (
            DEFAULT_ANTHROPIC_EXTRACTION_MODEL
            if backend == "anthropic"
            else DEFAULT_OLLAMA_EXTRACTION_MODEL
        )
    )
    discovery_model: str = (
        DEFAULT_OPENAI_DISCOVERY_MODEL
        if backend == "openai"
        else (
            DEFAULT_ANTHROPIC_DISCOVERY_MODEL
            if backend == "anthropic"
            else DEFAULT_OLLAMA_SCHEMA_MODEL
        )
    )
    max_tokens: int = 400
    overlap_tokens: int = 50
    max_concurrent: int = 8
    num_predict: int = (
        4096  # if too low this will truncate the JSON and it will fail to parse
    )
    temperature: float = 0.1
    use_cache: bool = True
    template: str = "default"

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
    def use_openai(self) -> bool:
        return self.backend == "openai"

    @property
    def use_anthropic(self) -> bool:
        return self.backend == "anthropic"

    @property
    def use_ollama(self) -> bool:
        return self.backend == "ollama"
