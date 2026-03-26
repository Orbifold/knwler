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
null_console = Console(quiet=True)

# ---------------------------------------------------------------------------
# Project root (parent of the knwler package directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "knwler"

# ---------------------------------------------------------------------------
# Default model names
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_DISCOVERY_MODEL = "qwen2.5:14b"
DEFAULT_OLLAMA_EXTRACTION_MODEL = "qwen2.5:7b"

DEFAULT_OPENAI_DISCOVERY_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EXTRACTION_MODEL = "gpt-4o-mini"

DEFAULT_ANTHROPIC_DISCOVERY_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_EXTRACTION_MODEL = "claude-haiku-4-5-20251001"

DEFAULT_GITHUB_DISCOVERY_MODEL = "github-copilot/claude-opus-4.5"
DEFAULT_GITHUB_EXTRACTION_MODEL = "github-copilot/claude-opus-4.5"

DEFAULT_LMSTUDIO_DISCOVERY_MODEL = "glm-4.7-flash"
DEFAULT_LMSTUDIO_EXTRACTION_MODEL = "glm-4.7-flash"

DEFAULT_GEMINI_DISCOVERY_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_GEMINI_EXTRACTION_MODEL = "gemini-3.1-flash-lite-preview"


# ---------------------------------------------------------------------------
# Default backend URLs
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com/v1"
DEFAULT_GITHUB_BASE_URL = "https://models.inference.ai.azure.com"
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/api/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """
    Pipeline configuration.
    Attributes:
        backend: Which backend to use for LLM calls. One of "ollama", "openai", "anthropic", "github", "lmstudio", or "gemini".
        api_key: API key for the selected backend (if required).
        base_url: Base URL for the selected backend's API.
        extraction_model: Model name to use for information extraction.
        discovery_model: Model name to use for discovery.
        max_tokens: Maximum number of tokens per chunk.
        overlap_tokens: Number of overlapping tokens between chunks.
        max_concurrent: Maximum number of concurrent LLM calls.
        num_predict: Number of tokens to predict in generation calls (set high to avoid truncation).
        temperature: Sampling temperature for generation calls.
        use_cache: Whether to cache LLM responses on disk.
        template: Which prompt template to use (default is "default").
    """

    backend: str = "ollama"
    """Which backend to use for LLM calls. One of "ollama", "openai", "anthropic", "github", "lmstudio", or "gemini"."""

    api_key: str = None
    """API key for the selected backend (if required)."""

    base_url: str = None
    """Base URL for the selected backend's API."""

    extraction_model: str = None
    """Model name to use for information extraction."""

    discovery_model: str = None
    """Model name to use for discovery."""

    max_tokens: int = 400
    """Maximum number of tokens per chunk."""

    overlap_tokens: int = 50
    """Number of overlapping tokens between chunks."""

    max_concurrent: int = 8
    """Maximum number of concurrent LLM calls."""

    num_predict: int = (
        4096  # if too low this will truncate the JSON and it will fail to parse
    )
    """Number of tokens to predict in generation calls (set high to avoid truncation)."""

    temperature: float = 0.1
    """Sampling temperature for generation calls."""

    use_cache: bool = True
    """Whether to cache LLM responses on disk."""

    template: str = "default"
    """Which prompt template to use (default is "default")."""

    def __post_init__(self):
        # in case the user doesn't specify these, set them based on the backend
        if self.base_url is None:
            if self.backend == "openai":
                self.base_url = DEFAULT_OPENAI_BASE_URL
            elif self.backend == "anthropic":
                self.base_url = DEFAULT_ANTHROPIC_URL
            elif self.backend == "github":
                self.base_url = DEFAULT_GITHUB_BASE_URL
            elif self.backend == "lmstudio":
                self.base_url = DEFAULT_LMSTUDIO_BASE_URL
            elif self.backend == "gemini":
                self.base_url = DEFAULT_GEMINI_BASE_URL
            else:
                self.base_url = DEFAULT_OLLAMA_URL

        if self.extraction_model is None:
            if self.backend == "openai":
                self.extraction_model = DEFAULT_OPENAI_EXTRACTION_MODEL
            elif self.backend == "anthropic":
                self.extraction_model = DEFAULT_ANTHROPIC_EXTRACTION_MODEL
            elif self.backend == "github":
                self.extraction_model = DEFAULT_GITHUB_EXTRACTION_MODEL
            elif self.backend == "lmstudio":
                self.extraction_model = DEFAULT_LMSTUDIO_EXTRACTION_MODEL
            elif self.backend == "gemini":
                self.extraction_model = DEFAULT_GEMINI_EXTRACTION_MODEL
            else:
                self.extraction_model = DEFAULT_OLLAMA_EXTRACTION_MODEL

        if self.discovery_model is None:
            if self.backend == "openai":
                self.discovery_model = DEFAULT_OPENAI_DISCOVERY_MODEL
            elif self.backend == "anthropic":
                self.discovery_model = DEFAULT_ANTHROPIC_DISCOVERY_MODEL
            elif self.backend == "github":
                self.discovery_model = DEFAULT_GITHUB_DISCOVERY_MODEL
            elif self.backend == "lmstudio":
                self.discovery_model = DEFAULT_LMSTUDIO_DISCOVERY_MODEL
            elif self.backend == "gemini":
                self.discovery_model = DEFAULT_GEMINI_DISCOVERY_MODEL
            else:
                self.discovery_model = DEFAULT_OLLAMA_DISCOVERY_MODEL
