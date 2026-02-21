"""
LLM backends, caching, language support, and core data models for knwler.

Provides:
- get_resource_dir       – PyInstaller-compatible resource path helper
- Language helpers       – load_languages, get_prompt, get_ui, set_language, …
- Response cache         – _cache_key, _get_cached_response, _save_to_cache
- Data models            – Config, ExtractionResult, Schema
- LLM backends           – ollama_generate, openai_generate, llm_generate
- NLP helpers            – detect_language, parse_json_response, discover_schema
"""

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests
from rich.console import Console


LANGUAGES_FILE = get_resource_dir() / "languages.json"
DEFAULT_LANGUAGE = "en"
_LANGUAGES: dict = {}
_CURRENT_LANG: str = DEFAULT_LANGUAGE

CACHE_DIR = get_resource_dir() / "cache"

DEFAULT_OLLAMA_SCHEMA_MODEL = "qwen2.5:14b"
# qwen2.5:14b gives better quality; 3b is faster for parallel extraction.
DEFAULT_OLLAMA_EXTRACTION_MODEL = "qwen2.5:3b"
DEFAULT_OPENAI_DISCOVERY_MODEL = "gpt-4o"
DEFAULT_OPENAI_EXTRACTION_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Resource directory (PyInstaller compatible)
# ---------------------------------------------------------------------------


def get_resource_dir() -> Path:
    """Return the directory that holds bundled resources (languages.json, templates/).
    Works both when running from source and when frozen with PyInstaller."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).parent


# ---------------------------------------------------------------------------
# Shared console
# ---------------------------------------------------------------------------

console = Console()

# ---------------------------------------------------------------------------
# Language Support
# ---------------------------------------------------------------------------


def load_languages() -> dict:
    """Load language definitions from JSON file."""
    global _LANGUAGES
    if not _LANGUAGES:
        if LANGUAGES_FILE.exists():
            _LANGUAGES = json.loads(LANGUAGES_FILE.read_text(encoding="utf-8"))
        else:
            console.print(
                f"[yellow]Warning: {LANGUAGES_FILE} not found, using English[/yellow]"
            )
            _LANGUAGES = {
                "en": {"name": "English", "prompts": {}, "ui": {}, "console": {}}
            }
    return _LANGUAGES


def get_lang() -> dict:
    """Return the current language dictionary."""
    langs = load_languages()
    return langs.get(_CURRENT_LANG, langs.get(DEFAULT_LANGUAGE, {}))


def get_prompt(key: str, **kwargs) -> str:
    """Return a localized prompt template formatted with *kwargs*."""
    lang = get_lang()
    template = lang.get("prompts", {}).get(key, "")
    if not template:
        template = load_languages().get("en", {}).get("prompts", {}).get(key, "")
    return template.format(**kwargs) if template else ""


def get_ui(key: str, **kwargs) -> str:
    """Return a localized UI string formatted with *kwargs*."""
    lang = get_lang()
    template = lang.get("ui", {}).get(key, "")
    if not template:
        template = load_languages().get("en", {}).get("ui", {}).get(key, "")
    return template.format(**kwargs) if template else ""


def get_console_msg(key: str, **kwargs) -> str:
    """Return a localized console message formatted with *kwargs*."""
    lang = get_lang()
    template = lang.get("console", {}).get(key, "")
    if not template:
        template = load_languages().get("en", {}).get("console", {}).get(key, "")
    return template.format(**kwargs) if template else ""


def set_language(lang_code: str) -> None:
    """Set the active language, falling back to English for unknown codes."""
    global _CURRENT_LANG
    langs = load_languages()
    if lang_code in langs:
        _CURRENT_LANG = lang_code
    else:
        console.print(
            f"[yellow]Language '{lang_code}' not found, using English[/yellow]"
        )
        _CURRENT_LANG = DEFAULT_LANGUAGE


def get_language() -> str:
    """Return the current language code."""
    return _CURRENT_LANG


# ---------------------------------------------------------------------------
# LLM Response Cache
# ---------------------------------------------------------------------------


def _cache_key(prompt: str, model: str, temperature: float, num_predict: int) -> str:
    """Return a SHA-256 cache key derived from the call parameters."""
    content = f"{model}|{temperature}|{num_predict}|{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def _get_cached_response(key: str) -> str | None:
    """Return a cached response string, or *None* if not cached."""
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            return data.get("response")
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _save_to_cache(key: str, response: str, model: str) -> None:
    """Persist *response* to the cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    data = {
        "model": model,
        "response": response,
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cache_file.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # Backend selection
    use_openai: bool = False

    # Ollama settings
    ollama_url: str = "http://localhost:11434/api/generate"

    # OpenAI settings
    openai_api_key: str = "sk-proj--your-own-key"
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

    # Default schema (used when discovery is skipped or fails)
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
        """Active extraction model for the configured backend."""
        return (
            self.openai_extraction_model
            if self.use_openai
            else self.ollama_extraction_model
        )

    @property
    def discovery_model(self) -> str:
        """Active discovery model for the configured backend."""
        return (
            self.openai_discovery_model
            if self.use_openai
            else self.ollama_discovery_model
        )


@dataclass
class ExtractionResult:
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
    entity_types: list[str]
    relation_types: list[str]
    reasoning: str = ""
    discovery_time: float = 0.0


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


def ollama_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call Ollama and return the response text (cached by default)."""
    actual_model = model or config.ollama_extraction_model

    if config.use_cache:
        key = _cache_key(prompt, actual_model, config.temperature, config.num_predict)
        cached = _get_cached_response(key)
        if cached is not None:
            return cached

    payload: dict = {
        "model": actual_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.num_predict,
        },
    }
    if format_json:
        payload["format"] = "json"

    resp = requests.post(config.ollama_url, json=payload, timeout=360)
    resp.raise_for_status()
    response: str = resp.json()["response"]

    if config.use_cache:
        _save_to_cache(key, response, actual_model)

    return response


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


def openai_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call the OpenAI (or compatible) API and return the response text (cached)."""
    actual_model = model or config.openai_extraction_model
    api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "OpenAI API key not set. Use the OPENAI_API_KEY env var or config.openai_api_key."
        )

    if config.use_cache:
        key = _cache_key(prompt, actual_model, config.temperature, config.num_predict)
        cached = _get_cached_response(key)
        if cached is not None:
            return cached

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model": actual_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.num_predict,
    }
    if format_json:
        payload["response_format"] = {"type": "json_object"}

    url = f"{config.openai_base_url.rstrip('/')}/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=360)
    resp.raise_for_status()
    response: str = resp.json()["choices"][0]["message"]["content"]

    if config.use_cache:
        _save_to_cache(key, response, actual_model)

    return response


# ---------------------------------------------------------------------------
# LLM dispatcher
# ---------------------------------------------------------------------------


def llm_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Dispatch to the appropriate backend (Ollama or OpenAI) based on *config*."""
    if config.use_openai:
        return openai_generate(prompt, config, model, format_json)
    return ollama_generate(prompt, config, model, format_json)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_json_response(response: str) -> dict:
    """Parse JSON from an LLM response, returning an empty dict on failure."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def detect_language(text: str, config: Config, sample_size: int = 1000) -> str:
    """Detect the language of *text* via LLM; falls back to English."""
    sample = text[:sample_size] if len(text) > sample_size else text

    prompt = f"""Detect the language of the following text. Return only a JSON object with the ISO 639-1 language code (e.g., "en" for English, "de" for German, "fr" for French, etc.).

TEXT:
\"\"\"{sample}\"\"\"

Return JSON:
{{
  "language": "en"
}}"""

    response = llm_generate(prompt, config, model=config.discovery_model)
    result = parse_json_response(response)

    detected = result.get("language", DEFAULT_LANGUAGE).lower().strip()
    langs = load_languages()
    return detected if detected in langs else DEFAULT_LANGUAGE


# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------


def discover_schema(
    text: str,
    config: Config,
    sample_size: int = 4000,
    max_entity_types: int = 20,
    max_relation_types: int = 25,
) -> Schema:
    """Analyze *text* to discover appropriate entity and relation types."""
    if len(text) <= sample_size:
        sample = text
    else:
        chunk = sample_size // 3
        sample = "\n\n[...]\n\n".join(
            [
                text[:chunk],
                text[len(text) // 2 - chunk // 2 : len(text) // 2 + chunk // 2],
                text[-chunk:],
            ]
        )

    prompt = get_prompt(
        "schema_discovery",
        sample=sample,
        max_entity_types=max_entity_types,
        max_relation_types=max_relation_types,
    )

    if not prompt:
        prompt = f"""Analyze this text and identify the best entity types and relation types for a knowledge graph.

TEXT SAMPLE:
\"\"\"{sample}\"\"\"

Guidelines:
- Use snake_case for all type names
- Be specific but not too narrow (e.g., "person" not "male_scientist")
- Focus on types that appear multiple times or are central to the text
- Entity types: what kinds of things are mentioned?
- Relation types: what relationships exist between them?

Return JSON:
{{
  "entity_types": ["type1", "type2", ...],
  "relation_types": ["rel1", "rel2", ...],
  "reasoning": "Brief explanation"
}}

Max {max_entity_types} entity types and {max_relation_types} relation types."""

    t0 = time.perf_counter()
    response = llm_generate(prompt, config, model=config.discovery_model)
    elapsed = time.perf_counter() - t0

    result = parse_json_response(response)

    if not result.get("entity_types"):
        return Schema(
            entity_types=config.default_entity_types,
            relation_types=config.default_relation_types,
            reasoning="Discovery failed, using defaults",
            discovery_time=elapsed,
        )

    return Schema(
        entity_types=result.get("entity_types", [])[:max_entity_types],
        relation_types=result.get("relation_types", [])[:max_relation_types],
        reasoning=result.get("reasoning", ""),
        discovery_time=elapsed,
    )
