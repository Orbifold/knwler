"""
Graph Extraction Pipeline
=========================
Extract knowledge graphs from text using Ollama or OpenAI with automatic schema discovery.

Uses two models:
- Discovery model (default: qwen2.5:14b) - larger model for schema discovery
- Extraction model (default: qwen2.5:3b) - faster model for parallel chunk extraction

LLM responses are cached to cache/ in the project root based on prompt+model hash.

Usage:
    python benchmark.py                                    # Use defaults (Ollama)
    python benchmark.py --file path/to/doc.txt             # Custom file
    python benchmark.py -e qwen2.5:7b                      # Different extraction model
    python benchmark.py -d qwen2.5:32b                     # Different discovery model
    python benchmark.py --no-discover                      # Skip schema discovery
    python benchmark.py --no-cache                         # Disable response caching
    python benchmark.py -o results.json                    # Save results to JSON
    python benchmark.py --help                             # Show all options

    # OpenAI usage (requires OPENAI_API_KEY env var):
    python benchmark.py --openai -e gpt-4o-mini -d gpt-4o
    python benchmark.py --openai --openai-base-url https://api.example.com/v1
"""

import sys
import networkx as nx
from networkx.algorithms.community import louvain_communities
from datetime import datetime
import asyncio
import hashlib
import html as html_mod
import json
import os
import re
import time
import fitz  # PyMuPDF

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Annotated, Any, Optional

import requests
import tiktoken
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Rich console and Typer app
console = Console()
app = typer.Typer(
    help="Extract knowledge graphs from text using Ollama or OpenAI.",
    rich_markup_mode="rich",
)

# ---------------------------------------------------------------------------
# Language Support
# ---------------------------------------------------------------------------
LANGUAGES_FILE = Path(__file__).parent / "languages.json"
DEFAULT_LANGUAGE = "en"
_LANGUAGES: dict = {}
_CURRENT_LANG: str = DEFAULT_LANGUAGE


def load_languages() -> dict:
    """Load language definitions from JSON file."""
    global _LANGUAGES
    if not _LANGUAGES:
        if LANGUAGES_FILE.exists():
            _LANGUAGES = json.loads(LANGUAGES_FILE.read_text(encoding="utf-8"))
        else:
            console.print(f"[yellow]Warning: {LANGUAGES_FILE} not found, using English[/yellow]")
            _LANGUAGES = {"en": {"name": "English", "prompts": {}, "ui": {}, "console": {}}}
    return _LANGUAGES


def get_lang() -> dict:
    """Get the current language dictionary."""
    langs = load_languages()
    return langs.get(_CURRENT_LANG, langs.get(DEFAULT_LANGUAGE, {}))


def get_prompt(key: str, **kwargs) -> str:
    """Get a localized prompt template, formatted with kwargs."""
    lang = get_lang()
    template = lang.get("prompts", {}).get(key, "")
    if not template:
        # Fallback to English
        template = load_languages().get("en", {}).get("prompts", {}).get(key, "")
    return template.format(**kwargs) if template else ""


def get_ui(key: str, **kwargs) -> str:
    """Get a localized UI string, formatted with kwargs."""
    lang = get_lang()
    template = lang.get("ui", {}).get(key, "")
    if not template:
        template = load_languages().get("en", {}).get("ui", {}).get(key, "")
    return template.format(**kwargs) if template else ""


def get_console_msg(key: str, **kwargs) -> str:
    """Get a localized console message, formatted with kwargs."""
    lang = get_lang()
    template = lang.get("console", {}).get(key, "")
    if not template:
        template = load_languages().get("en", {}).get("console", {}).get(key, "")
    return template.format(**kwargs) if template else ""


def set_language(lang_code: str):
    """Set the current language."""
    global _CURRENT_LANG
    langs = load_languages()
    if lang_code in langs:
        _CURRENT_LANG = lang_code
    else:
        console.print(f"[yellow]Language '{lang_code}' not found, using English[/yellow]")
        _CURRENT_LANG = DEFAULT_LANGUAGE

# ---------------------------------------------------------------------------
# LLM Response Cache
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / "cache"
DEFAULT_OLLAMA_SCHEMA_MODEL = "qwen2.5:14b"
# qwen2.5:14b gives better quality but 3b is good for faster extraction, especially with many chunks. Adjust as needed based on your hardware and quality requirements.
DEFAULT_OLLAMA_EXTRACTION_MODEL = "qwen2.5:3b"
DEFAULT_OPENAI_DISCOVERY_MODEL = "gpt-4o"
DEFAULT_OPENAI_EXTRACTION_MODEL = "gpt-4o-mini"


def _cache_key(prompt: str, model: str, temperature: float, num_predict: int) -> str:
    """Generate a cache key from prompt and model parameters."""
    content = f"{model}|{temperature}|{num_predict}|{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def _get_cached_response(key: str) -> str | None:
    """Retrieve cached response if it exists."""
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            return data.get("response")
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _save_to_cache(key: str, response: str, model: str):
    """Save response to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    data = {
        "model": model,
        "response": response,
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cache_file.write_text(json.dumps(data, indent=2))


DEFAULT_FILE = "./small.md"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Backend selection
    use_openai: bool = False  # If True, use OpenAI API instead of Ollama

    # Ollama settings
    ollama_url: str = "http://localhost:11434/api/generate"

    # OpenAI settings
    openai_api_key: str = (
        "sk-proj--your-own-key"  # Set via OPENAI_API_KEY env var or directly
    )
    openai_base_url: str = (
        "https://api.openai.com/v1"  # Can override for OpenAI-compatible APIs
    )

    # Model settings
    ollama_extraction_model: str = (
        DEFAULT_OLLAMA_EXTRACTION_MODEL  # Fast model for parallel chunk extraction
    )
    ollama_discovery_model: str = (
        DEFAULT_OLLAMA_SCHEMA_MODEL  # Larger model for schema discovery
    )
    openai_extraction_model: str = (
        DEFAULT_OPENAI_EXTRACTION_MODEL  # OpenAI model for extraction
    )
    openai_discovery_model: str = (
        DEFAULT_OPENAI_DISCOVERY_MODEL  # OpenAI model for discovery
    )
    max_tokens: int = 400
    overlap_tokens: int = 50
    max_concurrent: int = 8
    num_predict: int = 1024
    temperature: float = 0.1
    use_cache: bool = True  # Cache LLM responses

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
# Cached encoder
# ---------------------------------------------------------------------------
_ENCODER: tiktoken.Encoding | None = None


def get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------
def ollama_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call Ollama and return the response text.

    Caches responses based on hash of prompt + model parameters.
    """
    actual_model = model or config.ollama_extraction_model

    # Check cache first
    if config.use_cache:
        cache_key = _cache_key(
            prompt, actual_model, config.temperature, config.num_predict
        )
        cached = _get_cached_response(cache_key)
        if cached is not None:
            return cached

    # Make API call
    payload = {
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
    response = resp.json()["response"]

    # Save to cache
    if config.use_cache:
        _save_to_cache(cache_key, response, actual_model)

    return response


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------
def openai_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call OpenAI API and return the response text.

    Caches responses based on hash of prompt + model parameters.
    Works with OpenAI API and compatible endpoints (e.g., Azure, local servers).
    """
    actual_model = model or config.openai_extraction_model
    api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "OpenAI API key not set. Set OPENAI_API_KEY env var or config.openai_api_key"
        )

    # Check cache first
    if config.use_cache:
        cache_key = _cache_key(
            prompt, actual_model, config.temperature, config.num_predict
        )
        cached = _get_cached_response(cache_key)
        if cached is not None:
            return cached

    # Build request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": actual_model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.num_predict,
    }

    if format_json:
        payload["response_format"] = {"type": "json_object"}

    url = f"{config.openai_base_url.rstrip('/')}/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=360)
    resp.raise_for_status()
    response = resp.json()["choices"][0]["message"]["content"]

    # Save to cache
    if config.use_cache:
        _save_to_cache(cache_key, response, actual_model)

    return response


# ---------------------------------------------------------------------------
# LLM Dispatcher
# ---------------------------------------------------------------------------
def llm_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Dispatch to appropriate LLM backend based on config."""
    if config.use_openai:
        return openai_generate(prompt, config, model, format_json)
    return ollama_generate(prompt, config, model, format_json)


# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------
def detect_language(text: str, config: Config, sample_size: int = 1000) -> str:
    """Detect the language of the input text using LLM."""
    # Take a sample from the text
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
    
    # Validate that we support this language, fallback to English if not
    langs = load_languages()
    if detected not in langs:
        return DEFAULT_LANGUAGE
    
    return detected


def parse_json_response(response: str) -> dict:
    """Parse JSON from response, handling edge cases."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON object from response
        # match = re.search(r"\{.*\}", response, re.DOTALL)
        # if match:
        #     return json.loads(match.group())
        return {}


# ---------------------------------------------------------------------------
# Schema Discovery
# ---------------------------------------------------------------------------
def discover_schema(
    text: str,
    config: Config,
    sample_size: int = 4000,
    max_entity_types: int = 20,
    max_relation_types: int = 25,
) -> Schema:
    """Analyze text to discover appropriate entity and relation types."""

    # Sample from beginning, middle, and end
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

    # Use localized prompt
    prompt = get_prompt(
        "schema_discovery",
        sample=sample,
        max_entity_types=max_entity_types,
        max_relation_types=max_relation_types,
    )
    
    # Fallback to English prompt if localization fails
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


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, config: Config) -> list[str]:
    """Split text into overlapping chunks by token count."""
    enc = get_encoder()
    tokens = enc.encode(text)

    if len(tokens) <= config.max_tokens:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + config.max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]

        # Try to break at sentence boundary
        if end < len(tokens):
            for i in range(len(chunk_tokens) - 1, max(len(chunk_tokens) - 100, 0), -1):
                if enc.decode([chunk_tokens[i]]).rstrip().endswith((".", "!", "?")):
                    chunk_tokens = chunk_tokens[: i + 1]
                    end = start + i + 1
                    break

        chunks.append(enc.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start = end - config.overlap_tokens

    return chunks


# ---------------------------------------------------------------------------
# Graph Extraction
# ---------------------------------------------------------------------------
def extract_graph(text: str, schema: Schema, config: Config) -> dict[str, Any]:
    """Extract entities and relations from text."""
    # Use localized prompt
    prompt = get_prompt(
        "extract_graph",
        entity_types=json.dumps(schema.entity_types),
        relation_types=json.dumps(schema.relation_types),
        text=text,
    )
    
    # Fallback to English prompt if localization fails
    if not prompt:
        prompt = f"""Extract a knowledge graph from the text below.

ENTITY TYPES: {json.dumps(schema.entity_types)}
RELATION TYPES: {json.dumps(schema.relation_types)}

RULES:
- Only extract entities and relations clearly stated in the text
- Each entity: name, type, 1-2 sentence description
- Each relation: source, target, type, brief description, strength (1-10)

TEXT:
\"\"\"{text}\"\"\"

Return JSON:
{{
  "entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "relations": [{{"source": "...", "target": "...", "type": "...", "description": "...", "strength": 8}}]
}}"""

    response = llm_generate(prompt, config)
    result = parse_json_response(response)

    return {
        "entities": result.get("entities", []),
        "relations": result.get("relations", []),
    }


def extract_chunk(
    chunk: str, idx: int, schema: Schema, config: Config
) -> ExtractionResult:
    """Extract from a single chunk with timing."""
    t0 = time.perf_counter()
    result = extract_graph(chunk, schema, config)
    elapsed = time.perf_counter() - t0

    return ExtractionResult(
        entities=result["entities"],
        relations=result["relations"],
        chunk_idx=idx,
        chunk_time=elapsed,
        chunk_tokens=len(get_encoder().encode(chunk)),
    )


def _save_partial_results(
    output_path: Path,
    schema: Schema,
    results: list[ExtractionResult],
    completed: int,
    total: int,
):
    """Save intermediate results to a partial file."""
    partial_path = output_path.with_suffix(".partial.json")
    output = {
        "status": "in_progress",
        "progress": f"{completed}/{total}",
        "schema": {
            "entity_types": schema.entity_types,
            "relation_types": schema.relation_types,
            "reasoning": schema.reasoning,
        },
        "results": [
            {
                "chunk_idx": r.chunk_idx,
                "entities": r.entities,
                "relations": r.relations,
                "chunk_time": r.chunk_time,
                "chunk_tokens": r.chunk_tokens,
            }
            for r in sorted(results, key=lambda x: x.chunk_idx)
        ],
    }
    partial_path.write_text(json.dumps(output, indent=2))


async def extract_all(
    chunks: list[str],
    schema: Schema,
    config: Config,
    output_path: Path | None = None,
) -> list[ExtractionResult]:
    """Extract from all chunks with concurrency control and incremental saving."""
    semaphore = asyncio.Semaphore(config.max_concurrent)
    loop = asyncio.get_event_loop()
    total = len(chunks)
    results: list[ExtractionResult] = []
    lock = asyncio.Lock()
    progress_msg = get_console_msg("extracting") or "Extracting..."

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"[cyan]{progress_msg}", total=total)

        async def bounded(chunk: str, idx: int) -> ExtractionResult:
            async with semaphore:
                result = await loop.run_in_executor(
                    None, extract_chunk, chunk, idx, schema, config
                )
                async with lock:
                    results.append(result)
                    # Save intermediate results
                    if output_path:
                        _save_partial_results(
                            output_path, schema, results, len(results), total
                        )
                    progress.update(task, advance=1)
                return result

        await asyncio.gather(*[bounded(c, i) for i, c in enumerate(chunks)])

    # Remove partial file on successful completion
    if output_path:
        partial_path = output_path.with_suffix(".partial.json")
        if partial_path.exists():
            partial_path.unlink()

    return sorted(results, key=lambda x: x.chunk_idx)


# ---------------------------------------------------------------------------
# Graph Consolidation
# ---------------------------------------------------------------------------
def consolidate_graphs(
    results: list[ExtractionResult],
    config: Config,
    summarize: bool = True,
) -> tuple[dict, float]:
    """Consolidate chunk graphs with unique (name, type) and summarized descriptions.

    Returns (consolidated_graph, consolidation_time).
    """
    t0 = time.perf_counter()

    # Phase 1: Aggregate by key
    entity_map: dict[tuple[str, str], dict] = {}
    relation_map: dict[tuple[str, str, str], dict] = {}

    for r in results:
        for e in r.entities:
            name = (e.get("name") or "").strip()
            etype = (e.get("type") or "").strip()
            desc = (e.get("description") or "").strip()
            if not name or not etype:
                continue
            key = (name.lower(), etype.lower())
            if key not in entity_map:
                entity_map[key] = {
                    "name": name,
                    "type": etype,
                    "descriptions": [],
                    "chunk_ids": set(),
                }
            entity_map[key]["chunk_ids"].add(r.chunk_idx)
            if desc and desc not in entity_map[key]["descriptions"]:
                entity_map[key]["descriptions"].append(desc)

        for rel in r.relations:
            src = (rel.get("source") or "").strip()
            tgt = (rel.get("target") or "").strip()
            rtype = (rel.get("type") or "").strip()
            desc = (rel.get("description") or "").strip()
            strength = rel.get("strength", 5)
            if not src or not tgt or not rtype:
                continue
            key = (src.lower(), tgt.lower(), rtype.lower())
            if key not in relation_map:
                relation_map[key] = {
                    "source": src,
                    "target": tgt,
                    "type": rtype,
                    "descriptions": [],
                    "strengths": [],
                    "chunk_ids": set(),
                }
            relation_map[key]["chunk_ids"].add(r.chunk_idx)
            relation_map[key]["strengths"].append(strength)
            if desc and desc not in relation_map[key]["descriptions"]:
                relation_map[key]["descriptions"].append(desc)

    # Phase 2: Summarize descriptions that need merging
    if summarize:
        entity_map, relation_map = _summarize_descriptions(
            entity_map, relation_map, config
        )

    # Phase 3: Build final output
    entities = [
        {
            "name": e["name"],
            "type": e["type"],
            "description": e["descriptions"][0] if e["descriptions"] else "",
            "chunk_ids": sorted(e["chunk_ids"]),
        }
        for e in entity_map.values()
    ]
    relations = [
        {
            "source": r["source"],
            "target": r["target"],
            "type": r["type"],
            "description": r["descriptions"][0] if r["descriptions"] else "",
            "strength": round(sum(r["strengths"]) / len(r["strengths"]), 1),
            "chunk_ids": sorted(r["chunk_ids"]),
        }
        for r in relation_map.values()
    ]

    # Phase 4: Remove low-importance nodes
    entities, relations = _filter_low_importance_nodes(entities, relations)

    elapsed = time.perf_counter() - t0
    return {"entities": entities, "relations": relations}, elapsed


def _is_meaningful_name(name: str) -> bool:
    """Check if an entity name is meaningful (not just numbers, punctuation, etc.)."""
    name = name.strip()

    # Empty or whitespace-only
    if not name:
        return False

    # Pure numbers (integers, floats, percentages)
    if re.match(r"^[\d.,\-+%$€£¥]+$", name):
        return False

    # Single character (except meaningful ones like abbreviations)
    if len(name) == 1 and not name.isalpha():
        return False

    # Only punctuation/symbols
    if re.match(r"^[\W_]+$", name):
        return False

    # Common placeholder/filler words
    filler_words = {
        "the",
        "a",
        "an",
        "it",
        "this",
        "that",
        "these",
        "those",
        "etc",
        "n/a",
        "na",
        "none",
    }
    if name.lower() in filler_words:
        return False

    return True


def _filter_low_importance_nodes(
    entities: list[dict],
    relations: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Remove low-importance entities: meaningless names or zero degree."""
    # Build set of connected entity names (case-insensitive lookup)
    connected = set()
    for r in relations:
        connected.add(r["source"].lower())
        connected.add(r["target"].lower())

    original_count = len(entities)

    # Filter entities
    filtered_entities = [
        e
        for e in entities
        if _is_meaningful_name(e["name"]) and e["name"].lower() in connected
    ]

    removed = original_count - len(filtered_entities)
    if removed > 0:
        msg = get_console_msg("removed_entities", count=removed) or f"Removed {removed} low-importance entities (meaningless name or zero degree)"
        console.print(f"[dim]{msg}[/dim]")

    return filtered_entities, relations


def _summarize_descriptions(
    entity_map: dict,
    relation_map: dict,
    config: Config,
) -> tuple[dict, dict]:
    """Batch summarize entities/relations with multiple descriptions."""

    # Collect items needing summarization
    to_summarize = []
    for key, e in entity_map.items():
        if len(e["descriptions"]) > 1:
            to_summarize.append(
                {
                    "id": f"entity:{key[0]}:{key[1]}",
                    "name": e["name"],
                    "type": e["type"],
                    "descriptions": e["descriptions"],
                }
            )
    for key, r in relation_map.items():
        if len(r["descriptions"]) > 1:
            to_summarize.append(
                {
                    "id": f"relation:{key[0]}:{key[1]}:{key[2]}",
                    "name": f"{r['source']} -> {r['target']}",
                    "type": r["type"],
                    "descriptions": r["descriptions"],
                }
            )

    if not to_summarize:
        return entity_map, relation_map

    # Batch into chunks of ~20 items to keep prompt manageable
    batch_size = 20
    summaries = {}
    total_batches = (len(to_summarize) + batch_size - 1) // batch_size
    progress_msg = get_console_msg("summarizing", count=len(to_summarize)) or f"Summarizing {len(to_summarize)} items..."

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            f"[cyan]{progress_msg}", total=total_batches
        )

        for i in range(0, len(to_summarize), batch_size):
            batch = to_summarize[i : i + batch_size]
            prompt = _build_summarization_prompt(batch)
            response = llm_generate(prompt, config, model=config.extraction_model)
            result = parse_json_response(response)
            summaries.update(result.get("summaries", {}))
            progress.update(task, advance=1)

    # Apply summaries back
    for key, e in entity_map.items():
        sid = f"entity:{key[0]}:{key[1]}"
        if sid in summaries:
            e["descriptions"] = [summaries[sid]]
        elif len(e["descriptions"]) > 1:
            # Fallback: join if summarization failed
            e["descriptions"] = [" ".join(e["descriptions"])]

    for key, r in relation_map.items():
        sid = f"relation:{key[0]}:{key[1]}:{key[2]}"
        if sid in summaries:
            r["descriptions"] = [summaries[sid]]
        elif len(r["descriptions"]) > 1:
            r["descriptions"] = [" ".join(r["descriptions"])]

    return entity_map, relation_map


def _build_summarization_prompt(items: list[dict]) -> str:
    """Build prompt for batch description summarization."""
    lines = []
    for item in items:
        descs = ", ".join(f'"{d}"' for d in item["descriptions"])
        lines.append(f'- "{item["id"]}": [{descs}]')
    items_text = "\n".join(lines)
    
    # Use localized prompt
    prompt = get_prompt("summarize_descriptions", items_text=items_text)
    
    # Fallback to English prompt if localization fails
    if not prompt:
        prompt = f"""Summarize multiple descriptions for each item into ONE concise description (1-2 sentences).

ITEMS:
{items_text}

Return JSON:
{{
  "summaries": {{
    "<id>": "<merged description>",
    ...
  }}
}}"""
    return prompt


def rephrase_chunks(chunks: list[str], config: Config) -> list[str]:
    """Rephrase each chunk in simple language for UI display."""
    rephrased = []
    progress_msg = get_console_msg("rephrasing") or "Rephrasing..."

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"[cyan]{progress_msg}", total=len(chunks))

        for chunk in chunks:
            # Use localized prompt
            prompt = get_prompt("rephrase_chunk", chunk=chunk)
            
            # Fallback to English prompt if localization fails
            if not prompt:
                prompt = (
                    "Rewrite the following text so that it is easy to understand. "
                    "Keep all the key facts and names. Convey the information clearly and concisely.\n\n"
                    f'TEXT:\n"""{ chunk }"""\n\n'
                    'Return JSON:\n{\n  "rephrase": "..."\n}'
                )
            response = llm_generate(prompt, config, model=config.extraction_model)
            parsed = parse_json_response(response)
            rephrased.append(parsed.get("rephrase", chunk))
            progress.update(task, advance=1)

    return rephrased


def extract_title(chunks: list[str], config: Config, max_chunks: int = 3) -> str:
    """Extract a short document title from the first few chunks."""
    sample = "\n\n".join(chunks[:max_chunks])
    
    # Use localized prompt
    prompt = get_prompt("extract_title", sample=sample)
    
    # Fallback to English prompt if localization fails
    if not prompt:
        prompt = (
            "Create a short, clear document title based on the text below. "
            "Return only a JSON object.\n\n"
            f'TEXT:\n"""{sample}"""\n\n'
            'Return JSON:\n{\n  "title": "..."\n}'
        )
    response = llm_generate(prompt, config, model=config.extraction_model)
    parsed = parse_json_response(response)
    return (parsed.get("title") or "").strip()


def extract_summary(chunks: list[str], config: Config, max_chunks: int = 3) -> str:
    """Summarize the document based on the first few chunks."""
    sample = "\n\n".join(chunks[:max_chunks])
    
    # Use localized prompt
    prompt = get_prompt("extract_summary", sample=sample)
    
    # Fallback to English prompt if localization fails
    if not prompt:
        prompt = (
            "Summarize the document below in 3-5 sentences. "
            "Focus on the main topic and key points. Return only JSON.\n\n"
            f'TEXT:\n"""{sample}"""\n\n'
            'Return JSON:\n{\n  "summary": "..."\n}'
        )
    response = llm_generate(prompt, config, model=config.extraction_model)
    parsed = parse_json_response(response)
    return (parsed.get("summary") or "").strip()


# ---------------------------------------------------------------------------
# HTML Export
# ---------------------------------------------------------------------------
def _linkify_entities(text: str, entity_names: set[str]) -> str:
    """Replace known entity names in text with hyperlinks to #entity-<name>."""
    esc = html_mod.escape(text)
    if not entity_names:
        return esc

    esc_to_anchor: dict[str, str] = {}
    esc_names: list[str] = []
    for name in entity_names:
        esc_name = html_mod.escape(name)
        key = esc_name.lower()
        if key in esc_to_anchor:
            continue
        esc_to_anchor[key] = f"entity-{name.lower().replace(' ', '-')}"
        esc_names.append(esc_name)

    # Sort by length descending so longer names match first.
    esc_names.sort(key=len, reverse=True)
    pattern = r"(?i)\b(" + "|".join(re.escape(n) for n in esc_names) + r")\b"

    def repl(match: re.Match) -> str:
        matched = match.group(1)
        anchor = esc_to_anchor.get(matched.lower())
        if not anchor:
            return matched
        return f'<a href="#{anchor}" class="entity-link">{matched}</a>'

    return re.sub(pattern, repl, esc)


def export_html(results_data: dict, output_path: Path, title: str = "Knowledge Graph"):
    """Export results.json data to an HTML report."""
    title = results_data.get("title") or title
    summary = results_data.get("summary", "")
    url = results_data.get("url", "")
    communities = results_data.get("communities", [])
    chunks = results_data.get("chunks", [])
    consolidated = results_data.get("consolidated", {})
    entities = consolidated.get("entities", [])
    relations = consolidated.get("relations", [])

    entity_names = {e["name"] for e in entities}
    # Build relation lookup: entity -> list of (other, type, description, direction)
    rel_map: dict[str, list[dict]] = {}
    for r in relations:
        src, tgt = r.get("source", ""), r.get("target", "")
        rtype = r.get("type", "")
        desc = r.get("description", "")
        rel_map.setdefault(src, []).append(
            {"other": tgt, "type": rtype, "description": desc, "dir": "out"}
        )
        rel_map.setdefault(tgt, []).append(
            {"other": src, "type": rtype, "description": desc, "dir": "in"}
        )

    node_elements = []
    for e in entities:
        node_elements.append(
            {
                "data": {
                    "id": e.get("name", ""),
                    "label": e.get("name", ""),
                    "type": e.get("type", ""),
                    "description": e.get("description", ""),
                    "chunk_ids": e.get("chunk_ids", []),
                    "community_id": e.get("community_id"),
                }
            }
        )

    edge_elements = []
    for idx, r in enumerate(relations):
        src = r.get("source", "")
        tgt = r.get("target", "")
        if src not in entity_names or tgt not in entity_names:
            continue
        edge_elements.append(
            {
                "data": {
                    "id": f"e{idx}",
                    "source": src,
                    "target": tgt,
                    "type": r.get("type", ""),
                    "description": r.get("description", ""),
                }
            }
        )

    # --- Build HTML ---
    parts: list[str] = []
    # Get the current language code for the HTML lang attribute
    html_lang = _CURRENT_LANG or "en"
    
    parts.append(
        f"""<!DOCTYPE html>
<html lang="{html_lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_mod.escape(title)}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #f5f5f5;
  --card: #bbc7c9;
  --card-text: #303030;
  --border: #26c6da;
  --link: #3c506b;
  --entity-bg: #fff;
  --entity-border: #96b9bd;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: #333; line-height: 1.6; max-width: 820px; margin: 0 auto; padding: 2rem 1rem; }}
h1 {{ font-size: 1.8rem; margin-bottom: 1.5rem; }}
h2 {{ font-size: 1.3rem; margin: 2rem 0 1rem; border-bottom: 2px solid #ccc; padding-bottom: .3rem; }}
h3 {{ font-size: 1.1rem; margin-bottom: .3rem; }}
.card {{ background: var(--card); color: var(--card-text); border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: .8rem; }}
.card .tags {{ font-weight: 900; margin-bottom: .4rem; }}
.card ul {{ margin: .4rem 0 0 1.2rem; }}
.card li {{ margin-bottom: .2rem; }}
.card a {{ color: var(--link); text-decoration: none;  }}
.card a:hover {{ color:white; background: var(--link); padding: 0.2rem; border-radius: 2px; }}
.toggle {{ text-align: right; font-size: .85rem; color: #555; cursor: pointer; user-select: none; margin-bottom: .2rem; }}
.toggle:hover {{ color: var(--link); }}
.chunk-original {{ display: none; background: #e0e0e0; border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: .8rem; white-space: pre-wrap; font-size: .9rem; }}
a.entity-link {{ color: var(--link); text-decoration: none; border-bottom: 1px dashed var(--link); }}
a.entity-link:hover {{ color: white; border-bottom-style: solid; }}
.entity-card {{ background: var(--entity-bg); border: 1px solid var(--entity-border); border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: .8rem; }}
.entity-card h3 {{ color: #00796b; }}
.entity-card .desc {{ color: #555; margin: .3rem 0 .6rem; }}
.entity-card ul {{ margin: 0 0 .6rem 1.2rem; }}
.entity-card li {{ margin-bottom: .2rem; }}
.chunk-links {{ font-size: .85rem; color: #777; }}
.chunk-links a {{ color: var(--link); text-decoration: none; }}
.chunk-links a:hover {{ text-decoration: underline; }}
.footer {{font-size: .8rem; color: #696969; margin-top: 2rem; border-top: 1px solid #ddd; padding-top: 1rem; text-align: center; text-decoration: none; }}
.footer a {{ color: var(--link); text-decoration: none; }}
.footer a:hover {{ text-decoration: underline; }}
.detail{{font-size: .9rem; color: #555; margin-top: 1rem;}}
.disclaimer{{font-size: .6rem; color: #696969; margin-top: 1rem; border-top: 1px solid #ddd; padding-top: 1rem; text-align: left; border-bottom: 1px solid #ddd; padding-bottom: 1rem;}}
#graph {{ height: 420px; background: #fff; border: 1px solid #cfd8dc; border-radius: 8px; margin-bottom: 1rem; }}
#node-info {{ background: #fff; border: 1px solid #cfd8dc; border-radius: 8px; padding: .8rem 1rem; margin-bottom: 1rem; display: none; }}
#node-info a{{ color: var(--link); text-decoration: none; }}
#node-info a:hover {{ color:white; background: var(--link); padding: 0.2rem; border-radius: 2px; }}
#node-info h4 {{ margin-bottom: .4rem; }}
.node-info-name{{ font-weight: 900; color:var(--link); }}
</style>
</head>
<body>
"""
    )

    # Title + summary
    parts.append(f"<h1>{html_mod.escape(title)}</h1>\n")

    # Summary
    parts.append("<div class='card'>\n")
    if summary:
        parts.append(f"<p>{html_mod.escape(summary)}</p>\n")
    date_info = datetime.now().strftime("%b %d, %Y")

    extracted_info = get_ui("extracted_info", entities=len(entities), relations=len(relations), communities=len(communities), date=date_info)
    if not extracted_info:
        extracted_info = f"Extracted {len(entities)} entities, {len(relations)} relations and {len(communities)} topics on {date_info}"
    parts.append(f"<div class='detail'>{html_mod.escape(extracted_info)}</div>\n")
    
    if url:
        safe_url = html_mod.escape(url)
        source_label = get_ui("source_label") or "Source"
        parts.append(
            f'<div class="detail">{source_label}: <a href="{safe_url}" target="_blank" rel="noopener noreferrer">{safe_url}</a></div>\n'
        )
    parts.append("</div>\n")
    parts.append("<div class='card'>\n")

    nav_chunks = get_ui("nav_chunks") or "Text Chunks"
    nav_topics = get_ui("nav_topics") or "Topics"
    nav_entities = get_ui("nav_entities") or "Entities"
    parts.append(f"‣<a href='#chunks'>{html_mod.escape(nav_chunks)}</a>\n")
    parts.append(f"‣<a href='#topics'>{html_mod.escape(nav_topics)}</a>\n")
    parts.append(f"‣<a href='#entities'>{html_mod.escape(nav_entities)}</a>\n")
    parts.append("</div>\n")

    # --- Graph ---
    network_title = get_ui("network_title") or "Network"
    parts.append(f"<h2 id='graph-title'>{html_mod.escape(network_title)}</h2>\n")
    parts.append("<div id='graph'></div>\n")
    parts.append("<div id='node-info'></div>\n")
    # --- Topics (communities) ---
    topics_title = get_ui("topics_title") or "Topics"
    parts.append(f"<h2 id='topics'>{html_mod.escape(topics_title)}</h2>\n")
    for c in communities:
        topics = c.get("topics", [])
        desc = c.get("description", "")
        members = c.get("members", [])
        tags_str = ", ".join(html_mod.escape(t) for t in topics)
        parts.append('<div class="card">')
        parts.append(f'<div class="tags">{tags_str}</div>')
        if desc:
            parts.append(f"<ul><li>{html_mod.escape(desc)}</li>")
        else:
            parts.append("<ul>")
        # Entity members as links
        member_links = []
        for m in sorted(members):
            anchor = f"entity-{m.lower().replace(' ', '-')}"
            member_links.append(
                f'<a href="#{anchor}" class="entity-link">{html_mod.escape(m)}</a>'
            )
        if member_links:
            parts.append(f"<li>{', '.join(member_links)}</li>")
        parts.append("</ul></div>\n")

    # --- Content (chunks) ---
    content_title = get_ui("content_title") or "Content"
    show_original = get_ui("show_original") or "Show original text"
    show_rephrased = get_ui("show_rephrased") or "Show rephrased text"
    parts.append(f"<h2 id='chunks'>{html_mod.escape(content_title)}</h2>\n")
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk-{i}"
        rephrase = chunk.get("rephrase", "")
        original = chunk.get("text", "")
        display = rephrase if rephrase else original
        linkified = _linkify_entities(display, entity_names)

        parts.append(
            f'<div class="toggle" onclick="toggleChunk(\'{chunk_id}\')">{html_mod.escape(show_original)}</div>'
        )
        parts.append(f'<div class="card" id="{chunk_id}-rephrase">{linkified}</div>')
        parts.append(
            f'<div class="chunk-original" id="{chunk_id}-original">{html_mod.escape(original)}</div>'
        )

    # --- Entities ---
    entities_title = get_ui("entities_title") or "Entities"
    appears_in_label = get_ui("appears_in") or "Appears in"
    chunk_label = get_ui("chunk_label") or "Chunk"
    parts.append(f"<h2 id='entities'>{html_mod.escape(entities_title)}</h2>\n")
    for e in sorted(entities, key=lambda x: x.get("name", "").lower()):
        name = e.get("name", "")
        anchor = f"entity-{name.lower().replace(' ', '-')}"
        desc = e.get("description", "")
        chunk_ids = e.get("chunk_ids", [])
        rels = rel_map.get(name, [])

        parts.append(f'<div class="entity-card" id="{anchor}">')
        parts.append(f"<h3>{html_mod.escape(name)}</h3>")
        if desc:
            parts.append(f'<div class="desc">{html_mod.escape(desc)}</div>')
        if rels:
            parts.append("<ul>")
            for rel in rels:
                other = rel["other"]
                other_anchor = f"entity-{other.lower().replace(' ', '-')}"
                rel_desc = rel["description"]
                arrow = "\u2192" if rel["dir"] == "out" else "\u2190"
                label = f'{arrow} <a href="#{other_anchor}" class="entity-link">{html_mod.escape(other)}</a>'
                if rel_desc:
                    label += f": {html_mod.escape(rel_desc)}"
                parts.append(f"<li>{label}</li>")
            parts.append("</ul>")
        if chunk_ids:
            links = [
                f'<a href="#chunk-{cid}-rephrase">{chunk_label} {cid}</a>'
                for cid in chunk_ids
                if cid >= 0
            ]
            if links:
                parts.append(
                    f'<div class="chunk-links">{appears_in_label}: {" \u2022 ".join(links)}</div>'
                )
        parts.append("</div>\n")

    # --- Origin and disclaimer ---
    disclaimer = get_ui("disclaimer") or "The information contained in this document has been extracted and processed using automated artificial intelligence tools (Knwl.AI). While we strive for accuracy, this data is generated via machine learning algorithms and natural language processing, which may result in errors, omissions, or misinterpretations of the original source material. This document is provided \"as is\" and for informational purposes only. Orbifold Consulting makes no warranties, express or implied, regarding the accuracy, completeness, or reliability of this information. Users are advised to independently verify any critical data against original source documents before making business, legal, or financial decisions. Orbifold Consulting assumes no liability for any actions taken in reliance upon this information."
    parts.append(
        f'''<div class="footer">
<div>Generated by <a title="Knwl AI" target="_blank" href="https://knwl.ai">Knwl</a>, &copy; 2026 <a href="https://graphsandnetworks.com" title="Orbifold Consulting">Orbifold Consulting</a>.</div>
<div class='disclaimer'>{html_mod.escape(disclaimer)}</div>
</div>'''   
    )

    # Get localized UI strings for JavaScript
    js_type_label = get_ui("type_label") or "Type"
    js_desc_label = get_ui("description_label") or "Description"
    js_community_label = get_ui("community_label") or "Community"
    js_chunks_label = get_ui("chunks_label") or "Chunks"
    js_chunk_label = get_ui("chunk_label") or "Chunk"
    js_show_original = get_ui("show_original") or "Show original text"
    js_show_rephrased = get_ui("show_rephrased") or "Show rephrased text"

    # --- Script ---
    parts.append(
                f"""<script src=\"https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js\"></script>
        <script src=\"https://unpkg.com/layout-base@2.0.1/layout-base.js\"></script>
        <script src=\"https://unpkg.com/cose-base@2.2.0/cose-base.js\"></script>
        <script src=\"https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js\"></script>
<script>
        if (window.cytoscape && window.cytoscapeFcose) {{
            cytoscape.use(cytoscapeFcose);
        }}

// Localized labels
const labels = {{
    type: {json.dumps(js_type_label)},
    description: {json.dumps(js_desc_label)},
    community: {json.dumps(js_community_label)},
    chunks: {json.dumps(js_chunks_label)},
    chunk: {json.dumps(js_chunk_label)},
    showOriginal: {json.dumps(js_show_original)},
    showRephrased: {json.dumps(js_show_rephrased)}
}};

const graphElements = {{
    nodes: {json.dumps(node_elements)},
    edges: {json.dumps(edge_elements)}
}};

const communityDesc = {json.dumps({str(c.get("id")): c.get("description", "") for c in communities})};

const colors = [
    '#6baed6','#9ecae1','#c6dbef','#74c476','#a1d99b',
    '#c7e9c0','#fd8d3c','#fdae6b','#fdd0a2','#9e9ac8'
];

function colorForCommunity(cid) {{
    if (cid === null || cid === undefined || cid === '') return '#bdbdbd';
    return colors[Math.abs(parseInt(cid, 10)) % colors.length];
}}

const cy = cytoscape({{
    container: document.getElementById('graph'),
    elements: graphElements,
    layout: {{ name: 'fcose', animate: true, randomize: true, nodeRepulsion: 4500 }},
    style: [
        {{ selector: 'node', style: {{
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': 8,
                'background-color': (ele) => colorForCommunity(ele.data('community_id')),
                'width': 18,
                'height': 18,
                'color': '#263238'
        }} }},
        {{ selector: 'edge', style: {{
                'width': 1,
                'line-color': '#90a4ae',
                'target-arrow-color': '#90a4ae',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier'
        }} }}
    ]
}});

const info = document.getElementById('node-info');
cy.on('tap', 'node', (evt) => {{
    const d = evt.target.data();
    const chunks = (d.chunk_ids || [])
        .map((cid) => `<a href="#chunk-${{cid}}-rephrase">${{labels.chunk}} ${{cid}}</a>`)
        .join(', ');
    const cdesc = communityDesc[String(d.community_id)] || '';
    info.style.display = 'block';
    info.innerHTML = `
        <div class='node-info-name'>${{d.label.toUpperCase() || ''}}</div>
        <div><strong>${{labels.type}}:</strong> ${{d.type || ''}}</div>
        <div><strong>${{labels.description}}:</strong> ${{d.description || ''}}</div>
        <div><strong>${{labels.community}}:</strong> ${{cdesc}}</div>
        <div><strong>${{labels.chunks}}:</strong> ${{chunks}}</div>
    `;
}});
cy.on('tap', (evt) => {{
    if (evt.target === cy) {{
        info.style.display = 'none';
    }}
}});

function toggleChunk(id) {{
    const reph = document.getElementById(id + '-rephrase');
    const orig = document.getElementById(id + '-original');
    const toggle = reph.previousElementSibling;
    if (orig.style.display === 'block') {{
        orig.style.display = 'none';
        reph.style.display = 'block';
        toggle.textContent = labels.showOriginal;
    }} else {{
        orig.style.display = 'block';
        reph.style.display = 'none';
        toggle.textContent = labels.showRephrased;
    }}
}}
</script>
</body>
</html>"""
        )
    html_path = output_path.with_suffix(".html")
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def create_network(consolidated: dict):
    """Create a network visualization from consolidated data."""
    g = nx.MultiDiGraph()
    for e in consolidated["entities"]:
        g.add_node(
            e["name"],
            type=e["type"],
            description=e["description"],
            community_id=e.get("community_id"),
        )
    for r in consolidated["relations"]:
        g.add_edge(
            r["source"], r["target"], type=r["type"], description=r["description"]
        )
    return g


def _build_community_prompt(communities: list[dict]) -> str:
    """Build prompt for community labeling."""
    lines = []
    for c in communities:
        members = ", ".join(
            f'{{"name": "{m["name"]}", "type": "{m["type"]}", "description": "{m["description"]}"}}'
            for m in c["members"]
        )
        lines.append(f'- "{c["id"]}": [{members}]')
    communities_text = chr(10).join(lines)
    
    # Use localized prompt
    prompt = get_prompt("community_labeling", communities_text=communities_text)
    
    # Fallback to English prompt if localization fails
    if not prompt:
        prompt = f"""You are labeling graph communities. For each community, return 1-3 short topics and a 1-2 sentence description.

COMMUNITIES:
{communities_text}

Return JSON:
{{
  "communities": {{
    "<id>": {{"topics": ["topic1", "topic2"], "description": "..."}},
    ...
  }}
}}"""
    return prompt


def _fallback_community_labels(communities: list[dict]) -> dict:
    """Fallback labels using most common entity types."""
    labels = {}
    for c in communities:
        type_counts: dict[str, int] = {}
        for m in c["members"]:
            etype = (m.get("type") or "").strip().lower()
            if not etype:
                continue
            type_counts[etype] = type_counts.get(etype, 0) + 1
        topics = [
            t.replace("_", " ").capitalize()
            for t, _ in sorted(type_counts.items(), key=lambda x: -x[1])
        ][:3]
        if not topics:
            topics = ["misc"]

        labels[c["id"]] = {
            "topics": topics,
            "description": f"Around {', '.join(topics)}.",
        }
    return labels


def analyze_communities(consolidated: dict, config: Config) -> dict:
    """Detect communities and label them with topics and descriptions."""
    analyzing_msg = get_console_msg("analyzing_communities") or "Analyzing communities..."
    with console.status(f"[cyan]{analyzing_msg}"):
        g = nx.Graph()
        for e in consolidated.get("entities", []):
            g.add_node(e["name"])
        for r in consolidated.get("relations", []):
            weight = r.get("strength", 1)
            g.add_edge(r["source"], r["target"], weight=weight)

        if g.number_of_nodes() == 0:
            consolidated["communities"] = []
            return consolidated

        communities = louvain_communities(g, weight="weight", seed=0)
        entity_map = {e["name"]: e for e in consolidated.get("entities", [])}

        community_payload = []
        for cid, members in enumerate(communities):
            member_objs = [
                {
                    "name": name,
                    "type": entity_map.get(name, {}).get("type", ""),
                    "description": entity_map.get(name, {}).get("description", ""),
                }
                for name in sorted(members)
            ]
            community_payload.append({"id": str(cid), "members": member_objs})

        labels = {}
        if community_payload:
            prompt = _build_community_prompt(community_payload)
            response = llm_generate(prompt, config, model=config.extraction_model)
            parsed = parse_json_response(response)
            labels = parsed.get("communities", {})

        if not labels:
            labels = _fallback_community_labels(community_payload)

        communities_out = []
        for cid, members in enumerate(communities):
            label = labels.get(str(cid), {"topics": ["misc"], "description": ""})
            communities_out.append(
                {
                    "id": cid,
                    "topics": label.get("topics", ["misc"]),
                    "description": label.get("description", ""),
                    "members": sorted(members),
                }
            )
            for name in members:
                if name in entity_map:
                    entity_map[name]["community_id"] = cid

        consolidated["communities"] = communities_out
        detected_msg = get_console_msg("detected_communities", count=len(communities)) or f"Detected {len(communities)} communities"
        console.print(f"[white]{detected_msg}[/]")
    return consolidated


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_stats(
    results: list[ExtractionResult],
    schema_time: float,
    wall_time: float,
    consolidation_time: float = 0.0,
) -> dict:
    """Compute extraction statistics."""
    times = [r.chunk_time for r in results]
    tokens = [r.chunk_tokens for r in results]
    entities = [r.entities_count for r in results]
    relations = [r.relations_count for r in results]

    total_time = sum(times)
    total_tokens = sum(tokens)

    return {
        "num_chunks": len(results),
        "schema_discovery_time": round(schema_time, 2),
        "extraction_wall_time": round(wall_time, 2),
        "consolidation_time": round(consolidation_time, 2),
        "total_cpu_time": round(total_time, 2),
        "total_time": round(schema_time + wall_time + consolidation_time, 2),
        "parallelism": round(total_time / wall_time, 2) if wall_time > 0 else 0,
        "throughput_tps": round(total_tokens / wall_time, 1) if wall_time > 0 else 0,
        "time": {
            "min": round(min(times), 2),
            "max": round(max(times), 2),
            "avg": round(mean(times), 2),
            "median": round(median(times), 2),
        },
        "tokens": {
            "min": min(tokens),
            "max": max(tokens),
            "avg": round(mean(tokens)),
            "total": total_tokens,
        },
        "entities": {"total": sum(entities), "avg": round(mean(entities), 1)},
        "relations": {"total": sum(relations), "avg": round(mean(relations), 1)},
    }


def compute_community_stats(consolidated: dict) -> dict:
    """Compute community detection statistics."""
    communities = consolidated.get("communities", [])
    sizes = [len(c.get("members", [])) for c in communities]

    if not sizes:
        return {
            "count": 0,
            "singletons": 0,
            "largest": 0,
            "size": {"min": 0, "max": 0, "avg": 0.0, "median": 0.0},
        }

    return {
        "count": len(sizes),
        "singletons": sum(1 for s in sizes if s == 1),
        "largest": max(sizes),
        "size": {
            "min": min(sizes),
            "max": max(sizes),
            "avg": round(mean(sizes), 1),
            "median": round(median(sizes), 1),
        },
    }


def print_stats(stats: dict, schema: Schema, consolidated: dict | None = None):
    """Print formatted statistics using rich tables."""
    console.print()

    # Main stats table
    table = Table(
        title="Extraction Results", show_header=True, header_style="bold cyan"
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row(
        "Schema",
        f"{len(schema.entity_types)} entity types, {len(schema.relation_types)} relation types",
    )
    table.add_row("Discovery time", f"{stats['schema_discovery_time']:.2f}s")
    table.add_row("Chunks", str(stats["num_chunks"]))
    table.add_row("Extraction time", f"{stats['extraction_wall_time']:.2f}s")
    table.add_row("Consolidation time", f"{stats['consolidation_time']:.2f}s")
    if "community_detection_time" in stats:
        table.add_row("Community time", f"{stats['community_detection_time']:.2f}s")
    table.add_row("CPU time", f"{stats['total_cpu_time']:.2f}s")
    table.add_row("Parallelism", f"[green]{stats['parallelism']:.1f}x[/green]")

    t = stats["time"]
    table.add_row(
        "Time/chunk", f"min={t['min']:.2f}s, max={t['max']:.2f}s, avg={t['avg']:.2f}s"
    )

    tk = stats["tokens"]
    table.add_row("Tokens/chunk", f"min={tk['min']}, max={tk['max']}, avg={tk['avg']}")
    table.add_row(
        "Throughput", f"[green]{stats['throughput_tps']:.1f}[/green] tokens/sec"
    )

    e, r = stats["entities"], stats["relations"]
    table.add_row("Entities", f"{e['total']} total ({e['avg']:.1f}/chunk)")
    table.add_row("Relations", f"{r['total']} total ({r['avg']:.1f}/chunk)")

    if consolidated:
        table.add_row(
            "Consolidated",
            f"[bold green]{len(consolidated['entities'])}[/bold green] entities, "
            f"[bold green]{len(consolidated['relations'])}[/bold green] relations",
        )
        if "communities" in consolidated:
            cstats = stats.get("communities", {})
            size = cstats.get("size", {})
            table.add_row(
                "Communities",
                f"{cstats.get('count', 0)} total, {cstats.get('singletons', 0)} singletons",
            )
            table.add_row(
                "Community size",
                f"min={size.get('min', 0)}, max={size.get('max', 0)}, "
                f"avg={size.get('avg', 0):.1f}, median={size.get('median', 0):.1f}",
            )

    table.add_row("", "")
    table.add_row(
        "[bold]TOTAL TIME[/bold]", f"[bold cyan]{stats['total_time']:.2f}s[/bold cyan]"
    )

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@app.command()
def main(
    file: Annotated[
        Optional[Path],
        typer.Option("--file", "-f", help="Input text file"),
    ] = None,
    extraction_model: Annotated[
        str,
        typer.Option("--extraction-model", "-e", help="Model for graph extraction"),
    ] = DEFAULT_OLLAMA_EXTRACTION_MODEL,
    discovery_model: Annotated[
        str,
        typer.Option("--discovery-model", "-d", help="Model for schema discovery"),
    ] = DEFAULT_OLLAMA_SCHEMA_MODEL,
    concurrent: Annotated[
        int,
        typer.Option("--concurrent", "-c", help="Max concurrent requests"),
    ] = 10,
    no_discovery: Annotated[
        bool,
        typer.Option(
            "--no-discovery", help="Skip schema discovery and use the default schema"
        ),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable LLM response caching"),
    ] = False,
    openai: Annotated[
        bool,
        typer.Option("--openai", help="Use OpenAI API instead of Ollama"),
    ] = False,
    openai_base_url: Annotated[
        str,
        typer.Option("--openai-base-url", help="OpenAI API base URL"),
    ] = "https://api.openai.com/v1",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Save results to JSON"),
    ] = None,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Max tokens per chunk"),
    ] = 400,
    html_report: Annotated[
        bool,
        typer.Option("--html-report", help="Also export an HTML report"),
    ] = True,
    gml_export: Annotated[
        bool,
        typer.Option("--gml-export", help="Also export a GML graph file"),
    ] = True,
    html_only: Annotated[
        bool,
        typer.Option("--html-only", help="Only export HTML from existing results.json"),
    ] = False,
    url: Annotated[
        Optional[str],
        typer.Option("--url", "-u", help="Origin of the document (for metadata only)"),
    ] = None,
    language: Annotated[
        Optional[str],
        typer.Option("--language", "-l", help="Language code (e.g., en, de, fr, es, nl). Auto-detects if not specified."),
    ] = None,
):
    """Extract knowledge graphs from text using LLMs."""
    # Determine output path - all outputs go to results/ directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if output is None:
        output = results_dir / "results.json"
    else:
        # If user provided a path, put it in results/ directory
        output = results_dir / output.name

    if html_only:
        if not output.exists():
            raise FileNotFoundError(f"Missing results file: {output}")
        results_data = json.loads(output.read_text())
        # Set language from saved data or default
        saved_lang = results_data.get("language", DEFAULT_LANGUAGE)
        set_language(saved_lang)
        html_path = export_html(results_data, output, title=output.stem)
        console.print(f"[green]✓[/green] HTML report saved to [cyan]{html_path}[/cyan]")
        return

    # Config
    config = Config(
        ollama_extraction_model=extraction_model,
        ollama_discovery_model=discovery_model,
        max_concurrent=concurrent,
        max_tokens=max_tokens,
        use_cache=not no_cache,
        use_openai=openai,
        openai_base_url=openai_base_url,
    )

    # Load text
    file_path = file or Path(DEFAULT_FILE)
    # if the file is pdf, extract text using Pymupdf, otherwise read as text
    if file_path.suffix.lower() == ".pdf":
        # if existst in the results dir, use the extracted text to avoid re-extracting from pdf
        extracted_text_path = results_dir / f"{file_path.stem}_extracted.txt"
        if extracted_text_path.exists():
            console.print(
                f"[green]✓[/green] Using cached extracted text: {extracted_text_path}"
            )
            text = extracted_text_path.read_text(encoding="utf-8", errors="ignore")
        else:
            console.print(f"[yellow]Extracting text from PDF: {file_path}[/yellow]")
            doc = fitz.open(file_path)
            text = "\n\n".join(page.get_text() for page in doc)
            extracted_text_path.write_text(text, encoding="utf-8", errors="ignore")
    else:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    # return sys.exit(0)  # for testing
    # Load existing graph if output file already exists (augment mode)
    existing_data = None
    existing_result = None
    if output and output.exists():
        existing_data = json.loads(output.read_text())
        consolidated_ents = existing_data.get("consolidated", {}).get("entities", [])
        consolidated_rels = existing_data.get("consolidated", {}).get("relations", [])
        existing_result = ExtractionResult(
            entities=consolidated_ents,
            relations=consolidated_rels,
            chunk_idx=-1,
            chunk_time=0.0,
            chunk_tokens=0,
        )

    # Language detection/setting
    if language:
        set_language(language)
        detected_lang = language
    else:
        detected_lang = detect_language(text, config)
        set_language(detected_lang)
    
    lang_name = get_lang().get("name", detected_lang)

    # Show header
    backend = "[cyan]OpenAI[/cyan]" if config.use_openai else "[green]Ollama[/green]"
    mode = (
        f"  •  [yellow]Augmenting[/yellow]: [dim]{output.name}[/dim]"
        if existing_data
        else ""
    )
    console.print(
        Panel.fit(
            f"[bold]Graph Extraction Pipeline[/bold]\n"
            f"Backend: {backend}  •  File: [dim]{file_path.name}[/dim]  •  Language: [magenta]{lang_name}[/magenta]{mode}",
            border_style="blue",
        )
    )

    # Show cache status
    if config.use_cache:
        cache_count = len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0
        console.print(f"[dim]Cache: {CACHE_DIR} ({cache_count} entries)[/dim]")
    else:
        console.print("[dim]Cache: disabled[/dim]")

    # Schema discovery
    console.print()
    console.rule("[bold cyan]Schema Discovery[/bold cyan]")
    console.print(f"Model: [green]{config.discovery_model}[/green]")

    if no_discovery:
        schema = Schema(
            entity_types=config.default_entity_types,
            relation_types=config.default_relation_types,
            reasoning="Using defaults (discovery skipped)",
        )
        console.print("[yellow]Skipped (using defaults)[/yellow]")
    else:
        discovering_msg = get_console_msg("discovering_schema") or "Discovering schema..."
        with console.status(f"[bold green]{discovering_msg}[/bold green]"):
            schema = discover_schema(text, config)
        console.print(f"Time: [cyan]{schema.discovery_time:.2f}s[/cyan]")

    # Merge existing graph's schema types into discovered schema
    if existing_data and "schema" in existing_data:
        old_schema = existing_data["schema"]
        for et in old_schema.get("entity_types", []):
            if et.lower() not in {t.lower() for t in schema.entity_types}:
                schema.entity_types.append(et)
        for rt in old_schema.get("relation_types", []):
            if rt.lower() not in {t.lower() for t in schema.relation_types}:
                schema.relation_types.append(rt)

    console.print(f"Entity types: [green]{', '.join(schema.entity_types)}[/green]")
    console.print(f"Relation types: [green]{', '.join(schema.relation_types)}[/green]")

    # Chunk
    console.print()
    console.rule("[bold cyan]Text Chunking[/bold cyan]")
    chunks = chunk_text(text, config)
    console.print(
        f"\nChunks: [cyan]{len(chunks)}[/cyan] (~{config.max_tokens} tokens each)"
    )

    # Title
    console.print()
    console.rule("[bold cyan]Title Extraction[/bold cyan]")
    title = extract_title(chunks, config) or file_path.stem
    console.print(f"Title: [green]{title}[/green]")

    # Summary
    console.print()
    console.rule("[bold cyan]Document Summary[/bold cyan]")
    summary = extract_summary(chunks, config)
    if summary:
        console.print(f"Summary: [green]{summary}[/green]")
    else:
        console.print("[yellow]Summary not available[/yellow]")

    # Chunk rephrase
    console.print()
    console.rule("[bold cyan]Chunk Rephrase[/bold cyan]")
    console.print(f"Model: [green]{config.extraction_model}[/green]")
    rephrase_t0 = time.perf_counter()
    rephrased = rephrase_chunks(chunks, config)
    rephrase_time = time.perf_counter() - rephrase_t0
    console.print(f"Time: [cyan]{rephrase_time:.2f}s[/cyan]")

    # Extract
    console.print()
    console.rule("[bold cyan]Extraction[/bold cyan]")
    console.print(f"Model: [green]{config.extraction_model}[/green]")

    t0 = time.perf_counter()
    results = asyncio.run(extract_all(chunks, schema, config, output_path=output))
    wall_time = time.perf_counter() - t0

    # Consolidate (include existing graph data if augmenting)
    console.print()
    console.rule("[bold cyan]Consolidation[/bold cyan]")
    all_results = ([existing_result] + results) if existing_result else results
    consolidated, consolidation_time = consolidate_graphs(
        all_results, config, summarize=True
    )

    # Community analysis
    console.print()
    console.rule("[bold cyan]Community Analysis[/bold cyan]")
    community_t0 = time.perf_counter()
    consolidated = analyze_communities(consolidated, config)
    community_time = time.perf_counter() - community_t0

    # Stats
    stats = compute_stats(results, schema.discovery_time, wall_time, consolidation_time)
    stats["community_detection_time"] = round(community_time, 2)
    stats["communities"] = compute_community_stats(consolidated)
    print_stats(stats, schema, consolidated)

    # Save results
    if output:
        if existing_data and "schema" in existing_data:
            old_schema = existing_data["schema"]
            entity_types = list(schema.entity_types)
            relation_types = list(schema.relation_types)

            for et in old_schema.get("entity_types", []):
                if et.lower() not in {t.lower() for t in entity_types}:
                    entity_types.append(et)
            for rt in old_schema.get("relation_types", []):
                if rt.lower() not in {t.lower() for t in relation_types}:
                    relation_types.append(rt)

            old_reasoning = old_schema.get("reasoning", "")
            if old_reasoning and old_reasoning != schema.reasoning:
                reasoning = f"{old_reasoning} | {schema.reasoning}".strip(" |")
            else:
                reasoning = schema.reasoning or old_reasoning
        else:
            entity_types = schema.entity_types
            relation_types = schema.relation_types
            reasoning = schema.reasoning

        stats_entry = dict(stats)
        stats_entry["run"] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file": str(file_path),
            "backend": "openai" if config.use_openai else "ollama",
            "extraction_model": config.extraction_model,
            "discovery_model": config.discovery_model,
            "max_tokens": config.max_tokens,
            "overlap_tokens": config.overlap_tokens,
            "max_concurrent": config.max_concurrent,
            "use_cache": config.use_cache,
            "no_discovery": no_discovery,
        }

        if existing_data and "stats" in existing_data:
            prior_stats = existing_data.get("stats")
            if isinstance(prior_stats, list):
                stats_list = prior_stats + [stats_entry]
            else:
                stats_list = [prior_stats, stats_entry]
        else:
            stats_list = [stats_entry]

        output_data = {
            "title": title,
            "summary": summary,
            "url": url,
            "language": detected_lang,
            "schema": {
                "entity_types": entity_types,
                "relation_types": relation_types,
                "reasoning": reasoning,
            },
            "stats": stats_list,
            "communities": consolidated.get("communities", []),
            "consolidated": consolidated,
            "chunks": (existing_data.get("chunks", []) if existing_data else [])
            + [
                {
                    "chunk_idx": r.chunk_idx,
                    "text": chunks[r.chunk_idx],
                    "rephrase": (
                        rephrased[r.chunk_idx] if r.chunk_idx < len(rephrased) else ""
                    ),
                    "entities": r.entities,
                    "relations": r.relations,
                    "chunk_time": r.chunk_time,
                    "chunk_tokens": r.chunk_tokens,
                    "source_file": str(file_path),
                }
                for r in results
            ],
        }
        output.write_text(json.dumps(output_data, indent=2))
        console.print(f"\n[green]✓[/green] Results saved to [cyan]{output}[/cyan]")

        # Export to graph formats
        if gml_export:
            g = create_network(consolidated)

            nx.write_gml(g, output.with_suffix(".gml"))
            console.print(
                f"[green]✓[/green] Graph saved to [cyan]{output.with_suffix('.gml')}[/cyan]"
            )

        # Export to HTML
        if html_report:
            html_path = export_html(output_data, output, title=file_path.stem)
            console.print(
                f"[green]✓[/green] HTML report saved to [cyan]{html_path}[/cyan]"
            )

        # nx.write_graphml(g, output.with_suffix(".graphml"))
        # console.print(f"[green]✓[/green] Graph saved to [cyan]{output.with_suffix('.graphml')}[/cyan]")


if __name__ == "__main__":
    app()
