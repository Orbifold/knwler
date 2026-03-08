"""
Schema discovery and language detection.
"""

import json
import time

from knwler.config import Config, console
from knwler.language import DEFAULT_LANGUAGE, get_prompt, load_languages
from knwler.llm import llm_generate, parse_json_response
from knwler.models import ExtractionResult, Schema, Graph


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
async def detect_language(text: str, config: Config, sample_size: int = 1000) -> str:
    """Detect the language of the input text using LLM."""
    sample = text[:sample_size] if len(text) > sample_size else text

    prompt = (
        "Detect the language of the following text. "
        "Return only a JSON object with the ISO 639-1 language code "
        '(e.g., "en" for English, "de" for German, "fr" for French, etc.).\n\n'
        f'TEXT:\n"""{sample}"""\n\n'
        "Return JSON:\n"
        '{\n  "language": "en"\n}'
    )

    response = await llm_generate(prompt, config, model=config.discovery_model)
    result = parse_json_response(response)
    detected = result.get("language", DEFAULT_LANGUAGE).lower().strip()

    langs = load_languages()
    if detected not in langs:
        return DEFAULT_LANGUAGE

    return detected


# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------
async def discover_schema(
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
        prompt = (
            "Analyze this text and identify the best entity types and relation types "
            "for a knowledge graph.\n\n"
            f'TEXT SAMPLE:\n"""{sample}"""\n\n'
            "Guidelines:\n"
            "- Use snake_case for all type names\n"
            '- Be specific but not too narrow (e.g., "person" not "male_scientist")\n'
            "- Focus on types that appear multiple times or are central to the text\n"
            "- Entity types: what kinds of things are mentioned?\n"
            "- Relation types: what relationships exist between them?\n\n"
            "Return JSON:\n"
            "{\n"
            '  "entity_types": ["type1", "type2", ...],\n'
            '  "relation_types": ["rel1", "rel2", ...],\n'
            '  "reasoning": "Brief explanation"\n'
            "}\n\n"
            f"Max {max_entity_types} entity types and {max_relation_types} relation types."
        )

    t0 = time.perf_counter()
    response = await llm_generate(prompt, config, model=config.discovery_model)
    elapsed = time.perf_counter() - t0

    result = parse_json_response(response)

    if not result.get("entity_types"):
        return Schema(
            entity_types=Schema.default().entity_types,
            relation_types=Schema.default().relation_types,
            reasoning="Discovery failed, using defaults",
            discovery_time=elapsed,
        )

    return Schema(
        entity_types=result.get("entity_types", [])[:max_entity_types],
        relation_types=result.get("relation_types", [])[:max_relation_types],
        reasoning=result.get("reasoning", ""),
        discovery_time=elapsed,
    )
