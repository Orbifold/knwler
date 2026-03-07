"""
LLM backends (Ollama, OpenAI) and JSON response parsing.
"""

import json
import os
from typing import Any

import aiohttp

from knwler.config import Config
from knwler.cache import (
    create_llm_cache_key,
    get_cached_llm_response,
    save_llm_response_to_cache,
)


async def _post_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 360,
) -> dict[str, Any]:
    """POST JSON payload and parse JSON response using aiohttp."""
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
async def ollama_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call Ollama and return the response text (cached)."""
    actual_model = model or config.extraction_model

    if config.use_cache:
        key = create_llm_cache_key(
            prompt, actual_model, config.temperature, config.num_predict
        )

        cached = get_cached_llm_response(key)
        if cached is not None:
            return cached

    payload = {
        "model": actual_model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.num_predict,
        },
    }
    if format_json:
        payload["format"] = "json"

    response_json = await _post_json(config.ollama_url, payload=payload)
    response = response_json["response"]

    if config.use_cache:
        save_llm_response_to_cache(key, response, actual_model)

    return response


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
async def openai_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call OpenAI API and return the response text (cached)."""
    actual_model = model or config.openai_extraction_model
    api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "OpenAI API key not set. Set OPENAI_API_KEY env var or config.api_key"
        )

    if config.use_cache:
        key = create_llm_cache_key(
            prompt, actual_model, config.temperature, config.num_predict
        )
        cached = get_cached_llm_response(key)
        if cached is not None:
            return cached

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
    response_json = await _post_json(url, payload=payload, headers=headers)
    response = response_json["choices"][0]["message"]["content"]

    if config.use_cache:
        save_llm_response_to_cache(key, response, actual_model)

    return response


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
async def anthropic_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call Anthropic API and return the response text (cached)."""
    import anthropic as _anthropic

    actual_model = model or config.extraction_model
    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        raise ValueError(
            "Anthropic API key not set. Set ANTHROPIC_API_KEY env var or config.api_key"
        )

    if config.use_cache:
        key = create_llm_cache_key(
            prompt, actual_model, config.temperature, config.num_predict
        )
        cached = get_cached_llm_response(key)
        if cached is not None:
            print(cached)

            return cached

    client = _anthropic.Anthropic(api_key=api_key)

    system = (
        "You are a data extraction assistant. "
        "Always respond with valid JSON only — no markdown, no explanation, no code blocks. Do not enclose JSON with backticks."
        if format_json
        else "You are a helpful assistant."
    )

    messages = [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model=actual_model,
        max_tokens=config.num_predict,
        system=system,
        messages=messages,
        temperature=config.temperature 
    )
    content = response.content[0].text

    if config.use_cache:
        save_llm_response_to_cache(key, content, actual_model)

    return content


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
async def llm_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Dispatch to appropriate LLM backend based on config."""
    if config.backend == "anthropic":
        return await anthropic_generate(prompt, config, model, format_json)
    if config.backend == "openai":
        return await openai_generate(prompt, config, model, format_json)
    return await ollama_generate(prompt, config, model, format_json)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def parse_json_response(response: str) -> dict:
    """Parse JSON from response, handling edge cases."""
    text = response.strip()
    if text.startswith("```"):
        # Strip opening fence (```json or ```)
        text = text[text.index("\n") + 1 :] if "\n" in text else text[3:]
        # Strip closing fence
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
