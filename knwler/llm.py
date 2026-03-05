"""
LLM backends (Ollama, OpenAI) and JSON response parsing.
"""

import json
import os
import re

import requests

from knwler.config import Config
from knwler.cache import cache_key, get_cached_response, save_to_cache


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def ollama_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call Ollama and return the response text (cached)."""
    actual_model = model or config.ollama_extraction_model

    if config.use_cache:
        key = cache_key(prompt, actual_model, config.temperature, config.num_predict)

        cached = get_cached_response(key)
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

    resp = requests.post(config.ollama_url, json=payload, timeout=360)
    resp.raise_for_status()
    response = resp.json()["response"]

    if config.use_cache:
        save_to_cache(key, response, actual_model)

    return response


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
def openai_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call OpenAI API and return the response text (cached)."""
    actual_model = model or config.openai_extraction_model
    api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "OpenAI API key not set. Set OPENAI_API_KEY env var or config.openai_api_key"
        )

    if config.use_cache:
        key = cache_key(prompt, actual_model, config.temperature, config.num_predict)
        cached = get_cached_response(key)
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
    resp = requests.post(url, headers=headers, json=payload, timeout=360)
    resp.raise_for_status()
    response = resp.json()["choices"][0]["message"]["content"]

    if config.use_cache:
        save_to_cache(key, response, actual_model)

    return response


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
def anthropic_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call Anthropic API and return the response text (cached)."""
    import anthropic as _anthropic

    actual_model = model or config.anthropic_extraction_model
    api_key = config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        raise ValueError(
            "Anthropic API key not set. Set ANTHROPIC_API_KEY env var or config.anthropic_api_key"
        )

    if config.use_cache:
        key = cache_key(prompt, actual_model, config.temperature, config.num_predict)
        cached = get_cached_response(key)
        if cached is not None:
            return cached

    client = _anthropic.Anthropic(api_key=api_key)

    system = (
        "You are a data extraction assistant. "
        "Always respond with valid JSON only — no markdown, no explanation, no code blocks."
        if format_json
        else "You are a helpful assistant."
    )

    # Assistant prefill with "{" nudges the model to return raw JSON.
    messages = [{"role": "user", "content": prompt}]
    if format_json:
        messages.append({"role": "assistant", "content": "{"})

    response = client.messages.create(
        model=actual_model,
        max_tokens=config.num_predict,
        system=system,
        messages=messages,
        temperature=config.temperature,
    )
    content = response.content[0].text

    # Restore the prefilled "{" that Anthropic strips from its own response.
    if format_json:
        content = "{" + content

    if config.use_cache:
        save_to_cache(key, content, actual_model)

    return content


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def llm_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Dispatch to appropriate LLM backend based on config."""
    if config.backend == "anthropic":
        return anthropic_generate(prompt, config, model, format_json)
    if config.backend == "openai":
        return openai_generate(prompt, config, model, format_json)
    return ollama_generate(prompt, config, model, format_json)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def parse_json_response(response: str) -> dict:
    """Parse JSON from response, handling edge cases."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}
