"""
LLM backends (Ollama, OpenAI) and JSON response parsing.
"""

import json
import os
from typing import Any

import aiohttp
import asyncio
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
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
    except aiohttp.ClientResponseError as e:
        if e.status == 401:
            raise RuntimeError(
                "Authentication failed (HTTP 401): the API key is missing or invalid. "
                "Check your api_key config or the relevant environment variable."
            ) from e
        raise RuntimeError(
            f"LLM API request failed with status {e.status}: {e.message}"
        ) from e
    except aiohttp.ClientError as e:

        raise RuntimeError(f"LLM API request failed: {str(e)}") from e
    except asyncio.TimeoutError:
        raise RuntimeError(f"LLM API request timed out after {timeout_seconds} seconds")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
async def check_ollama_available(base_url: str) -> None:
    """Raise a clear RuntimeError if the Ollama server cannot be reached."""
    # Derive the root URL from whatever endpoint URL is configured
    # e.g. http://localhost:11434/api/generate → http://localhost:11434/
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    root_url = f"{parsed.scheme}://{parsed.netloc}/"
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(root_url) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"Ollama server at {root_url} returned HTTP {resp.status}. "
                        "Make sure Ollama is running."
                    )
    except aiohttp.ClientConnectorError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {root_url}. "
            "Please start Ollama (e.g. `ollama serve`) and try again."
        )
    except TimeoutError:
        raise RuntimeError(
            f"Timed out connecting to Ollama at {root_url}. "
            "Make sure Ollama is running and reachable."
        )


_ollama_checked: set[str] = set()


async def ollama_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call Ollama and return the response text (cached)."""
    if config.base_url not in _ollama_checked:
        await check_ollama_available(config.base_url)
        _ollama_checked.add(config.base_url)
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
    try:
        response_json = await _post_json(config.base_url, payload=payload)
        response = response_json["response"]

        if config.use_cache:
            save_llm_response_to_cache(key, response, actual_model)
    except Exception as e:
        response = f"Error calling Ollama: {str(e)}"

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
    actual_model = model or config.extraction_model
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

    url = f"{config.base_url.rstrip('/')}/chat/completions"
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
        temperature=config.temperature,
    )
    content = response.content[0].text

    if config.use_cache:
        save_llm_response_to_cache(key, content, actual_model)

    return content


# ---------------------------------------------------------------------------
# GitHub Models
# ---------------------------------------------------------------------------
async def github_generate(
    prompt: str,
    config: Config,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """Call the GitHub Models API (OpenAI-compatible) and return the response text (cached).

    Requires a ``GITHUB_TOKEN`` environment variable (or ``config.api_key``).
    See https://docs.github.com/en/github-models for available models.
    """
    actual_model = model or config.extraction_model
    api_key = config.api_key or os.environ.get("GITHUB_TOKEN", "")

    if not api_key:
        raise ValueError(
            "GitHub token not set. Set GITHUB_TOKEN env var or config.api_key"
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

    url = f"{config.base_url.rstrip('/')}/chat/completions"
    response_json = await _post_json(url, payload=payload, headers=headers)
    response = response_json["choices"][0]["message"]["content"]

    if config.use_cache:
        save_llm_response_to_cache(key, response, actual_model)

    return response


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
    if config.backend == "github":
        return await github_generate(prompt, config, model, format_json)
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
