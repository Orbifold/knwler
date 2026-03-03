"""
LLM response caching.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from knwler.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------
CACHE_DIR = PROJECT_ROOT / "cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hash_args(*args):
    """
    Computes an MD5 hash for the given arguments.

    Args:
        *args: Variable length argument list.

    Returns:
        str: The MD5 hash of the arguments as a hexadecimal string.
    """
    return hashlib.sha256(str(args).encode()).hexdigest()


def hash_with_prefix(content: Any, prefix: str = ""):
    """
    Computes an MD5 hash of the given content and returns it as a string with an optional prefix.

    Args:
        content (str): The content to hash.
        prefix (str, optional): A string to prepend to the hash. Defaults to an empty string.

    Returns:
        str: The MD5 hash of the content, optionally prefixed.
    """
    if isinstance(content, dict):
        content = json.dumps(content, sort_keys=True)
    elif hasattr(content, "model_dump_json"):
        content = content.model_dump_json()
    else:
        content = str(content)
    return prefix + hashlib.sha256(content.encode()).hexdigest()


def cache_key(prompt: str, model: str, temperature: float, num_predict: int) -> str:
    """Generate a cache key from prompt and model parameters."""
    content = f"{model}|{temperature}|{num_predict}|{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached_response(key: str) -> str | None:
    """Retrieve cached response if it exists."""
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            return data.get("response")
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_to_cache(key: str, response: str, model: str):
    """Save response to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    data = {
        "model": model,
        "response": response,
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cache_file.write_text(json.dumps(data, indent=2))

def find_cache_items(model: str = None) -> list[dict]:
    """Find all cache items, optionally filtered by model."""
    items = []
    if not CACHE_DIR.exists():
        return items
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(cache_file.read_text())
            if model is None or data.get("model") == model:
                items.append({
                    "key": cache_file.stem,
                    "model": data.get("model"),
                    "cached_at": data.get("cached_at"),
                })
        except (json.JSONDecodeError, IOError):
            continue
    return items