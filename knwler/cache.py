"""
LLM response caching.
"""

import hashlib
import json
import time
from pathlib import Path

from knwler.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------
CACHE_DIR = PROJECT_ROOT / "cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
