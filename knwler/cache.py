"""
Generic file caching for knwler.
Cache types (webpage, wikipedia, llm, documents) are stored in separate subdirectories
under ~/.knwler/cache/.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path.home() / ".knwler" / "cache"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _subdir(name: str) -> Path:
    path = CACHE_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_cache(subdir: str, key: str) -> Any | None:
    cache_file = CACHE_DIR / subdir / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text()).get("response")
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _write_cache(subdir: str, key: str, response: Any, extra: dict | None = None):
    data = {"response": response, "cached_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        data.update(extra)
    (_subdir(subdir) / f"{key}.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _clear_cache(subdir: str):
    cache_dir = CACHE_DIR / subdir
    if cache_dir.exists():
        for f in cache_dir.glob("*.*"):
            try:
                f.unlink()
            except IOError:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hash_args(*args) -> str:
    """SHA-256 hash of the given arguments."""
    return hashlib.sha256(str(args).encode()).hexdigest()


def hash_with_prefix(content: Any, prefix: str = "") -> str:
    """SHA-256 hash of content with an optional prefix."""
    if isinstance(content, dict):
        content = json.dumps(content, sort_keys=True)
    elif hasattr(content, "model_dump_json"):
        content = content.model_dump_json()
    else:
        content = str(content)
    return prefix + hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# LLM cache
# ---------------------------------------------------------------------------


def create_llm_cache_key(
    prompt: str, model: str, temperature: float, num_predict: int
) -> str:
    """Generate a cache key from prompt and model parameters."""
    return hashlib.sha256(
        f"{model}|{temperature}|{num_predict}|{prompt}".encode()
    ).hexdigest()


def get_cached_llm_response(key: str) -> str | None:
    """Retrieve a cached LLM response."""
    return _read_cache("llm", key)


def save_llm_response_to_cache(
    key: str, response: str, model: str, id: str | None = None
):
    """Save an LLM response to cache."""
    extra = {"model": model}
    if id is not None:
        extra["id"] = id
    _write_cache("llm", key, response, extra)


def find_items_by_model(model: str = None) -> list[dict]:
    """Find all cached LLM items, optionally filtered by model."""
    llm_dir = CACHE_DIR / "llm"
    if not llm_dir.exists():
        return []
    items = []
    for f in llm_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if model is None or data.get("model") == model:
                items.append(
                    {
                        "key": f.stem,
                        "model": data.get("model"),
                        "cached_at": data.get("cached_at"),
                    }
                )
        except (json.JSONDecodeError, IOError):
            continue
    return items


def find_item_by_id(id: str) -> dict | None:
    """Find a cached LLM item by its ID."""
    llm_dir = CACHE_DIR / "llm"
    if not llm_dir.exists():
        return None
    for f in llm_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if data.get("id") == id:
                return {
                    "key": f.stem,
                    "model": data.get("model"),
                    "cached_at": data.get("cached_at"),
                    "response": data.get("response"),
                }
        except (json.JSONDecodeError, IOError):
            continue
    return None


def clear_llm_cache():
    """Clear all LLM cache files."""
    _clear_cache("llm")


# ---------------------------------------------------------------------------
# Webpage cache
# ---------------------------------------------------------------------------


def cache_webpage(url: str, response: str):
    """Save a webpage response to cache."""
    _write_cache("webpage", hash_args(url), response)


def get_cached_webpage(url: str) -> str | None:
    """Retrieve a cached webpage response."""
    return _read_cache("webpage", hash_args(url))


def clear_webpage_cache():
    """Clear all webpage cache files."""
    _clear_cache("webpage")


# ---------------------------------------------------------------------------
# Wikipedia cache
# ---------------------------------------------------------------------------


def cache_wikipedia_response(title: str, response: dict):
    """Save a Wikipedia response to cache."""
    _write_cache("wikipedia", hash_args(title), response)


def get_cached_wikipedia_response(title: str) -> dict | None:
    """Retrieve a cached Wikipedia response."""
    return _read_cache("wikipedia", hash_args(title))


def clear_wikipedia_cache():
    """Clear all Wikipedia cache files."""
    _clear_cache("wikipedia")


# ---------------------------------------------------------------------------
# Document cache (binary + metadata)
# ---------------------------------------------------------------------------


def cache_document(url: str, content: bytearray, metadata: dict):
    """Save a document (binary) and its metadata to cache."""
    key = hash_args(url)
    file_path = _subdir("documents") / f"{key}{url[-4:]}"
    file_path.write_bytes(content)
    metadata = {
        **metadata,
        "file_path": str(file_path),
        "extension": url.split(".")[-1],
    }
    _write_cache("documents", key, metadata)
    return metadata


def get_cached_document(url: str) -> tuple[bytes, dict] | tuple[None, None]:
    """Retrieve a cached document (binary + metadata)."""
    key = hash_args(url)
    meta = _read_cache("documents", key)
    if meta is None:
        return None, None
    file_path = CACHE_DIR / "documents" / f"{key}.{meta.get('extension')}"
    if not file_path.exists():
        return None, None
    try:
        return file_path.read_bytes(), meta
    except IOError:
        return None, None


def clear_document_cache():
    """Clear all document cache files."""
    _clear_cache("documents")


# ---------------------------------------------------------------------------
# Clear all
# ---------------------------------------------------------------------------


def clear_all_cache():
    """Clear all cache files."""
    clear_llm_cache()
    clear_webpage_cache()
    clear_wikipedia_cache()
    clear_document_cache()
