# Knwler API

Knwler can be used as a Python library inside your own project. The pipeline is fully async and built around a central `Config` object that controls every step.

---

## Installation

```bash
uv add knwler
```

See the [setup document](./setup.md) for other options.

---

## Quick start

The following example mirrors the canonical end-to-end flow and is a good copy-paste starting point.

```python
import asyncio
from knwler.chunking import chunk_text
from knwler.config import Config
from knwler.discovery import detect_language, discover_schema
from knwler.extras import rephrase_chunks, extract_title, extract_summary
from knwler.extraction import extract_chunk, extract_all
from knwler.consolidation import consolidate_extracted_graphs

async def main():
    text = open("my_document.md").read()

    # 1. Configure the pipeline
    config = Config(max_tokens=200, overlap_tokens=20)

    # 2. Split into chunks
    chunks = chunk_text(text, config)

    # 3. Discover schema and language
    lang   = await detect_language(text, config)
    schema = await discover_schema(text, config)

    # 4. Extract metadata
    title   = await extract_title(chunks, config)
    summary = await extract_summary(chunks, config)

    # 5. Extract a knowledge graph from each chunk
    graphs = await extract_all(chunks, schema, config)

    # 6. Consolidate into one graph
    consolidated, elapsed = await consolidate_extracted_graphs(graphs, config)

    print(consolidated["entities"])
    print(consolidated["relations"])

asyncio.run(main())
```

---

## Config

`knwler.config.Config` is a dataclass that is accepted by every API function. All fields have sensible defaults so you only need to override what you care about.

```python
from knwler.config import Config

config = Config()                              # Ollama, local defaults
config = Config(backend="openai",
                openai_api_key="sk-...")       # OpenAI
config = Config(backend="anthropic",
                anthropic_api_key="...")       # Anthropic
```

### Backend & connection

| Field               | Default                                 | Description                                              |
| ------------------- | --------------------------------------- | -------------------------------------------------------- |
| `backend`           | `"ollama"`                              | Active backend: `"ollama"`, `"openai"`, or `"anthropic"` |
| `ollama_url`        | `"http://localhost:11434/api/generate"` | Ollama server endpoint                                   |
| `openai_api_key`    | `None`                                  | OpenAI API key (or set `OPENAI_API_KEY` env var)         |
| `openai_base_url`   | `"https://api.openai.com/v1"`           | Override for OpenAI-compatible providers                 |
| `anthropic_api_key` | `None`                                  | Anthropic API key                                        |

### Models

| Field                        | Default                       | Description                                |
| ---------------------------- | ----------------------------- | ------------------------------------------ |
| `ollama_extraction_model`    | `"qwen2.5:3b"`                | Ollama model for chunk extraction          |
| `ollama_discovery_model`     | `"qwen2.5:14b"`               | Ollama model for schema/language discovery |
| `openai_extraction_model`    | `"gpt-4o-mini"`               | OpenAI model for extraction                |
| `openai_discovery_model`     | `"gpt-4o"`                    | OpenAI model for discovery                 |
| `anthropic_extraction_model` | `"claude-haiku-4-5-20251001"` | Anthropic model for extraction             |
| `anthropic_discovery_model`  | `"claude-sonnet-4-6"`         | Anthropic model for discovery              |

Switch models at any point by creating a new `Config`:

```python
# Use a larger model for one call, then switch back
config_large = Config(ollama_extraction_model="qwen2.5:14b")
summary = await extract_summary(chunks, config_large)

config = Config()   # back to the default smaller model
graphs = await extract_all(chunks, schema, config)
```

### Chunking & generation

| Field            | Default | Description                                 |
| ---------------- | ------- | ------------------------------------------- |
| `max_tokens`     | `400`   | Maximum tokens per chunk                    |
| `overlap_tokens` | `50`    | Token overlap between consecutive chunks    |
| `max_concurrent` | `8`     | Maximum number of concurrent LLM requests   |
| `num_predict`    | `1024`  | Max tokens the LLM may generate per request |
| `temperature`    | `0.1`   | Sampling temperature                        |
| `use_cache`      | `True`  | Cache LLM responses to `~/.knwler`          |

---

## Chunking

```python
from knwler.chunking import chunk_text
from knwler.config import Config

chunks: list[str] = chunk_text(text, config)
```

Splits `text` into overlapping token-based chunks respecting `config.max_tokens` and `config.overlap_tokens`.

---

## Discovery

### `detect_language`

```python
from knwler.discovery import detect_language

lang: str = await detect_language(text, config)
# e.g. "en", "de", "fr"
```

Returns an ISO 639-1 two-letter language code for the document.

### `discover_schema`

```python
from knwler.discovery import discover_schema
from knwler.models import Schema

schema: Schema = await discover_schema(text, config)

print(schema.entity_types)    # ["person", "organization", ...]
print(schema.relation_types)  # ["works_at", "created", ...]
print(schema.reasoning)       # the model's explanation
```

The `Schema` is inferred from the document content. It is passed to every extraction call so the LLM produces consistently typed entities and relations.

---

## Extras

### `extract_title`

```python
from knwler.extras import extract_title

title: str = await extract_title(chunks, config)
```

Generates a short document title from the first `max_chunks` chunks (default: 3).

### `extract_summary`

```python
from knwler.extras import extract_summary

summary: str = await extract_summary(chunks, config)
```

Produces a 3–5 sentence summary from the first `max_chunks` chunks (default: 3).

### `rephrase_chunks`

```python
from knwler.extras import rephrase_chunks

rephrased: list[str] = await rephrase_chunks(chunks, config)
```

Rewrites each chunk in plain language. Useful for display in UIs or downstream summarisation.

---

## Extraction

### `extract_chunk` — single chunk

```python
from knwler.extraction import extract_chunk
from knwler.models import ExtractionResult

result: ExtractionResult = await extract_chunk(chunk, idx, schema, config)

print(result.chunk_idx)   # the idx you passed in – kept as-is for traceability
print(result.id)          # auto-generated UUID string
for e in result.entities:
    print(e["name"], e["type"], e["description"])
for r in result.relations:
    print(r["source"], r["type"], r["target"], r["strength"], r["description"])
```

`idx` is any integer you choose (e.g. the position in the chunks list). It is stored verbatim in the result so you can correlate chunks and graphs later.

### `extract_all` — all chunks in parallel

```python
from knwler.extraction import extract_all

graphs: list[ExtractionResult] = await extract_all(chunks, schema, config)
assert len(graphs) == len(chunks)
```

Runs `extract_chunk` over all chunks with a concurrency limit of `config.max_concurrent`. Optionally pass `output_path` to enable incremental saving of partial results.

---

## Consolidation

```python
from knwler.consolidation import consolidate_extracted_graphs

consolidated, elapsed = await consolidate_extracted_graphs(
    graphs,
    config,
    summarize=False,            # True: merge descriptions with LLM
    filter_low_importance=True, # False: keep singletons and low-value entities
)

# consolidated is a plain dict
entities  = consolidated["entities"]   # list[dict]
relations = consolidated["relations"]  # list[dict]
```

Merges individual chunk graphs into a single deduplicated graph. Entities are matched by `(name, type)` — same name but different type are treated as distinct entities.

| Parameter               | Default | Description                                              |
| ----------------------- | ------- | -------------------------------------------------------- |
| `summarize`             | `True`  | Use the LLM to merge and condense duplicate descriptions |
| `filter_low_importance` | `True`  | Drop singleton entities and low-strength relations       |

Pass `filter_low_importance=False` to retain every extracted entity regardless of how rarely it appeared:

```python
consolidated, _ = await consolidate_extracted_graphs(
    graphs, config, summarize=False, filter_low_importance=False
)
```

---

## Cache

All LLM responses are cached under `~/.knwler` by default (controlled by `config.use_cache`). You can inspect or filter the cache programmatically:

```python
from knwler.cache import find_cache_items

all_items = find_cache_items()
# or filter by model
items = find_cache_items(model="qwen2.5:14b")

for item in items:
    print(item["key"], item["model"], item["cached_at"])
```

Set `config.use_cache = False` to bypass the cache entirely for a run.

---

## Data models

### `Schema`

```
entity_types:   list[str]   # entity categories discovered in the document
relation_types: list[str]   # relation categories discovered in the document
reasoning:      str         # model's explanation for its choices
discovery_time: float       # seconds taken
```

### `ExtractionResult`

```
entities:     list[dict]  # {"id", "name", "type", "description"}
relations:    list[dict]  # {"source", "target", "type", "strength", "description"}
chunk_idx:    int         # the idx passed to extract_chunk
chunk_time:   float       # seconds taken
chunk_tokens: int         # token count of the source chunk
id:           str         # auto-generated UUID
```

### Consolidated graph

A plain `dict` with two keys:

```
entities:  list[dict]  # {"id", "name", "type", "description", "chunk_ids"}
relations: list[dict]  # {"source", "target", "type", "strength", "description"}
```
