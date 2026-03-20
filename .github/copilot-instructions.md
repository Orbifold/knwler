# Knwler — Copilot Instructions

Knwler is a Python tool that extracts structured knowledge graphs from documents using LLMs. It turns unstructured text (PDFs, URLs, text files) into a richly connected network of entities, relationships, and topics.

## Architecture

Pipeline-based flow — each stage feeds the next:

```
Text Input → chunk_text() → detect_language() + discover_schema()
  → extract_all() → consolidate_extracted_graphs()
  → cluster_graph() → export_html() → KnowledgeGraph
```

Key modules in `knwler/`:

| Module | Responsibility |
|--------|---------------|
| `config.py` | `@dataclass Config` — backend selection, model names, defaults per provider |
| `models.py` | Dataclasses: `Graph`, `AugmentedChunk`, `ExtractionResult`, `Schema`, `KnowledgeGraph` |
| `llm.py` | Multi-backend LLM abstraction via `llm_generate(prompt, config)` dispatching to provider-specific handlers |
| `extraction.py` | `extract_graph()` → `extract_chunk()` → `extract_all()` with async batching |
| `discovery.py` | `detect_language()` and `discover_schema()` for adaptive prompting |
| `consolidation.py` | Merges per-chunk results, deduplicates entities/relations |
| `clustering.py` | Community detection via Louvain, `create_network()` using NetworkX |
| `chunking.py` | Token-aware overlapping text chunking |
| `language.py` | Localized prompts from `languages.json`, supports EN/DE/FR/ES/NL/IT/PT/ZH |
| `cache.py` | JSON-based caching in `~/.knwler/cache/` by type (llm, webpage, wikipedia, document) |
| `export.py` | Jinja2-templated HTML report generation from `knwler/templates/` |
| `api.py` | Synchronous public API: `parse_file()`, `fetch_url()` for downstream integrations |
| `stats.py` | Graph analytics: entity/relation counts, timing, parallelism metrics |
| `collect/` | Content fetching: `webpage.py`, `wikipedia.py` |
| `integrations/` | Database imports: Neo4j, SurrealDB, AWS Neptune; JSON-LD/RDF export |

## Code Conventions

- **Async-first**: All I/O (LLM calls, HTTP, extraction) uses `async def` with `aiohttp`. Concurrency is limited via `config.max_concurrent` semaphore.
- **Dataclass models**: Lightweight `@dataclass` types with UUID id fields and computed properties (`entities_count`, `relations_count`). No ORMs.
- **Private functions**: Prefixed with `_` (e.g., `_post_json()`, `_save_partial_results()`).
- **Error handling**: `RuntimeError` with user-friendly context messages for LLM/auth/connection failures. File validation before processing. `typer.Exit(1)` for CLI validation failures.
- **LLM provider pattern**: `llm_generate()` dispatches to `llm_ollama_generate()`, `llm_openai_generate()`, `llm_anthropic_generate()`, etc. Each uses `_post_json()`. JSON is extracted from LLM output via `parse_json_response()`.
- **Caching**: Deterministic keys via `hash_args()`. Stored as JSON with `response` + `cached_at` metadata in `~/.knwler/cache/`.
- **Rich console**: Shared `console = Console(record=True)` used for all UI output and progress bars.

## CLI Structure

Typer-based CLI defined in `knwler/cli.py`. Each command area is a separate module with its own `typer.Typer()` app, registered via `app.add_typer()`:

| File | Command |
|------|---------|
| `cli_extract.py` | `knwler extract` — main extraction |
| `cli_fetch.py` | `knwler fetch` — download & cache content |
| `cli_consolidate.py` | `knwler consolidate` — merge graphs |
| `cli_cache.py` | `knwler cache` — view/clear cache |
| `cli_graph.py` | `knwler graph` — graph conversion & analytics |
| `cli_batch.py` | `knwler batch` — batch processing |
| `cli_benchmark.py` | `knwler benchmark` — performance benchmarks |
| `cli_info.py` | `knwler info` — version/metadata |
| `cli_demo.py` | `knwler demo` — demo workflows |

Common CLI options: `--backend`, `--api-key`, `--model`, `--max-tokens`, `--temperature`, `--max-concurrent`, `--output`, `--language`.

## Build & Test

```bash
# Install dependencies
uv sync

# Run unit tests (excludes llm, integration, cli markers)
python tests/run.py

# Run specific test file
pytest tests/test_extraction.py

# Run by marker
pytest -m "unit"
pytest -m "not llm and not integration"

# Run the CLI
uv run main.py extract -f document.pdf
# or via installed entry point:
knwler extract -f document.pdf
```

**Pytest markers** (from `pytest.ini`):
- `llm` — requires Ollama/LLM service running
- `integration` — requires external services
- `cli` — CLI tests
- `unit` — standalone unit tests
- `slow` — slow-running tests
- `basic` — basic functionality tests

## Supported LLM Backends

OpenAI, Anthropic (Claude), Google Gemini, Ollama (local), LM Studio (local), GitHub Models. Backend is selected via `Config.backend` or CLI `--backend` flag. Environment variables provide API keys.

## Export Formats

HTML (Jinja2 templates), GraphML, GML, JSON-LD/RDF, raw JSON. Database imports for Neo4j, SurrealDB, AWS Neptune via `knwler/integrations/`.

## When Adding New Features

- Follow the pipeline pattern: new processing stages go between existing pipeline steps.
- New LLM backends: add a `llm_{name}_generate()` function in `llm.py` and a dispatch branch in `llm_generate()`.
- New CLI commands: create `cli_{name}.py` with its own `typer.Typer()` app, register in `cli.py` via `app.add_typer()`.
- New export formats: add to `export.py` or `integrations/`.
- Tests: mirror source structure in `tests/`. Use `@pytest.mark.asyncio` for async tests. Mock LLM calls — don't require live services in unit tests.
- Python 3.12 is required (see `pyproject.toml`).
