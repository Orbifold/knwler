# Knwler CLI Reference

Knwler extracts structured knowledge graphs from documents (PDF, text, Markdown) using LLMs. Results are saved as JSON, GML, and an interactive HTML report.

```
knwler [OPTIONS] COMMAND [ARGS]
```

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-V` | Show version and exit |
| `--help` | | Show help and exit |

---

## Commands

| Command | Description |
|---------|-------------|
| [`extract`](#extract) | Run the extraction pipeline on a file or directory |
| [`consolidate`](#consolidate) | Consolidate multiple extracted graphs into one |
| [`info`](#info) | View info about the Knwler installation |

---

## `extract`

Run the full extraction pipeline on a single file or a directory of files.

```
knwler extract [OPTIONS]
```

Either `--file` or `--dir` is required. The pipeline runs schema discovery, chunking, title/summary extraction, chunk rephrasing, entity/relation extraction, community detection, and export in sequence.

### Input

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--file PATH` | `-f` | — | Path to a single `.txt`, `.pdf`, or `.md` file |
| `--dir PATH` | `-D` | — | Path to a directory; all `.txt`, `.pdf`, and `.md` files inside are processed |

### Backend

Three backends are supported. Exactly one may be active per run; `--openai` and `--anthropic` are mutually exclusive.

| Option | Default | Description |
|--------|---------|-------------|
| *(none)* | ✓ | **Ollama** — fully local, no API key required. Needs an Ollama server at `http://localhost:11434` |
| `--openai` | `false` | **OpenAI** — requires `OPENAI_API_KEY` env var |
| `--openai-base-url URL` | `https://api.openai.com/v1` | Override the OpenAI base URL. Use this for any OpenAI-compatible provider (see [Compatible providers](#openai-compatible-providers)) |
| `--anthropic` | `false` | **Anthropic** — requires `ANTHROPIC_API_KEY` env var |

For best Ollama throughput, launch it with parallel processing enabled:

```bash
OLLAMA_NUM_PARALLEL=8 ollama serve
```

### Models

`--extraction-model` and `--discovery-model` are universal flags that work for both **Ollama** and **OpenAI** backends. The default changes automatically based on which backend is selected. For **Anthropic**, use the dedicated flags below.

| Option | Short | Ollama default | OpenAI default | Description |
|--------|-------|----------------|----------------|-------------|
| `--extraction-model MODEL` | `-e` | `qwen2.5:3b` | `gpt-4o-mini` | Model for chunk-by-chunk entity/relation extraction |
| `--discovery-model MODEL` | `-d` | `qwen2.5:14b` | `gpt-4o` | Model for schema discovery |
| `--anthropic-extraction-model MODEL` | | — | — | Anthropic-only: extraction model (default: `claude-haiku-4-5-20251001`) |
| `--anthropic-discovery-model MODEL` | | — | — | Anthropic-only: discovery model (default: `claude-sonnet-4-6`) |

> **Tip:** Larger or reasoning/MoE models are not necessarily better for extraction. Avoid models with "thinking" mode enabled — it significantly degrades graph extraction quality.

### Performance

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--concurrent N` | `-c` | `10` | Maximum number of simultaneous LLM requests. Increase for faster processing if your hardware and API limits allow |
| `--max-tokens N` | | `400` | Maximum tokens per text chunk. Smaller chunks produce more granular graphs; larger chunks reduce total LLM calls |
| `--no-cache` | | `false` | Disable LLM response caching. By default all responses are cached locally — re-runs (e.g. changing export settings) cost zero additional API calls |

### Schema

| Option | Default | Description |
|--------|---------|-------------|
| `--no-discovery` | `false` | Skip the schema discovery step and fall back to built-in generic defaults. Discovery uses an LLM to infer entity and relation types that are optimal for your specific document |

**Default schema (used when `--no-discovery` is set):**

- Entity types: `person`, `organization`, `technology`, `location`, `project`, `concept`, `event`
- Relation types: `works_at`, `created`, `lives_in`, `located_in`, `uses`, `partners_with`, `supports`, `integrates_with`, `related_to`, `requires`, `leads_to`

### Language

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--language CODE` | `-l` | auto-detect | Force a specific language (e.g. `en`, `de`, `fr`, `es`, `nl`). When omitted, the language is detected automatically from the document text. All prompts and console output are localized |

Supported language codes out of the box: `en`, `de`, `fr`, `es`, `nl`. Adding new languages requires only extending `languages.json`.

### Output

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output PATH` | `-o` | `results/<timestamp>/` | Directory where results are saved. Created if it does not exist. When omitted, a timestamped directory is created and renamed to the document title after extraction |
| `--html-report` | | `true` | Generate a self-contained `index.html` interactive report (Cytoscape.js visualization, entity index, topic overview) |
| `--gml-export` | | `true` | Export a `graph.gml` file, openable directly in Gephi, yEd, and other graph tools |
| `--html-only` | | `false` | Skip extraction entirely and re-render the HTML report from an existing `graph.json`. Requires `--output` pointing to an existing results directory or a `graph.json` file |
| `--url URL` | `-u` | — | Source URL of the document. Stored as metadata in the JSON output; not used for extraction |
| `--overwrite-dir` | | `false` | Allow the output directory to be overwritten if it already exists. By default a timestamp suffix is appended instead |

### Multi-document & Consolidation

| Option | Default | Description |
|--------|---------|-------------|
| `--consolidate` | `false` | After processing all files in a `--dir` run, merge all individual graphs into a single consolidated graph (saved as `consolidated_graph.json`). Automatically applies entity clustering across documents |

When `--dir` is used, each file gets its own sub-directory inside `--output` (named after the file stem). The individual `graph.json` files can later be merged with the standalone `consolidate` command.

---

## `consolidate`

Merge multiple previously extracted `graph.json` files into a single unified graph.

```
knwler consolidate [OPTIONS]
```

The command recursively searches for `graph.json` files, merges entities and relations across all documents, applies entity clustering (to group semantically equivalent entities from different sources), and saves `consolidated_graph.json`.

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--dir PATH` | `-D` | `results/` | Directory to search recursively for `graph.json` files. Defaults to the `results/` directory in the current working directory |

The consolidated output is written to `results/consolidated_graph.json` (or inside `--dir` if specified).

---

## `info`

View information about the Knwler installation.

```
knwler info COMMAND
```

### Sub-commands

| Command | Description |
|---------|-------------|
| `knwler info version` | Print the installed Knwler version |

---

## Output Files

Every extraction run produces the following files inside the results directory:

| File | Description |
|------|-------------|
| `graph.json` | Full structured output: title, summary, schema, entities, relations, communities, per-chunk data, and run statistics |
| `index.html` | Interactive HTML report (if `--html-report` is enabled) |
| `graph.gml` | GML graph export (if `--gml-export` is enabled) |
| `log.txt` | Plain-text console log |
| `log.html` | Rich HTML console log |
| `<stem>_extracted.txt` | Cached plain-text extracted from PDF (only for PDF inputs; avoids redundant parsing on re-runs) |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required when `--openai` is used |
| `ANTHROPIC_API_KEY` | Required when `--anthropic` is used |

---

## OpenAI-Compatible Providers

The `--openai` backend works with any OpenAI-compatible API — just point `--openai-base-url` at the provider's endpoint. No code changes required.

| Provider | Command |
|----------|---------|
| **Groq** (fast inference) | `--openai --openai-base-url https://api.groq.com/openai/v1 -e llama-3.3-70b-versatile -d llama-3.3-70b-versatile` |
| **LM Studio** (local) | `--openai --openai-base-url http://localhost:1234/v1 -e my-model` |
| **Azure OpenAI** | `--openai --openai-base-url https://<resource>.openai.azure.com/openai/deployments/<deployment>` |
| **Mistral AI** | `--openai --openai-base-url https://api.mistral.ai/v1 -e mistral-small-latest` |

Set the provider's API key in `OPENAI_API_KEY` before running (Azure uses `OPENAI_API_KEY` for the token too).

---

## Examples

### Basic extraction with Ollama (local, default)

```bash
knwler extract -f document.pdf
```

### Extract using OpenAI with a source URL

```bash
export OPENAI_API_KEY=sk-...
knwler extract --openai -f samples/EUAI.pdf \
  --url "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689"
```

### Extract using Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
knwler extract --anthropic -f document.pdf
```

### Anthropic with custom models

```bash
export ANTHROPIC_API_KEY=sk-ant-...
knwler extract --anthropic -f report.pdf \
  --anthropic-extraction-model claude-haiku-4-5-20251001 \
  --anthropic-discovery-model claude-sonnet-4-6
```

### Use Groq (fast inference, OpenAI-compatible)

```bash
export OPENAI_API_KEY=<groq-api-key>
knwler extract --openai \
  --openai-base-url https://api.groq.com/openai/v1 \
  -e llama-3.3-70b-versatile \
  -f document.pdf
```

### Process a directory of files, then consolidate

```bash
knwler extract --dir ./documents --output ./results --consolidate
```

### Process a directory without auto-consolidation, then consolidate separately

```bash
knwler extract --dir ./documents --output ./results
knwler consolidate --dir ./results
```

### Force German, skip schema discovery, save to a specific directory

```bash
knwler extract -f bericht.pdf --language de --no-discovery --output ./results/bericht
```

### Use a faster local model for extraction with more concurrency

```bash
knwler extract -f report.pdf -e qwen2.5:7b -d qwen2.5:14b --concurrent 16
```

### Re-export the HTML report without re-running the LLM pipeline

```bash
knwler extract --html-only --output ./results/My_Document
```

### Use LM Studio (local, OpenAI-compatible)

```bash
knwler extract -f doc.txt --openai \
  --openai-base-url http://localhost:1234/v1 \
  -e my-local-model
```

### Augmenting an existing graph

If you run `extract` with `--output` pointing to a directory that already contains a `graph.json`, the pipeline enters **augment mode**: new entities and relations are merged into the existing graph rather than replacing it. This lets you incrementally enrich the same graph from different documents or with updated schemas.

---

## Integration

The `graph.json` output is structured for direct downstream use:

- **Neo4j / Memgraph / SurrealDB** — entities map to nodes, relations to edges. Import scripts are in `integrations/`
- **Gephi / yEd** — open `graph.gml` directly
- **Vector search** — use entity descriptions and rephrased chunks for embedding
- **n8n / workflow automation** — parse JSON with any HTTP node or script step

```bash
# Neo4j import
uv run integrations/neo4j-import.py ./results/My_Document/graph.json

# SurrealDB import
uv run integrations/surrealdb-import.py ./results/My_Document/graph.json
```

---

## Performance Tips

- Run Ollama with `OLLAMA_NUM_PARALLEL=8 ollama serve` to fully saturate `--concurrent` requests locally.
- Tune `--max-tokens` (default 400) — smaller values create more chunks and finer-grained graphs at the cost of more LLM calls; larger values do the opposite.
- The cache is enabled by default. Re-runs with different export flags (e.g. adding `--html-report`) are instant and free.
- For large document sets, use `--dir` with `--consolidate` to batch-process and merge in a single command.
