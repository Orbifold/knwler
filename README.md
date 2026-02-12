# Knwler

**Turn any document into a structured knowledge graph**

Knwler is a lightweight, single-file Python tool that extracts structured knowledge graphs from documents using AI. Feed it a PDF or text file and receive a richly connected network of entities, relationships, and topics — complete with an interactive HTML report and exports ready for your favorite graph analytics platform.

Built for compliance teams, legal departments, research analysts, and anyone who needs to rapidly understand the structure hidden inside dense documents.

---

## Why Knwler?

| Challenge | How Knwler Solves It |
|---|---|
| Manually mapping entities and relationships in 100+ page regulatory documents | Automated extraction produces a navigable knowledge graph in minutes |
| Expensive vendor lock-in for document intelligence | Runs fully local with Ollama (zero data leaves your machine) or via OpenAI for speed |
| Documents in multiple languages across jurisdictions | Auto-detects language and adapts all prompts — supports English, German, French, Spanish, and Dutch out of the box |
| Results trapped inside one tool | Exports to HTML, GML, GraphML, and raw JSON — import directly into Neo4j, Gephi, yEd, Memgraph, SurrealDB, or any graph platform |
| High per-document processing costs | ~$0.20 per 20-page PDF with OpenAI/GPT-4o; completely free when running locally; LLM response caching means re-runs cost nothing |

---

## Key Features

### Dual LLM Backend — Cloud or Fully Local
Choose between **OpenAI** for maximum speed, or **Ollama** for fully offline, air-gapped operation. Qwen 2.5 at 3B–14B parameters delivers strong results locally. You can even switch backends between runs and incrementally augment the same graph.

### Automatic Schema Discovery
The pipeline analyzes a sample of your document and **infers the optimal entity types and relation types** — no manual ontology engineering required. You can also supply a schema if you wish. A schema is a set of types of entities (person, concept, location...) and relations (knows, has_accepted, has_signed...).

### Multilingual by Design
Language is **auto-detected** on every run. All prompts (summarization, extraction, community labeling) and all console/UI output are localized. Adding a new language is as simple as extending a single JSON file.

### Incremental & Augmentable
Re-run on new documents or updated schemas and **the existing graph is augmented** rather than rebuilt. Entity descriptions from multiple sources are intelligently consolidated via LLM-powered summarization.

### Community Detection & Topic Assignment
The Louvain algorithm automatically **discovers clusters of related entities** and an LLM labels each community with human-readable topics — giving you instant thematic insight into the document's structure.

### Self-Contained HTML Report
Export a **single HTML file** with interactive Cytoscape.js network visualization, entity index, topic overview, and rephrased text chunks — shareable without any server or dependencies.

### Rich Export Ecosystem
- **JSON** — the canonical output; import into Neo4j, Memgraph, SurrealDB, or generate vector embeddings
- **GML / GraphML** — open directly in yEd, Gephi, or any standards-compliant graph tool
- **HTML** — standalone interactive report

### Intelligent Caching
Every LLM call is **hashed and cached** locally. Re-generating reports, tweaking export settings, or re-running with a different schema costs zero additional API calls.

### Human-Readable Chunk Rephrasing
Each text chunk is rephrased for readability alongside the original, making the report accessible to non-expert stakeholders while preserving full traceability to source text.

### PDF & Text Ingestion
Handles **PDF-to-text extraction** (via PyMuPDF) as well as plain text and Markdown files. Extracted text is cached to avoid redundant PDF parsing on subsequent runs.

### Portable & Minimal
A **single Python file (~2,000 lines)**, managed via `uv` with minimal dependencies. No database, no backend server, no Docker required.

---

## Cost & Performance

| Scenario | Time (20-page PDF) | Cost |
|---|---|---|
| OpenAI GPT-4o / GPT-4o-mini | ~2–4 minutes | ~$0.20 |
| Ollama Qwen 2.5 (Mac M4 Pro, 64 GB) | ~20–40 minutes | Free |
| Cached re-run (any backend) | Seconds | Free |

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run with OpenAI
uv run main.py --openai -f document.pdf

# Run fully local with Ollama
uv run main.py -f document.pdf

# Re-export HTML only (no LLM calls)
uv run main.py --html-only
```

> **Tip:** When running Ollama locally, launch it via CLI with parallel processing for best throughput:
> ```bash
> OLLAMA_NUM_PARALLEL=8 ollama serve
> ```
> Adjust the number based on your machine specs (8 is suitable for a Mac M4 Pro with 64 GB RAM).

## CLI Options

| Option | Description |
|---|---|
| `--file`, `-f` | Input PDF or text file |
| `--openai` | Use OpenAI API instead of Ollama |
| `--extraction-model`, `-e` | Model for chunk extraction (default: `qwen2.5:3b` / `gpt-4o-mini`) |
| `--discovery-model`, `-d` | Model for schema discovery (default: `qwen2.5:14b` / `gpt-4o`) |
| `--concurrent`, `-c` | Max concurrent LLM requests (default: 10) |
| `--max-tokens` | Max tokens per chunk (default: 400) |
| `--no-discovery` | Skip schema discovery, use built-in defaults |
| `--no-cache` | Disable LLM response caching |
| `--language`, `-l` | Force language code (e.g., `en`, `de`, `fr`) — auto-detects if omitted |
| `--url`, `-u` | Source URL for metadata |
| `--output`, `-o` | Output JSON filename (saved to `results/`) |
| `--html-report` | Generate HTML report (default: on) |
| `--gml-export` | Generate GML graph file (default: on) |
| `--html-only` | Re-export HTML from existing results without re-running extraction |

## Examples

```bash
# EU AI Act (English)
uv run main.py --openai \
  --url "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689" \
  -f samples/EUAI.pdf

# NIST AI Risk Management Framework
uv run main.py --openai \
  --url "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf" \
  -f samples/Nist.pdf

# Belgian Civil Code (Dutch — auto-detected)
uv run main.py --openai \
  --url "https://www.ejustice.just.fgov.be/cgi/article_body.pl?language=nl&pub_date=2022-07-01&caller=list&numac=2022032058" \
  -f samples/BurgerlijkBoek5.pdf

# Deloitte Sustainability Report (German — auto-detected)
uv run main.py --openai \
  --url "https://www.deloitte.com/de/de/legal/publikationen.html" \
  -f examples/Deloitte/Deloitte-Nachhaltigkeitsbericht-2024.pdf
```

## Integration

The raw JSON output is designed for downstream integration:

- **Import into Neo4j / Memgraph / SurrealDB** — entities and relations map directly to nodes and edges
- **Generate vector embeddings** — use entity descriptions for semantic search
- **Feed into n8n workflows** — connect document intelligence to CRM, alerting, or reporting pipelines without code
- **Visualize in yEd or Gephi** — open the GML/GraphML export for advanced layout and analysis

---

## Examples

You can find example reports and raw graph data in diverse languages in the `examples` directory.

## Language 

Everything language related sits in the `languages.json` and this contains both the language-specific prompts as well as the text used for console output.
You can easily add additional languages, simply ask Copilot, Gemini or any AI to translate the JSON.

## OpenAI Key

If you run the process in your terminal the code will look for the usual `OPENAI_API_KEY`.
You can assign it explicitly via a terminal export

```bash
export OPENAI_API_KEY=...
```
or in the code (look for `os.environ.get("OPENAI_API_KEY", "")`).


## Ollama

Ollama is just a convenient local LLM service, you can use LMStudio or any other service. 
The default model is Qwen 2.5 but here as well, experiment and see what works best for you.
We have done lots of benchmarks and bigger models are not better, sometimes quite the opposite. Small models of 3 or 7 billion parameters will be fine and a lot faster.
Thinking, in particular, is really standing in the way of graph extraction. Whatever you do, don't enable thinking and don't use advanced MOE models.

## Disclaimer

The information extracted by Knwler is generated via machine learning and natural language processing, which may result in errors, omissions, or misinterpretations of the original source material. This tool is provided "as is" for informational purposes only. Users are advised to independently verify any critical data against original source documents before making business, legal, or financial decisions.

---

*Built by [Orbifold Consulting](https://orbifold.net) and inspired by [Knwl](https://knwl.ai)*.
