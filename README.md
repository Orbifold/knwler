# Knwler

[![Pypi](https://img.shields.io/pypi/dm/knwler)](https://pypi.org/project/knwler/)
[![Version](https://img.shields.io/badge/version-0.5.0-green.svg)](https://github.com/Orbifold/knwler)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

[![Stars](https://img.shields.io/github/stars/Orbifold/knwler)](https://github.com/Orbifold/knwler)

**Turn documents into structured knowledge.**

Knwler is a lightweight Python tool that extracts structured knowledge graphs from documents using AI. Feed it a PDF or text file and receive a richly connected network of entities, relationships, and topics — complete with an interactive HTML report and exports ready for your favorite graph analytics platform.

Built for compliance teams, legal departments, research analysts, and anyone who needs to rapidly understand the structure hidden inside dense documents.

No big package dependencies, runs local if you wish, no licenses, no fuss.

![](./Screenshot1.png)

![](./Screenshot2.png)

---

## Table of Contents

- [Why Knwler?](#why-knwler)
- [What makes Knwler different?](#what-makes-knwler-different)
- [Key Features](#key-features)
- [Cost & Performance](#cost--performance)
- [Quick Start](#quick-start)
- [CLI Options](#cli-options)
- [Examples](#examples)
- [Integration](#integration)
- [Documentation](#documentation)
- [Disclaimer](#disclaimer)

---

## Why Knwler?

| Challenge                                                                     | How Knwler Solves It                                                                                                             |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Manually mapping entities and relationships in 100+ page regulatory documents | Automated extraction produces a navigable knowledge graph in minutes                                                             |
| Expensive vendor lock-in for document intelligence                            | Runs fully local with Ollama (zero data leaves your machine) or via providers for speed                                          |
| Documents in multiple languages across jurisdictions                          | Auto-detects language and adapts all prompts — supports English, German, French, Spanish, and Dutch out of the box               |
| Results trapped inside one tool                                               | Exports to HTML, GML, GraphML, and raw JSON — import directly into Neo4j, Gephi, yEd, Memgraph, SurrealDB, or any graph platform |
| High per-document processing costs                                            | ~$0.20 per 20-page PDF with OpenAI/GPT-4o; completely free when running locally; LLM response caching means re-runs cost nothing |
| Unstructured data to semantically organized knowledge                         | Simple CLI you can plug into your automation tool or as Python package. Small footprint. Customizable.                           |
| High threshold of graph RAG adoption                                          | Simple graph extraction. You can decide how to embed, which graph database, what agentic framework to use.                       |

Knwler does not implement graph RAG — it focuses on one thing: turning unstructured text into a knowledge graph.

## What makes Knwler different?

Its simplicity and direct use of LLMs. No agentic framework or sophisticated things. You are not bound to Neo4j or any vendor specific approach, you can change the LLM provider/model as you see fit, change the prompts since that's where the essence of Knwler resides.
If you do want to use LangChain or LlamaIndex, go ahead. You can leave out the rephrase step or the HTML export, you can use other tools to extract a schema and if the default Louvain clustering is not your thing, simply switch to another one.
You can engage PyKeen and Qdrant with the Knwler output, Pytorch Geometric or Neo4j's GDS, Memgraph Sage and Surreal's multi-modal storage. The extracted `graph.json` can articulate anything you like.

If you do need graph RAG, advanced graph visualization and enterprise knowledge graphs we can help, just [send us a mail](https://graphsandnetworks.com/contact).

---

## Key Features

### Dual LLM Backend — Cloud or Fully Local

Choose between **OpenAI** or **Anthropic** for maximum speed, or **Ollama** for fully offline, air-gapped operation. Qwen 2.5 at 3B–14B parameters delivers strong results locally. You can even switch backends between runs and incrementally augment the same graph.

### Automatic Schema Discovery

The pipeline analyzes a sample of your document and **infers the optimal entity types and relation types** — no manual ontology engineering required. You can also supply a schema if you wish. A schema is a set of types of entities (person, concept, location...) and relations (knows, has_accepted, has_signed...).

### Disambiguation

Apple as a company or apple as a fruit? Knwler identifies nodes based on name and type, so you can be certain that things are pinned correctly.

### Multilingual by Design

Language is **auto-detected** on every run. All prompts (summarization, extraction, community labeling) and all console/UI output are localized. Adding a new language is as simple as extending a single JSON file.

### Incremental & Augmentable

Re-run on new documents or updated schemas and **the existing graph is augmented** rather than rebuilt. Entity descriptions from multiple sources are intelligently consolidated via LLM-powered summarization.

### Community Detection & Topic Assignment

The Louvain algorithm automatically **discovers clusters of related entities** and an LLM labels each community with human-readable topics — giving you instant thematic insight into the document's structure.

### Self-Contained HTML Report

Export a **single HTML file** with interactive network visualization, entity index, topic overview, and rephrased text chunks — shareable without any server or dependencies.
It's based on a template (we deliver multiple styles or examples) and you are free to brand it to your needs.
We offer you out of the box:

- a standard report with a small network visualization
- a 3-column report without graph viz
- a graph viz focused example with custom layout algorithm.

### Rich Export Ecosystem

- **JSON** — the canonical output, all in one for whatever downstream you have in mind
- **GML / GraphML** — open directly in yEd, Gephi, or any standards-compliant graph tool
- **HTML** — standalone interactive report, templates you can tune to your needs (branding)
- **SurrealDB** — export to SurrealDB out of the box
- **Neo4j** — export to Neo4j included (indexing, constraints and all)

### Intelligent Caching

Every LLM call is **hashed and cached** locally. Re-generating reports, tweaking export settings, or re-running with a different schema costs zero additional API calls.

### Human-Readable Chunk Rephrasing

Each text chunk is rephrased for readability alongside the original, making the report accessible to non-expert stakeholders while preserving full traceability to source text.

### PDF & Text Ingestion

Handles **PDF-to-text extraction** (via PyMuPDF) as well as plain text and Markdown files. Extracted text is cached to avoid redundant PDF parsing on subsequent runs.

### Consolidation

Multiple runs (pdf extractions) can be consolidated into one knowledge graph. This effectively merges entities with summarization and graph refactorings. If you have a set of pdfs which cover a topic (say, legal domain) you can compile the knowledge graphs from the different documents into one.

### Portable & Minimal

Minimal dependencies, no database, no backend server, no Docker required. You can tune quality and speed using different models, online or local. It does not depend on any LLM or graph framework.
It's designed so you can use it as a Python package or via CLI, opening up integration with n8n, OpenClaw, or whatever automation system you like.

---

## Cost & Performance

| Scenario                            | Time (20-page PDF) | Cost   |
| ----------------------------------- | ------------------ | ------ |
| OpenAI GPT-4o / GPT-4o-mini         | ~2–4 minutes       | ~$0.20 |
| Ollama Qwen 2.5 (Mac M4 Pro, 64 GB) | ~20–40 minutes     | Free   |
| Cached re-run (any backend)         | Seconds            | Free   |

---

## Quick Start

**Requirements:** Python 3.12

```bash
# Install dependencies (recommended)
uv sync

# Or install as a package with pip
pip install knwler
```

```bash
# Run with OpenAI
uv run main.py --openai -f document.pdf

# Run fully local with Ollama
uv run main.py -f document.pdf

# Re-export HTML only (no LLM calls)
uv run main.py --html-only
```

When installed as a package, you can also use the `knwler` command directly:

```bash
knwler --openai -f document.pdf
```

> **Tip:** When running Ollama locally, launch it via CLI with parallel processing for best throughput:
>
> ```bash
> OLLAMA_NUM_PARALLEL=8 ollama serve
> ```
>
> Adjust the number based on your machine specs (8 is suitable for a Mac M4 Pro with 64 GB RAM).

## CLI Options

You can get help via `uv run main.py --help` or simply `uv run main.py`. See the comprehensive [CLI help doc here](./docs/cli.md).

## Examples

```bash
# EU AI Act (English)
uv run main.py --openai \
  --url "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689" \
  -f ./pdfs/EUAI.pdf

# NIST AI Risk Management Framework
uv run main.py --openai \
  --url "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf" \
  -f ./pdfs/Nist.pdf

# Belgian Civil Code (Dutch — auto-detected)
uv run main.py --openai \
  --url "https://www.ejustice.just.fgov.be/cgi/article_body.pl?language=nl&pub_date=2022-07-01&caller=list&numac=2022032058" \
  -f ./pdfs/BurgerlijkBoek5.pdf

# Deloitte Sustainability Report (German — auto-detected)
uv run main.py --openai \
  --url "https://www.deloitte.com/de/de/legal/publikationen.html" \
  -f ./pdfs/Deloitte/Deloitte-Nachhaltigkeitsbericht-2024.pdf
```

## Integration

The raw JSON output is designed for downstream integration:

- **Import into Neo4j / Memgraph / SurrealDB** — entities and relations map directly to nodes and edges
- **Generate vector embeddings** — use entity descriptions for semantic search
- **Feed into n8n workflows** — connect document intelligence to CRM, alerting, or reporting pipelines without code
- **Visualize in yEd or Gephi** — open the GML/GraphML export for advanced layout and analysis

There is in the `integrations` directory a script for:

- **Neo4j**: change the credentials and use `uv run integrations/neo4j_import.py ./results/graph.json` or wherever you have stored the JSON graph output from the ingestion.
- **SurrealDB**: similarly `uv run integrations/surrealdb_import.py ./results/graph.json`

The Surreal script uses the latest v3 version which now can act as a multi-modal store for documents and blobs. This means that you could store everything from chunk to vector embeddings in Surreal.

---

## Documentation

- [CLI](./docs/cli.md)
- [Setup](./docs/setup.md)
- [API](./docs/api.md)
- [Localization](./docs/language.md)
- [Models](./docs/models.md)
- [Ollama](./docs/ollama.md)
- [OpenAI](./docs/openai.md)
- [pipx](./docs/pipx.md)
- [Templates](./docs/templates.md)
- [Visualization](./docs/visualization.md)

## Disclaimer

The information extracted by Knwler is generated via machine learning and natural language processing, which may result in errors, omissions, or misinterpretations of the original source material. This tool is provided "as is" for informational purposes only. Users are advised to independently verify any critical data against original source documents before making business, legal, or financial decisions.

---

_Built by [Orbifold Consulting](https://orbifold.net) and inspired by [Knwl](https://knwl.ai)_.
