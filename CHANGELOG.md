# Changelog

## v1.0.4

- fix: benchmark output dir check
- other: reinstated global try-catch

## v1.0.3

- fix: pipx deployment fixes

## v1.0.0

- Version info via CLI
- Stack trace is now hidden on error using CLI
- You can now specify **a whole directory** via `--directory` rather than a single file `--file`
- Graph entities now carry document info (document node) towards **multi-document consolidation**
- CLI split in separate apps with `extract` as default for backward compatibility
- `html_only` now allows a specific `graph.json`, you can render a custom report
- Network viz has a slider to change the degree threshold
- `--overwrite` flag stops the proliferation of versions
- description summarization tuned to ensure correct use of mapping <id>
- **graph consolidation of multiple extractions**, both as a post-processing step and standalone
- **Anthropic** as LLM provider added
- Ollama thinking disabled by default, really does not help with graph extraction
- export template renders only what's available, sections without content are omitted
- **disambiguation of entities** with the same name but different type, also emphasized in the export via type badges
- dark/light **theme toggle**
- **new 3-column report template**
- uv optional dependencies
- poetry install issues fixed
- async await (as should be) across the board
- quite of bit of documentation and TLC towards adoption and usability
- cache and results dirs have moved to user dir
- the report now renders markdown output from rephrased chunks and descriptions
- an api (`knwler.collect`) to fetch url as markdown, fetch documents (pdf, xlsx...) and Wikipedia article. All cached for convenience.
- you can now give a URL to a pdf or just a webpage, all the rest happens automatically for you
- a higher level API (`knwler.api`) has been added to simplify downstream integrations
- the CLI has a new `fetch` command allowing to fetch data and parse in one go
- the CLI has a new `cache` command allowing to clear the cache
- benchmark suite added to compare speed and quality of models
- improved Neo4j import
- dropped Helix import, needs to be reconsidered
- improved SurrealDB import
- JSONLD/RDF export with docs for GraphDB
- batch processing for OpenAI
- graph analytics CLI commands, including an analytics report (what is the most important chunk in this document...?)
- graph conversion CLI command to take the `graph.json` to JSONLD, GML, GraphML and more.
- AWS Neptune import script and documentation
- additional languages: Italian, Portuguese and Simplified Chinese
- Google Gemini added as backend
- batch processing using Google Gemini, same characteristics as OpenaI (SQLite resume etc.)

## v0.4.1

- Fixed package deployment issue not taking the template dir into account

## v0.4.0

- Import scripts for Neo4j, SurrealDB and HelixDB
- Complete refactory of the monolithic main into a proper package
- MIT license specified
- knwler script in UV added, you can now use it via pipx
- Template rendering flitches are fixed
- Unit tests

## v0.3.0

- Published knwler to pypi (publishing GH workflow)
- Renamed `results.json` to `graph.json`, `results.gml` to `graph.gml`, `results.html` to `index.html`
- Output argument now correctly used to save diverse results
- Rich console output saved to txt and html in the results dir
- Running `uv run main.py` now shows the help system, as should be
- Removed the examples, they can be found on https://knwler.com
- Openai key is now picked up from the env

## v0.2.0

- Refactored `export_html` to use Jinja2 templates for cleaner HTML generation and easier customization
- Replaced Cytoscape visualization with custom concentric one, no hairballs anymore
- The four examples updated with the new graphviz

## v0.1.0

- Knowledge graph extraction from PDF and text documents
- Dual LLM backend support: OpenAI API and Ollama (local)
- Automatic schema discovery (entity types and relation types)
- Multilingual support with auto-detection (English, German, French, Spanish, Dutch)
- Localized prompts and UI strings via `languages.json`
- Community detection using Louvain algorithm with LLM-powered topic labeling
- Interactive HTML report with Cytoscape.js network visualization
- Export formats: JSON, GML, GraphML, HTML
- LLM response caching for cost-efficient re-runs
- Text chunk rephrasing for readability
- PDF-to-text extraction via PyMuPDF
- Incremental graph augmentation on re-runs
- Entity and relation consolidation with description summarization
- CLI interface with Typer and Rich console output
