# Changelog

## v0.5.0

- Version info via CLI
- Stack trace is now hidden on error using CLI
- You can now specify a whole directory via `--directory` rather than a single file `--file`
- Graph entities now carry document info (document node) towards multi-document consolidation
- CLI split in separate apps with `extract` as default for backward compatibility
- `html_only` now allows a specific `graph.json`, you can render a custom report


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
