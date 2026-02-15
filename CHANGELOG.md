# Changelog

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
