# QA Checklist

So many new features added in v0.5.0 so here is a comprehensive checklist (for myself).

## Setup

- [x] uv install
- [x] pipx install
- [x] optional dependencies
- [x] pyproject cleanup

## Chore and PR

- [x] version set
- [ ] Pypi deployment
- [ ] Screenshots
- [ ] LinkedIn
- [x] license

## CLI

- [ ] version

## Documentation

- [ ] CLI help
- [ ] get started
- [ ] models and providers
- [x] pipx
- [ ] wetboeken dataset sample
- [ ] knwler.com website
- [ ] readme
- [x] knwler as API
- [x] Ollama
- [x] OpenAI
- [ ] Anthropic
- [ ] rerun the four sample docs
- [ ] templates

## Integrations

- Neo4j export
- Surreal export

## Tests

- [ ] green green super green `uv run tests`

## Nice to have

- [ ] cache CLI `knwler cache clear`
- [ ] full graphviz report template

## Diverse

- [ ] apple test network viz should show two apple nodes
- [ ] promised RDF
- [ ] rephrased chunks are markdown and should be rendered as such
- [ ] benchmark

```bash
uv run main.py -f ./pdfs/HumanRights.pdf --url https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/eng.pdf --overwrite-dir

uv run main.py -f ./pdfs/HumanRights.pdf --url https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/eng.pdf --overwrite-dir --discovery-model gemma3:4b  --extraction-model gemma3:4b

uv run main.py -f ./pdfs/HumanRights.pdf --url https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/eng.pdf --overwrite-dir --discovery-model qwen3.5:0.8b  --extraction-model qwen3.5:0.8b
```
