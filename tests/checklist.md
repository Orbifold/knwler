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
- [x] Screenshots
- [x] LinkedIn
- [x] license

## Documentation

- [ ] CLI help
- [ ] get started
- [x] models and providers
- [x] pipx
- [ ] wetboeken dataset sample
- [ ] knwler.com website
- [ ] readme
- [x] knwler as API
- [x] Ollama
- [x] OpenAI
- [x] Anthropic
- [ ] rerun the four sample docs
- [ ] templates

## Integrations

- [x] Neo4j export
- [x] Surreal export
- [x] GraphDB import
- [x] AWS import
- [ ] Gemini + batch

## Tests

- [x] green green super green `uv run tests`

## Nice to have

- [x] cache CLI `knwler cache clear`
- [x] full graphviz report template

## Diverse

- [x] apple test network viz should show two apple nodes
- [x] promised RDF
- [x] rephrased chunks are markdown and should be rendered as such
- [x] benchmark
- [ ] Windows
- [ ] Linux
- [x] LMStudio
- [ ] OpenRouter
- [ ] rename community to cluster
- [x] check localization file again

## Commands

- [x] uv run main.py demo
- [x] uv run main.py fetch url https://knwler.com/pdfs/HumanRights.pdf
- [x] uv run main.py fetch wiki quantum
- [x] uv run main.py cache clear wikipedia
- [x] uv run main.py fetch url http://cnn.com --output ~/temp --open
- [x] uv run main.py extract -f https://cnn.com --output ~/temp
- [x] uv run main.py fetch wiki quantum --open
- [x] uv run main.py fetch url https://knwler.com/pdfs/HumanRights.pdf --output ~/temp
- [x] uv run main.py fetch wiki topology --output ~/temp/ --open
- [x] uv run main.py fetch url https://knwler.com/pdfs/HumanRights.pdf --output ~/temp/things/ --parse
- [x] uv run main.py extract -f https://knwler.com/pdfs/mbti.pdf --output ~/temp/things/ --backend openai
- [x] uv run main.py -f https://knwler.com --backend gemini
- [x] uv run main.py batch run --input /Users/swa/temp/pdfs --output ~/temp/batching
- [x] uv run main.py batch run --input /Users/swa/temp/pdfs --output ~/temp/batching --backend gemini
