# QA Checklist

So many new features added in v1.0.0 so here is a comprehensive checklist (for myself).

## Setup

- [x] uv install
- [x] pipx install
- [x] optional dependencies
- [x] pyproject cleanup

## Chore and PR

- [x] version set
- [x] Pypi deployment
- [x] Screenshots
- [x] LinkedIn
- [x] license

## Documentation

- [x] CLI help
- [x] get started
- [x] models and providers
- [x] pipx
- [x] wetboeken dataset sample
- [x] knwler.com website
- [x] readme
- [x] knwler as API
- [x] Ollama
- [x] OpenAI
- [x] Anthropic
- [x] rerun the four sample docs
- [x] templates

## Integrations

- [x] Neo4j export
- [x] Surreal export
- [x] GraphDB import
- [x] AWS import
- [x] Gemini + batch

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
- [x] Windows
- [x] Linux
- [x] LMStudio
- [x] OpenRouter
- [x] rename community to cluster
- [x] check localization file again
- [x] Jupyter notebooks
- [x] Spanish, Italian, Chinese examples

## Commands

- [x] knwler --version
- [x] knwler info version
- [x] knwler demo
- [x] knwler demo --backend gemini
- [x] knwler fetch url https://knwler.com/pdfs/HumanRights.pdf
- [x] knwler fetch wiki quantum
- [x] knwler cache clear wikipedia
- [x] knwler fetch url http://cnn.com --output ~/temp --open
- [x] knwler extract -f https://cnn.com --output ~/temp
- [x] knwler fetch wiki quantum --open
- [x] knwler fetch url https://knwler.com/pdfs/HumanRights.pdf --output ~/temp/knwler/ --parse
- [x] knwler extract -f https://knwler.com/pdfs/mbti.pdf --output ~/temp/knwler/ -m qwen2.5:7b
- [x] knwler -f https://knwler.com --backend gemini --output ~/temp/knwler
- [x] knwler batch run --input ~/temp/pdfs --output ~/temp/batching
- [x] knwler batch run --input ~/temp/pdfs --output ~/temp/batching --backend gemini --consolidate
- [x] knwler graph analyze ~/Desktop/AI/knwler-collect/examples/burgerlijk/BurgerlijkAll.json --output ~/Desktop/AI/knwler-website/examples/CivilAnalytics --open
- [x] knwler -f ~/Desktop/AI/knwler-collect/examples/Deloitte/Deloitte.pdf --output ~/Desktop/AI/knwler-website/examples/Deloitte --url https://www.deloitte.com/de/de/legal/publikationen.html --backend gemini
- [x] knwler -f ~/Desktop/AI/knwler-website/pdfs/NIST.pdf --output ~/Desktop/AI/knwler-website/examples/NIST --url https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf --backend gemini --template columns
- [x] knwler graph analyze ~/Desktop/AI/knwler-website/examples/EUAI_anthropic/default/graph.json --output ~/Desktop/AI/knwler-website/examples/HumanRightsAnalytics --open
- [x] knwler parse file ~/Desktop/AI/PdfKnowledgeBase/metagpt.pdf --output ~/temp/test.md
- [x] knwler export neo4j ~/temp/knwler.com/graph.json
- [x] knwler export surrealdb /Users/swa/temp/knwler.com/graph.json
- [x] knwler export jsonld /Users/swa/temp/knwler.com/graph.json

## Ideas

- [ ] SHACL
- [ ] kuzu/ladybug
- [ ] contradictions
- [ ] n8n
- [ ] FastAPI
- [ ] no console prints option
- [ ] https://github.com/Pro-GenAI/Index-RAG
- [ ] Ontocast https://growgraph.github.io/ontocast/
