# Cache

The Knwler cache sits in your user folder, typically under `~/.knwler/cache` and is divided in four categories:

- documents: caches documents (pdf, docx...) fetched data via e.g. with `knwler fetch https://knwler.com/pdfs/HumanRights.pdf`
- llm: caches any LLM call (applies to all providers)
- webpages: caches web pages fetched with for instance `knwler fetch https://abc.com`
- wikipedia: caches 