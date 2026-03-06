# Providers and Models

Knwler can be used with local and online LLM providers. The general wisdom here is that providers like OpenAI or Anthropic are faster and deliver more than a local LLM. On the other hand, you usually do not need that extra and the value you get out of a small model locally is often sufficient. It takes a little longer locally unless you have a solid GPU at your disposal.
One thing that is true for both local and online providers: bigger models do not perform better and are often a lot slower. Extracting information from text works with small models (3B, 7B, 9B models are fine).
This is the reason why the default in Knwler is set to `qwen2.5:7b` for graph extraction and `qwen2.5:14` for summarization.
We have also set 'thinking' off by default, it increases the processing time immensely without giving much added value.

All in all:

- use **a small local model if you want free and good-enough output**
- use small/cheap models with the big providers (OpenAI, Anthropic...) if you wish extract quality (more entities and relationships)

Of course, everything depends on your use-case. If your documents are full of images, maths, chemical formulas and whatnot you will need special models. Potentially special OCR as well. Knwler is especially good with legal and plain texts. It was designed as a utility for common usage and ease of use, not with high volumes or specialized documents in mind.

If you have massive amount of data you need batch processing and workflow infrastructure (Prefect, Airflow, Kafka and such). You can contact us for help and advice in that direction.
