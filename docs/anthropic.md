# Anthropic

[Anthropic](https://www.anthropic.com) will give you high-quality extractionns and i
If you run the process in your terminal the code will look for the usual `ANTRHOPIC_API_KEY`.
You can assign it explicitly via a terminal export

```bash
export ANTRHOPIC_API_KEY=...
```

or in the code (look for `os.environ.get("ANTRHOPIC_API_KEY", "")`).

There are two Anthropic model parameters you can set:

- `--extraction-model`: the model used for summary, title and graph extraction. Default is `claude-haiku-4-5-20251001`.
- `--discovery-model`: used for language and schema detection. Default is `claude-sonnet-4-6`.

These are used only if you enable the Anthropic flag, like so:

```bash
knwler extract --file <file path> --anthropic
```
