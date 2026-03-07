# OpenAI

If you run the process in your terminal the code will look for the usual `OPENAI_API_KEY`.
You can assign it explicitly via a terminal export

```bash
export OPENAI_API_KEY=...
```

or in the code (look for `os.environ.get("OPENAI_API_KEY", "")`).

There are two OpenaAI model parameters you can set:

- `--extraction-model`: the model used for summary, title and graph extraction. Default is `gpt-4o-mini`.
- `--discovery-model`: used for language and schema detection. Default is `gpt-4o`.

These are used only if you enable the OpenAI flag, like so:

```bash
knwler extract --file <file path> --openai --extraction-model gpt-5.2
```
