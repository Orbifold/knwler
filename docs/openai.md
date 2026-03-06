# OpenAI


If you run the process in your terminal the code will look for the usual `OPENAI_API_KEY`.
You can assign it explicitly via a terminal export

```bash
export OPENAI_API_KEY=...
```

or in the code (look for `os.environ.get("OPENAI_API_KEY", "")`).