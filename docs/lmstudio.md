# LM Studio

[LM Studio](https://lmstudio.ai) is just like [Ollama](./ollama.md) a great platform to run LLMs locally.

By default LMStudio does not run the local server and you need to enable it explicitly ('Local Server' in the menu). The default port is `1234` (unlike Ollama's 11434), so when calling Knwler:

```bash
uv run main.py -f https://knwler.com/pdfs/mbti.pdf --backend lmstudio
```

You can optionally specify the base url

```bash
uv run main.py -f https://knwler.com/pdfs/mbti.pdf --backend lmstudio --base-url http://localhost:1234/ap/v1
```
