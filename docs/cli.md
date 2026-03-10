# Knwler CLI

Knwler can be used via your terminal or as a Python package. See the [API page](./api.md) for more details on how to use individual methods.

The CLI contains a lot of functionality and rather than listing all possible actions, you will find below concrete examples. Note that if you have installed Knwler via pipx you can simply use `knwler ...` rather than `uv run main.py ...`.

There is a demo which uses the Human Rights declaration as short pdf:

```bash
uv run main.py demo
```

This will run the whole pipeline with all options:

- download the pdf
- parse the pdf to markdown
- chunk the text
- infer a schema
- rephrase the chunks
- extract a little knowledge graph for every chunk
- consolidate the graphs
- extract a title
- extract a summary
- render a report (html).

The `cli_demo.py` contains all of this and can serve as a guide on how to use Knwler as a Python package. You can use any of the steps individually or leave out what you don't need (e.g. the rephrasing of chunks).

Of course, the main reason you would try Knwler is to extract your own document:

```bash
uv run main.py -f https://knwler.com/pdfs/HumanRights.pdf
```

and if you have a local file:

```bash
uv run main.py -f ./HumanRights.pdf
```

This will output lots of files in a `results` directory:

- `graph.gml` a format convenient for graph visualization and graph analytics
- `graph.json` contains everything you need for downstream tasks
- `index.html` a rendering of the `graph.json` data
- `log.htnl` the log in html format
- `log.txt` the log in text format.

You can output things to a different directory with

```bash
uv run main.py -f ./HumanRights.pdf --output ./stuff
```

Every time you run the same document a new directory will be created, you can override this with:

```bash
uv run main.py -f ./HumanRights.pdf --output ./stuff --overwrite
```

The above commands will all use Ollama as backend. If you want OpenAI instead simply use:

```bash
uv run main.py -f ./HumanRights.pdf --output ./stuff --overwrite --openai
```

and make sure the API key is in your environment. If not, use

```bash
export OPENAIAPI_KEY=sk-....
```

and similarly for Anthropic:

```bash
uv run main.py -f ./HumanRights.pdf --output ./stuff --overwrite --anthropic
```

which, according to our benchmarks, will give you the highest qualirty output.

The rendered HTML uses a template and you can switch to another one via:

```bash
uv run main.py -f ./HumanRights.pdf --output ./stuff --overwrite --anthropic --template columns
```

Note that this will use the cached LLM exchanges and you won't have to pay again for a different style.

If you wish to understand the quality a model's output you can use the benchmark utility:

```bash
uv run main.py benchmark run
```

The `grid.json` file contains the items which will be benchmarked. This will render a comprehensive report and sort the results based on a knowledge yield score.

## Performance Tips

- Run Ollama with `OLLAMA_NUM_PARALLEL=8 ollama serve` to fully saturate `--concurrent` requests locally.
- Tune `--max-tokens` (default 400) — smaller values create more chunks and finer-grained graphs at the cost of more LLM calls; larger values do the opposite.
- The cache is enabled by default. Re-runs with different export flags (e.g. adding `--html-report`) are instant and free.
- For large document sets, use `--dir` with `--consolidate` to batch-process and merge in a single command.
