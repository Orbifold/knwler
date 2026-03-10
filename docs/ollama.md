# Ollama

Ollama is just a convenient local LLM service, you can use LMStudio or any other service. The default model is Qwen 2.5 but here as well, experiment and see what works best for you.
We have done lots of benchmarks and bigger models are not better, sometimes quite the opposite. Small models of 3 or 7 billion parameters will be fine and a lot faster.
Thinking, in particular, is really standing in the way of graph extraction. Whatever you do, don't enable thinking and don't use advanced MOE models.

There are two Ollama model parameters you can set:

- `--extraction-model`: the model used for summary, title and graph extraction. Default is `qwen2.5:7b`.
- `--discovery-model`: used for language and schema detection. Default is `qwen2.5:14b`.


> **Tip:** When running Ollama locally, launch it via CLI with parallel processing for best throughput:
>
> ```bash
> OLLAMA_NUM_PARALLEL=8 ollama serve
> ```
>
> Adjust the number based on your machine specs (8 is suitable for a Mac M4 Pro with 64 GB RAM).
