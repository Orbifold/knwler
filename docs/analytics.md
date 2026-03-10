# Graph Analytics

There are two post-processing CLI commands:

- `graph convert` to convert from `graph.json` to GML, GraphML, JSONLD and more
- `graph analyze` to apply graph analytical techniques to the extraction

These are post-processing commands in the sense that you need to have `graph.json` before you can use them. Use the [CLI extraction](./cli.md) to generate a json file and next use

```bash
uv run main.py graph analyze ./graph.json
```

to generate a comprehensive report of the most important entities, chunks and other information.
