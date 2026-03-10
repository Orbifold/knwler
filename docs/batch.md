# OpenAI Batch Processing

[Batch processing](https://developers.openai.com/api/docs/guides/batch/) is a substantial cost reduction if you want to extract knowledge graphs from many documents. You can find a utility to handle a whole directory in batch. It will

- perform the whole extraction pipeline
- create batches for the API
- submit the batches
- poll for completion
- download the results
- continue the pipeline
- finalize everything neatly in separate directires.

In essence, simply give it a dir with documents and wait for it. OpenAI commits to process batches within 24 hours but in practice you usually only have to wait a short while. The processor uses a simple SQLite database to keep track of things and allows to resume processing if something goes wrong (or you terminated the process).

To start or resume processing:

```bash
uv run batch_openai.py --input ./documents --output ./results
```

If you want to check the status:

```bash
uv run batch_openai.py --input ./documents --output ./results --status
```

Change the default models:

```bash
uv rub batch_openai.py -i ./docs -o ./out --discovery-model gpt-4o --extraction-model gpt-4o-mini
```

Of course you need the `OPENAI_API_KEY` environment variable to be set.

> [!IMPORTANT]
> If you start a new batch, add additional document or change the process in any way you need to delete the output directory. The SQLites database is meant for resume but not for incremental processing.

When processing is done you can use Knwler's consolidation command to merge the separate knowledge graphs:

```bash
uv run main.py consolidate --dir ./out --output ./merged
```
