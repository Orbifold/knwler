# Neo4j Import

You can run NEo4j via the [desktop app](https://neo4j.com/download/) or via Docker

```bash
docker run --name neo -d \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=~/neo4j:/data \
    --env NEO4J_AUTH=neo4j/123456789 \
    neo4j

```

Assuming you have a database instance running and ingested a document via Knwler, you have a `graph.json` file from Knwler somewhere (typically in a `results` folder):

```bash
uv run main.py ./integrations/neo4j_import.py ./results/graph.json
```

If you have a specific Neo4j user, url and password you can specify it:

```bash
uv run main.py ./integrations/neo4j_import.py ./results/graph.json --url bolt://localhost:7687 --user Ian --password my-S3cret-thing
```

This will import chunks, entities, relations, clusters and create constraints, index and more.

From here on you can use Neo4j's Graph Data Science extension or other downstream tasks.
