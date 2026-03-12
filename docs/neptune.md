# AWS Neptune Import

You can find an import script for AWS Neptune but there are many other paths to loading the data:

- use the SPARQL endpoint with `LOAD SILENT ...`
- use the OpenCypher enpoint and use the `neo4j_import` script
- use Neptune's bulk loading facility.

If you have extracted a `graph.json` file from one or more documents you can convert it to a JSONLD format via

```bash
uv run main.py graph convert /path/to/graph.json --format jsonld
```

Neptune runs inside a VPC and does not accept connections from the public internet. You have a few options to reach it:

**Option 1** — SSH tunnel through a bastion host (most common)

Then change the endpoint in main.py to localhost:

```python
NEPTUNE_ENDPOINT = "localhost"
```

and run `uv run aws_import.py` again.

**Option 2** — Run the script on an EC2 instance inside the same VPC
Copy main.py, graph.jsonld, and pyproject.toml to an EC2 instance in the same VPC/subnet as Neptune, install uv, and run it there.

**Option 3** — Enable public access on the Neptune cluster (only for dev/test)
In the AWS Console → Neptune → your cluster → Connectivity & security → enable "Publicly accessible". You may also need to update the security group to allow inbound TCP on port 8182 from your IP.

Once you have network access, re-run the script will upload the triples in batches into the named graph <https://knwler.com/graph/main>https://knwler.com/graph/main (or whatever namespace you have specified).
