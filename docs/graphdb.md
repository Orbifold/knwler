# Ontotext GraphDB Import

GraphDB is just one of many triple stores you can use to navigate RDF data. Things like [StarDog](https://www.stardog.com), [Qlever](https://github.com/ad-freiburg/qlever) and others are worth your time. Many factors are involved in picking the right DB for your business or use case.

You can easily install [GraphDB](https://graphdb.ontotext.com) via docker with something like

```bash
docker run -d -p 127.0.0.1:7200:7200 -v ~/graphdb:/opt/graphdb/home --name graphdb -t ontotext/graphdb:10.8.13
```

Navigate to `http://localhost:7200` and add a repository (say `knwler`). In the left panel you have an import menu and in there you can click the 'Upload RDF files'. Simply select the `graph.jsonld` file you exported via the `integrations/jsonld_export.py` script and GraphDB will do the rest.

The schema can be seen in the export script and a simple SPARQL query to fetch entities looks like

```sparql
prefix knwler:<https://knwler.com/ontology#>
prefix schema:<https://schema.org/>
SELECT ?name ?type WHERE { ?e a knwler:Entity ; schema:name ?name ; knwler:entityType ?type }
```
