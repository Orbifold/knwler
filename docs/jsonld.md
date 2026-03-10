# JSONLD Export

Export to RDF goes via JSONLD, the most popular RDF format. You can easily turn this into any other format, say to Turtle (*.ttl) like so

```python

from rdflib import Graph
g = Graph()
g.parse("graph.jsonld", format="json-ld")
print(len(g), "triples")
# Export to Turtle
print(g.serialize(format="turtle"))
```
Add the `rdflib` to you stack with `uv add rdflib`.
