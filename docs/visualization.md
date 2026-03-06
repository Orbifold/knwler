## Graph Visualization

Knwler output a `graph.gml` file you can use with various packages:

- [yEd Online](https://www.yworks.com/yed-live/) is a free web-based graph visualization tool
- [yEd Desktop](https://www.yworks.com/products/yed) is also free and runs as a desktop app on all OS
- [Gephi](https://gephi.org) is a free app with powerful network analysis

and many more.

It's also very easy to export other formats (GraphML e.g.) but GML seems to be the most lightweight and compatible format. Internally, Knwler uses NetworkX to serialize things and you can find [here](https://networkx.org/documentation/stable/reference/readwrite/index.html) various alternatives to export the knowledge graphs.

The raw JSON data sits in `graph.json` and can also be used for visualization purposes, especially in HTML/JS apps.