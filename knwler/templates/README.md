This dir contains the Jinja templates:

- **blank** is a blank base you can use to start your own custom presentation (also useful for debugging)
- **default** is the default 1-column rendering of the knowledge graph
- **columns** is the 3-columns rendering without graphviz
- **graph_analysis** renders the graph analytics and can't be used to render the `graph.json`.
- **research** is a 2-column academic article layout using serif typography, with an abstract block, column-ruled body, and no graph visualization.
- **cypher** is an interactive Cypher query console that runs entirely in the browser. It embeds the graph data and lets users query it with a subset of the Cypher query language (MATCH, WHERE, RETURN, ORDER BY, LIMIT).
