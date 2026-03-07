# HTML Export

The HTML output is a combination of the `graph.json` (aka knowledge graph or KG) and a template. it is a rendering of the knowledge graph using simple constructs (Jinja, Mustache syntax). This means you can define and brand your own template and render the KG in a way appropriate to your use case.

[You can find the template documentation here.](https://jinja.palletsprojects.com/en/stable/templates/)

We have provided a couple of templates as an example, the essence of Knwler is really the extracted JSON and not the HTML output.

Similarly, the network visualization is just an example of what one can do with the KG and not in any way tied to data. In fact, we encourage you to discover [yFiles](https://www.yfiles.com/demos) and [Ogma](https://doc.linkurious.com/ogma/latest/) towards enterprise level visualization of KGs. 

> [!NOTE]
> Even though the extraction prompt explicitly asks to only create relations between extracted entities, smaller models sometimes fail to honor the request. These mistakes depend on the model and its parameter count.