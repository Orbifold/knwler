# Rendering Templates

The [Jinja](https://jinja.palletsprojects.com/en/stable/) template are easily customized and you can find below the parameters the template expects.

There is a plain 'blank' template from which you can start which contain the bare parameters without any styling.

## Variables

- **lang**: the language used as HTML metadata
- **title**: the title appearing at the top
- **summary**: a paragraph about the document
- **metadata**: some numbers like the entity count, the date and so on
- **labels**: the localized labels used, defaults to English
- **clusters**: the cluster list
- **chunks**: the chunk list
- **entities_display**: the entities with extra in order to link to chunks etc.
- **disclarimer**: a disclaimer
- **node_elements**: the actual nodes
- **edge_elements**: the actual edges
- **rawData**: which can be used to feed a graphviz
- **minimumDegree**: the minimum node degree to filter out (helps to show only linked data)
