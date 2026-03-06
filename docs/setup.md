# Setup

## Pipx

The easieast way to use Knwler is via `pipx`, see [this](./pipx.md) for more details.
If installed, simply use

```bash
pipx install knwler
```

and from here on you can use Knwler in your terminal

```
knwler --file ./my-document.pdf
```

## Pip

Knwler sits in the Pypi repository and it means you can install it like any other Python package

```bash
pip install knwler
```

## Poetry

Add Knwler like any other dependency

```bash
mkdir trying
cd trying
poetry init -q
poetry env use python3.12
poetry run pip install knwler
```

## UV

UV is the modern alternative to `pip` and `poetry`. If you have a new directory, init things like so

```bash
mkdir trying
cd trying
uv init
uv add knwler
```

If you cloned the code from Github use the usual

```bash
uv sync
```

which add the core dependencies. You can add all dependencies optional dependencies via

```bash
uv sync --all-groups
```

or only one specific like `neo4j` with

```bash
uv sync --group "neo4j"
```

Things like HelixDB and Neo4j have been moved to optional dependencies so it doesn't polute the Python env unless you need it in your work.
