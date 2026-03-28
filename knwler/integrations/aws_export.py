# ============================================================================================
#  Below is the content of pyproject.toml, which defines the project metadata and dependencies.
#
# [project]
# name = "knwler-neptune"
# version = "0.1.0"
# description = "Importing JSONLD into AWS Neptune using Python"
# readme = "README.md"
# requires-python = ">=3.12"
# dependencies = [
#     "rdflib>=6.3",
#     "requests>=2.31",
# ]
# ============================================================================================

from pathlib import Path

import requests
from rdflib import Graph, URIRef

NEPTUNE_ENDPOINT = (
    "db-neptune-1-instance-1.cpy8riandhgs.eu-central-1.neptune.amazonaws.com"
)
NEPTUNE_PORT = 8182
SPARQL_ENDPOINT = f"https://{NEPTUNE_ENDPOINT}:{NEPTUNE_PORT}/sparql"
NAMED_GRAPH = "https://knwler.com/graph/main"
BATCH_SIZE = 100  # triples per INSERT DATA request

# Characters invalid in N-Triples URI references that would break SPARQL parsing
_INVALID_URI_CHARS = frozenset("^\\{}")


def _is_valid_node(node) -> bool:
    return not (isinstance(node, URIRef) and _INVALID_URI_CHARS.intersection(str(node)))


def parse_jsonld(path: Path) -> list[tuple]:
    g = Graph()
    g.parse(str(path), format="json-ld")
    valid, skipped = [], 0
    for s, p, o in g:
        if _is_valid_node(s) and _is_valid_node(p) and _is_valid_node(o):
            valid.append((s, p, o))
        else:
            skipped += 1
    if skipped:
        print(f"  WARNING: skipped {skipped} triple(s) with invalid URI characters")
    return valid


def triples_to_nt(triples: list[tuple]) -> str:
    return "\n".join(f"{s.n3()} {p.n3()} {o.n3()} ." for s, p, o in triples)


def insert_batch(nt_block: str, session: requests.Session) -> None:
    update = (
        f"INSERT DATA {{\n"
        f"  GRAPH <{NAMED_GRAPH}> {{\n"
        f"{nt_block}\n"
        f"  }}\n"
        f"}}"
    )
    resp = session.post(
        SPARQL_ENDPOINT,
        data={"update": update},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()


def main() -> None:
    path = Path(__file__).parent / "graph.jsonld"
    print(f"Parsing {path} ...")
    triples = parse_jsonld(path)
    print(f"  {len(triples)} triples found")

    session = requests.Session()
    total_batches = max(1, (len(triples) + BATCH_SIZE - 1) // BATCH_SIZE)

    for i in range(0, len(triples), BATCH_SIZE):
        batch = triples[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(
            f"  Uploading batch {batch_num}/{total_batches} ({len(batch)} triples) ..."
        )
        insert_batch(triples_to_nt(batch), session)

    print(f"Done – {len(triples)} triples loaded into <{NAMED_GRAPH}>.")


if __name__ == "__main__":
    main()
