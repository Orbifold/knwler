import pytest
from knwler.api import *
from pathlib import Path


@pytest.mark.asyncio
async def test_parse_file():
    file_path = Path("tests/data/ada.md")
    text, metadata = await parse_file(file_path)
    assert isinstance(text, str)
    assert "William King" in text
    assert isinstance(metadata, dict)
    assert metadata["file_path"] == str(file_path)
    print(f"Time taken: {metadata['time']}")


@pytest.mark.asyncio
async def test_fetch_url():
    url = "https://www.knwler.com"
    result = await fetch_url(url)
    assert isinstance(result, tuple)
    metadata, content = result
    assert isinstance(metadata, dict)
    print(metadata["content"][:100])  # Print the first 100 characters of the content
    assert isinstance(content, bytes)
    assert metadata["id"] == url
    assert isinstance(content, bytes)

    metadata, content = await fetch_url(
        "https://knwler.com/pdfs/HumanRights.pdf", no_cache=True
    )
    assert isinstance(metadata, dict)
    assert metadata["id"] == "https://knwler.com/pdfs/HumanRights.pdf"
    assert metadata["name"] == "HumanRights.pdf"
    assert metadata["extension"] == "pdf"
    assert isinstance(content, bytes)
    assert len(content) > 0

    # write the content to a file and check if it is a valid PDF
    with open("tests/data/test.pdf", "wb") as f:
        f.write(content)
    assert Path("tests/data/test.pdf").exists()
    # clean up
    Path("tests/data/test.pdf").unlink()


@pytest.mark.asyncio
async def test_cat():
    schema = Schema(entity_types=["animal", "object"], relation_types=["is_on"])
    r = await extract("The cat is on the table.", schema, config=Config())
    assert r.chunks[0].text == "The cat is on the table."
    assert len(r.graph.entities) == 2
    assert len(r.graph.relations) == 1
    assert r.graph.entities[0]["name"] == "cat"
    assert r.graph.entities[0]["type"] == "animal"
    assert r.graph.entities[1]["name"] == "table"
    assert r.graph.entities[1]["type"] == "object"
    assert r.graph.relations[0]["source"] == "cat"
    assert r.graph.relations[0]["target"] == "table"
    assert r.graph.relations[0]["type"] == "is_on"
    assert (
        r.graph.relations[0]["strength"] > 0.5
    )  # strength should be reasonably high for such a simple sentence

    # chunk references should be correctly set
    chunk_id = r.chunks[0].id
    assert r.graph.entities[0]["chunk_ids"] == [chunk_id]
    assert r.graph.entities[1]["chunk_ids"] == [chunk_id]
    assert r.graph.relations[0]["chunk_ids"] == [chunk_id]
    print(json.dumps(asdict(r), indent=2))

    schema = Schema(entity_types=["animal", "object"], relation_types=["is_on"])
    chunk = Chunk(text="The cat is on the table.", chunk_idx=16)
    g = await extract(chunk, schema, config=Config())
    assert len(g.graph.entities) == 2


@pytest.mark.asyncio
async def test_extract_pdf():
    file_path = Path("tests/data/People.pdf")
    if not file_path.exists():
        from knwler.api import fetch_url

        url = "https://knwler.com/pdfs/People.pdf"
        metadata, bytes = await fetch_url(url, no_cache=True)
        with open(file_path, "wb") as f:
            f.write(bytes)
    schema = Schema(entity_types=["person", "location"], relation_types=["is_from"])
    r = await extract(
        file_path,
        schema=schema,
        config=Config(
            extraction_model="gemma3:12b"
        ),  # different models wildly vary here
    )
    assert len(r.chunks) > 0
    assert len(r.graph.entities) > 0
    assert all(e["type"] in ["person", "location"] for e in r.graph.entities)
    assert all(r["type"] == "is_from" for r in r.graph.relations)
    assert all("chunk_ids" in e and len(e["chunk_ids"]) > 0 for e in r.graph.entities)
    assert all("chunk_ids" in r and len(r["chunk_ids"]) > 0 for r in r.graph.relations)
    assert len(r.graph.relations) > 0
    expected_people = [
        "Elena Petrova",
        "Samuel Okoye",
        "Lucía Fernández",
        "Noah Williams",
        "Sofia Dimitriou",
        "Hassan Al‑Masri",
        "Mei‑Ling Chen",
        "Arjun Mehta",
        "Oliver Grant",
    ]
    print("\nExtracted entities:")
    for e in r.graph.entities:
        print(f"- {e['name']} ({e['type']})")
    assert len(r.graph.entities) >= len(expected_people)
    print("\nExtracted relations:")
    for rel in (r for r in r.graph.relations if r["type"] == "is_from"):
        print(f"- {rel['source']}  {rel['target']}")
    assert len(r.graph.relations) == 9

    # print(json.dumps(asdict(r), indent=2))


@pytest.mark.asyncio
async def test_consolidation():
    schema1 = Schema(entity_types=["person", "location"], relation_types=["is_from"])
    file_path1 = Path("tests/data/People.pdf")
    if not file_path1.exists():
        url = "https://knwler.com/pdfs/People.pdf"
        metadata, bytes = await fetch_url(url, no_cache=True)
        with open(file_path1, "wb") as f:
            f.write(bytes)

    schema2 = Schema(entity_types=["city", "country"], relation_types=["located_in"])
    file_path2 = Path("tests/data/Places.pdf")
    if not file_path2.exists():
        url = "https://knwler.com/pdfs/Places.pdf"
        metadata, bytes = await fetch_url(url, no_cache=True)
        with open(file_path2, "wb") as f:
            f.write(bytes)
    config = Config(extraction_model="gemma3:12b")
    g1: DocumentGraph = await extract(file_path1, schema=schema1, config=config)
    g2: DocumentGraph = await extract(file_path2, schema=schema2, config=config)

    g = await consolidate_document_graphs([g1, g2], config=config, include_chunks=True)
    print(json.dumps(asdict(g), indent=2))
