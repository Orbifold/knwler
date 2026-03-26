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
    print(json.dumps(asdict(r), indent=2))
