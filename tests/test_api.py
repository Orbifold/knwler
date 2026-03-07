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