import pytest
from unittest.mock import patch
from knwler.chunking import chunk_text
from knwler.config import Config
from knwler.discovery import detect_language, discover_schema
from knwler.extras import rephrase_chunks, extract_title, extract_summary
from knwler.cache import find_items_by_model
from knwler.extraction import extract_chunk, extract_chunks
from knwler.consolidation import consolidate_chunk_graphs
from knwler.models import *
import asyncio


@pytest.mark.asyncio
async def test_ada():
    # ==============================================================================================
    # The followinng can also serve as an example of how to use the Knwler API.
    # This uses Ollama by default and all requests are cached, so repeated testing is fast and free.
    # ==============================================================================================

    # read in text
    with open("tests/data/ada.md", "r") as f:
        text = f.read()

    # small text, so no chunking needed, but we can test the chunking function anyway
    config = Config(max_tokens=200, overlap_tokens=20)  # gives four chunks
    chunks = chunk_text(text, config)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)

    # these are the chunks
    for i, c in enumerate(chunks):
        assert len(c.text) > 0
        print(f"\n-------Chunk {i + 1}-------\n")
        print(c)

    print(f"\n-------Discovery-------\n")
    # detect language
    lang = await detect_language(text, config)
    assert isinstance(lang, str) and len(lang) == 2 and lang == "en"
    print(f"\nDetected language: {lang}")

    # discover schema
    schema = await discover_schema(text, config)
    assert schema is not None
    assert isinstance(schema.entity_types, list)
    assert isinstance(schema.relation_types, list)
    print(f"\nEntity types: {", ".join(schema.entity_types)}")
    print(f"\nRelation types: {", ".join(schema.relation_types)}")
    print(f"\nReasoning: {schema.reasoning}")

    print(f"\n-------Extras-------\n")
    title = await extract_title(chunks, config)
    assert isinstance(title, str) and len(title) > 0
    print(f"\nTitle: {title}")

    rephrased_chunk1 = await rephrase_chunks([chunks[1]], config)
    assert isinstance(rephrased_chunk1, list) and len(rephrased_chunk1) > 0
    print("\n-----Rephrased chunk 1-------\n")
    print(f"\n {rephrased_chunk1[0]}")

    summary = await extract_summary(chunks, config)
    assert isinstance(summary, str) and len(summary) > 0
    print("\n-----Summary using Qwen 2.5:3B-------\n")
    print(f"\n {summary}")

    # as an example, let's change the LLM model used for the summary extraction
    config = Config(extraction_model="qwen2.5:14b")
    print("\n-----Summary using Qwen 2.5:14b-------\n")
    summary = await extract_summary(chunks, config)
    print(f"\n {summary}")

    fourteen_items = find_items_by_model(model="qwen2.5:14b")
    assert len(fourteen_items) > 0

    chunk1 = chunks[1]
    config = Config()  # back to the smaller model
    many_little_graphs = await extract_chunk(chunk1, schema, config)
    assert isinstance(many_little_graphs, ChunkGraph)
    print("\n-----Graph of chunk 1-------\n")
    assert (
        many_little_graphs.chunk.chunk_idx == 1
    )  # kept the index free in order to make the `extract_chunk` method as reusable as possible
    assert many_little_graphs.id is not None and isinstance(many_little_graphs.id, str)
    print(
        f"Chunk index {many_little_graphs.chunk.chunk_idx} received id {many_little_graphs.id}"
    )
    print(f"\nEntities ({len(many_little_graphs.entities)}):\n")
    for e in many_little_graphs.entities:
        print(f"\t- {e['name']} ({e['type']}): {e['description']}")
    print(f"\nRelations ({len(many_little_graphs.relations)}):\n")
    for r in many_little_graphs.relations:
        print(
            f"\t- {r['source']} - [{r['type']} (strength {r['strength']})] -> {r['target']} ℹ️  {r['description']}"
        )

    # the whole lot, this is async
    many_little_graphs = await extract_chunks(chunks, schema, config)
    assert isinstance(many_little_graphs, list) and len(many_little_graphs) == len(
        chunks
    )
    assert isinstance(many_little_graphs[0], ChunkGraph)

    # now you can consolidate the list of little graphs into one big graph
    consolidated, consolidation_time = await consolidate_chunk_graphs(
        many_little_graphs,
        config,
        summarize=False,
    )
    assert (
        isinstance(consolidated, Graph)
        and hasattr(consolidated, "entities")
        and hasattr(consolidated, "relations")
    )

    print("\n-----Consolidated graph-------\n")
    print(f"\nEntities ({len(consolidated.entities)}):\n")
    for e in consolidated.entities:
        print(f"\t- {e['name']} ({e['type']}): {e['description']}")
    print(f"\nRelations ({len(consolidated.relations)}):\n")
    for r in consolidated.relations:
        print(
            f"\t- {r['source']} - [{r['type']} (strength {r['strength']})] -> {r['target']} ℹ️  {r['description']}"
        )

    # you can also keep the singletons and low-value entities
    consolidated, consolidation_time = await consolidate_chunk_graphs(
        many_little_graphs, config, summarize=False, filter_low_importance=False
    )
    print("\n-----Consolidated graph with low-value entities-------\n")
    print(f"\nEntities ({len(consolidated.entities)}):\n")
    for e in consolidated.entities:
        print(f"\t- {e['name']} ({e['type']}): {e['description']}")
    print(f"\nRelations ({len(consolidated.relations)}):\n")
    for r in consolidated.relations:
        print(
            f"\t- {r['source']} - [{r['type']} (strength {r['strength']})] -> {r['target']} ℹ️  {r['description']}"
        )
