"""
Graph extraction from text chunks.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from knwler.config import Config,  console
from knwler.models import ExtractionResult, Schema, Graph

from knwler.chunking import get_encoder
from knwler.language import get_console_msg, get_prompt
from knwler.llm import llm_generate, parse_json_response


# ---------------------------------------------------------------------------
# Single-chunk extraction
# ---------------------------------------------------------------------------
def extract_graph(text: str, schema: Schema, config: Config) -> dict[str, Any]:
    """Extract entities and relations from text."""
    prompt = get_prompt(
        "extract_graph",
        entity_types=json.dumps(schema.entity_types),
        relation_types=json.dumps(schema.relation_types),
        text=text,
    )

    if not prompt:
        prompt = (
            "Extract a knowledge graph from the text below.\n\n"
            f"ENTITY TYPES: {json.dumps(schema.entity_types)}\n"
            f"RELATION TYPES: {json.dumps(schema.relation_types)}\n\n"
            "RULES:\n"
            "- Only extract entities and relations clearly stated in the text\n"
            "- Each entity: name, type, 1-2 sentence description\n"
            "- Each relation: source, target, type, brief description, strength (1-10)\n\n"
            f'TEXT:\n"""{text}"""\n\n'
            "Return JSON:\n"
            "{\n"
            '  "entities": [{"name": "...", "type": "...", "description": "..."}],\n'
            '  "relations": [{"source": "...", "target": "...", "type": "...", '
            '"description": "...", "strength": 8}]\n'
            "}"
        )

    response = llm_generate(prompt, config)
    result = parse_json_response(response)

    return {
        "entities": result.get("entities", []),
        "relations": result.get("relations", []),
    }


def extract_chunk(
    chunk: str, idx: int, schema: Schema, config: Config
) -> ExtractionResult:
    """Extract from a single chunk with timing."""
    t0 = time.perf_counter()
    result = extract_graph(chunk, schema, config)
    elapsed = time.perf_counter() - t0

    return ExtractionResult(
        entities=result["entities"],
        relations=result["relations"],
        chunk_idx=idx,
        chunk_time=elapsed,
        chunk_tokens=len(get_encoder().encode(chunk)),
    )


# ---------------------------------------------------------------------------
# Partial results
# ---------------------------------------------------------------------------
def _save_partial_results(
    output_path: Path,
    schema: Schema,
    results: list[ExtractionResult],
    completed: int,
    total: int,
):
    """Save intermediate results to a partial file."""
    partial_path = output_path.with_suffix(".partial.json")
    partial_json = {
        "status": "in_progress",
        "progress": f"{completed}/{total}",
        "schema": {
            "entity_types": schema.entity_types,
            "relation_types": schema.relation_types,
            "reasoning": schema.reasoning,
        },
        "results": [
            {
                "chunk_idx": r.chunk_idx,
                "entities": r.entities,
                "relations": r.relations,
                "chunk_time": r.chunk_time,
                "chunk_tokens": r.chunk_tokens,
            }
            for r in sorted(results, key=lambda x: x.chunk_idx)
        ],
    }
    partial_path.write_text(json.dumps(partial_json, indent=2))


# ---------------------------------------------------------------------------
# Parallel extraction
# ---------------------------------------------------------------------------
async def extract_all(
    chunks: list[str],
    schema: Schema,
    config: Config,
    output_path: Path | None = None,
) -> list[ExtractionResult]:
    """Extract from all chunks with concurrency control and incremental saving."""
    semaphore = asyncio.Semaphore(config.max_concurrent)
    loop = asyncio.get_event_loop()
    total = len(chunks)
    results: list[ExtractionResult] = []
    lock = asyncio.Lock()
    progress_msg = get_console_msg("extracting") or "Extracting..."

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"[cyan]{progress_msg}", total=total)

        async def bounded(chunk: str, idx: int) -> ExtractionResult:
            async with semaphore:
                result = await loop.run_in_executor(
                    None, extract_chunk, chunk, idx, schema, config
                )
                async with lock:
                    results.append(result)
                    if output_path:
                        _save_partial_results(
                            output_path, schema, results, len(results), total
                        )
                    progress.update(task, advance=1)
                return result

        await asyncio.gather(*[bounded(c, i) for i, c in enumerate(chunks)])

    # Remove partial file on successful completion
    if output_path:
        partial_path = output_path.with_suffix(".partial.json")
        if partial_path.exists():
            partial_path.unlink()

    return sorted(results, key=lambda x: x.chunk_idx)
