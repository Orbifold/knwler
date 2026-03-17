import json
import os
import asyncio
from pathlib import Path
import time
from knwler.config import Config
from benchmark.report import generate_report
from dataclasses import asdict
from knwler.api import (
    chunk,
    cluster_graph,
    consolidate,
    discover_language,
    extract_chunks,
    extract_summary,
    extract_title,
    fetch_url,
    infer_schema,
    parse_file,
    rephrase_chunks,
)
from rich.panel import Panel
from rich.padding import Padding
from rich.console import Console

console = Console()
root_path = Path(__file__).parent.parent
output_path = root_path / "benchmark" / "output"


def get_path(config: Config):
    output = (
        output_path
        / config.backend
        / f"{config.discovery_model.replace(':', '_')}_{config.extraction_model.replace(':', '_')}"
    )
    os.makedirs(output, exist_ok=True)
    return output


async def get_text():

    pdf_path = output_path / "HumanRights.pdf"
    md_path = output_path / "HumanRights.md"
    if not md_path.exists():
        metadata, content = await fetch_url("https://knwler.com/pdfs/HumanRights.pdf")
        pdf_path.write_bytes(content)

        text, metadata = await parse_file(pdf_path)
        md_path.write_text(text, encoding="utf-8")
    else:
        text = md_path.read_text(encoding="utf-8")
    return text


async def crunch(config: Config):
    text = await get_text()
    result = {
        "backend": config.backend,
        "extraction_model": config.extraction_model,
        "discovery_model": config.discovery_model,
        "max_tokens": config.max_tokens,
        "overlap_tokens": config.overlap_tokens,
        "max_concurrent": config.max_concurrent,
        "num_predict": config.num_predict,
        "error": None,
        "schema_time": None,
        "rephrase_time": None,
        "extraction_time": None,
        "consolidation_time": None,
        "num_entities": None,
        "num_relations": None,
    }
    try:
        output = get_path(config)
        # if processed before, we return the stats
        if (output / "stats.json").exists():
            stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
            console.print(
                "Benchmark already exists for this config. Returning existing stats."
            )
            return stats

        # Schema inference
        start_time = time.perf_counter()
        schema = await infer_schema(text, config)
        schema_time = time.perf_counter() - start_time
        result["schema_time"] = round(schema_time, 2)
        schema_path = output / "schema.json"
        schema_path.write_text(json.dumps(asdict(schema), indent=2), encoding="utf-8")

        # Chunking is not timed but necessary for the pipeline to run end-to-end
        chunks = await chunk(text, config)
        chunks_path = output / "chunks.json"
        chunks_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

        # Rephrase chunks
        start_time = time.perf_counter()
        rephrased_chunks = await rephrase_chunks(chunks, config)
        rephrase_time = time.perf_counter() - start_time
        result["rephrase_time"] = round(rephrase_time, 2)
        rephrased_chunks_path = output / "rephrased_chunks.json"
        rephrased_chunks_path.write_text(
            json.dumps(rephrased_chunks, indent=2), encoding="utf-8"
        )

        # Extraction
        start_time = time.perf_counter()
        extraction_results = await extract_chunks(chunks, schema, config)
        extraction_time = time.perf_counter() - start_time
        result["extraction_time"] = round(extraction_time, 2)
        extraction_path = output / "extraction.json"
        extraction_path.write_text(
            json.dumps([asdict(result) for result in extraction_results], indent=2),
            encoding="utf-8",
        )

        # Consolidation
        start_time = time.perf_counter()
        consolidated_graph = await consolidate(
            extraction_results,
            summarize=True,
            filter_low_importance=False,
            config=config,
        )
        consolidation_time = time.perf_counter() - start_time
        result["num_entities"] = len(consolidated_graph.entities)
        result["num_relations"] = len(consolidated_graph.relations)
        result["consolidation_time"] = round(consolidation_time, 2)
        consolidated_path = output / "consolidated_graph.json"
        consolidated_path.write_text(
            json.dumps(asdict(consolidated_graph), indent=2), encoding="utf-8"
        )
        stats_path = output / "stats.json"
        stats_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as e:
        result["error"] = str(e)
    return result


async def run(grid=None):
    if grid is None:
        # feel free to use whatever you think should be benchmarked in the `grid.json` file
        grid_path = root_path / "benchmark" / "grid.json"
        if not grid_path.exists():
            console.print(f"[red]Error: grid.json not found at {grid_path}[/red]")
            return
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
    results = []
    for backend, configs in grid.items():
        for cfg in configs:
            config = Config(
                backend=backend,
                discovery_model=cfg["discovery_model"],
                extraction_model=cfg["extraction_model"],
                use_cache=True,  # if nothing has changed you can rely on the cache for faster iterations, but for benchmarking we want to disable it
            )
            console.print(
                f"[green]Running benchmark with config:[/green] {config.backend} - discovery={config.discovery_model}, extraction={config.extraction_model}"
            )
            result = await crunch(config)
            results.append(result)

    results_path = output_path / f"benchmark_results_{int(time.time())}.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(
        f"[green]Benchmark completed. Results saved to {results_path}[/green]"
    )

    # Generate the HTML report from the just-collected results and open it
    report_path = output_path / f"{results_path.stem}_report.html"
    console.print("[cyan]Generating benchmark report…[/cyan]")
    generate_report(results, report_path=report_path, open_browser=True)
    console.print(f"[green]Report opened → {report_path}[/green]")


def main():

    asyncio.run(run())


if __name__ == "__main__":
    main()
