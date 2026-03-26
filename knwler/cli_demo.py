import asyncio
import json
import os
import webbrowser
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import typer
from rich.markdown import Markdown

from knwler.api import (
    chunk,
    cluster_graph,
    consolidate_chunk_graphs,
    discover_language,
    extract_chunks,
    extract_summary,
    extract_title,
    fetch_url,
    infer_schema,
    parse_file,
    rephrase_chunks,
)
from knwler.config import console, Config
from knwler.export import export_html

demo_app = typer.Typer(help="Demo commands for Knwler.")


async def run_demo(backend: str | None = None):
    # default config, can be customized as needed
    config = Config(backend=backend or "ollama", use_cache=False)
    config_dic = asdict(config)
    console.print(f"Running demo with config:")
    for k, v in config_dic.items():
        console.print(f"\t• [bold]{k}[/bold]: {v}")

    os.makedirs("knwler_demo", exist_ok=True)
    output = Path("knwler_demo")
    pdf_path = output / "HumanRights.pdf"

    # Step 1: take a pdf or some document
    metadata, content = await fetch_url("https://knwler.com/pdfs/HumanRights.pdf")
    pdf_path.write_bytes(content)
    console.print(f"[green]✔ Saved document to[/green] {pdf_path}")

    # Step2: parse the document if it's not a text-based format, and save the parsed text as a .md file
    text, metadata = await parse_file(pdf_path)
    md_path = output / "HumanRights.md"
    md_path.write_text(text, encoding="utf-8")
    console.print(f"[green]✔ Saved parsed text to[/green] {md_path}")

    # Step 3: detect the language of the text
    language = await discover_language(text, config)
    console.print(f"[green]✔ Detected language: {language}[/green]")

    # Step 4: infer a schema
    schema = await infer_schema(text, config)
    schema_path = output / "schema.json"
    schema_path.write_text(json.dumps(asdict(schema), indent=2), encoding="utf-8")
    console.print(f"[green]✔ Discovered schema saved to[/green] {schema_path}")

    # Step 5: chunk the text
    chunks = await chunk(text, config)
    chunks_path = output / "chunks.json"
    chunks_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    console.print(
        f"[green]✔ Text chunked into {len(chunks)} chunks and saved to {chunks_path}[/green]"
    )

    # Step 6: rephrase the chunks for better readability in the UI
    rephrased_chunks = await rephrase_chunks(chunks, config)
    rephrased_chunks_path = output / "rephrased_chunks.json"
    rephrased_chunks_path.write_text(
        json.dumps(rephrased_chunks, indent=2), encoding="utf-8"
    )
    console.print(f"[green]✔ Rephrased chunks saved to[/green] {rephrased_chunks_path}")

    # Step 7: extract according to the discovered schema
    extraction_results = await extract_chunks(chunks, schema, config)

    # Step 8: consolidate the extracted graphs
    consolidated_graph = await consolidate_chunk_graphs(
        extraction_results, summarize=True, filter_low_importance=False, config=config
    )
    console.print(f"[green]✔ Consolidated graph created[/green]")

    # Step 9: analyze clusters
    clustered_graph = await cluster_graph(consolidated_graph, config)
    console.print(f"[green]✔ Cluster analysis completed[/green]")

    graph_path = output / "graph.json"
    graph_path.write_text(
        json.dumps(asdict(clustered_graph), indent=2), encoding="utf-8"
    )

    # Step 10: infer or assign a title
    title = await extract_title(chunks, config)
    console.print(f"[green]✔ Extracted title: {title}[/green]")

    # Step 11: extract a summary
    summary = await extract_summary(chunks, config)
    console.print(f"[green]✔ Extracted summary:[/green]")
    console.print(Markdown(summary))

    # Step 12: save the whole lot

    # for convenience we add the entities found within a chunk
    extended_chunks = []
    document_id = str(uuid4())
    for idx, u in enumerate(extraction_results):
        entities = u.entities
        relations = u.relations
        extended_chunks.append(
            {
                "id": u.id,
                "chunk_idx": idx,
                "text": chunks[idx],
                "rephrase": rephrased_chunks[idx],
                "entities": entities,
                "relations": relations,
                "document": document_id,
            }
        )
    all_data = {
        "id": document_id,
        "url": metadata.get("source_url"),
        "title": title,
        "summary": summary,
        "language": language,
        "schema": asdict(schema),
        "graph": asdict(clustered_graph),
        "chunks": extended_chunks,
    }
    all_data_path = output / "all_data.json"
    all_data_path.write_text(json.dumps(all_data, indent=2), encoding="utf-8")
    console.print(f"[green]✔ All data saved to[/green] {all_data_path}")

    # Step 13: export to HTML
    html_path = output / "index.html"
    html_content = export_html(all_data, template="default")
    html_path.write_text(html_content, encoding="utf-8")
    console.print(f"[green]✔ Exported HTML saved to[/green] {html_path}")
    if os.name == "nt":
        os.startfile(html_path)  # type: ignore[attr-defined]
    else:
        webbrowser.open(html_path.resolve().as_uri())


@demo_app.callback(invoke_without_command=True)
def _app_callback(
    ctx: typer.Context,
    backend: str = typer.Option(
        "ollama", help="LLM backend to use: ollama, openai, gemini or anthropic."
    ),
):
    """
    Called in case no subcommand is given, ie. `knwler demo`.
    """
    if ctx.invoked_subcommand is None:
        asyncio.run(run_demo(backend=backend))
        raise typer.Exit(0)
