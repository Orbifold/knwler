import re
import tempfile
import typer
from rich.panel import Panel
from rich.padding import Padding
import time
import asyncio
import networkx as nx

from pathlib import Path
import json
from uuid import uuid4
from typing import Annotated, Optional
import fitz  # pymupdf4llm
from knwler.collect.webpage import WebpageCollector

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
from knwler.cache import CACHE_DIR, hash_args
from knwler.chunking import chunk_text
from knwler.clustering import cluster_graph, create_network
from knwler.consolidation import consolidate_extracted_graphs, consolidate_graphs
from knwler.discovery import detect_language, discover_schema
from knwler.language import (
    DEFAULT_LANGUAGE,
    get_console_msg,
    get_lang,
    set_language,
    get_current_language,
)
from knwler.export import export_html
from knwler.extraction import extract_all
from knwler.extras import extract_summary, extract_title, rephrase_chunks
from knwler.stats import compute_community_stats, compute_stats, print_stats
from knwler.config import (
    DEFAULT_OLLAMA_EXTRACTION_MODEL,
    DEFAULT_OLLAMA_DISCOVERY_MODEL,
    DEFAULT_OPENAI_EXTRACTION_MODEL,
    DEFAULT_OPENAI_DISCOVERY_MODEL,
    DEFAULT_ANTHROPIC_EXTRACTION_MODEL,
    DEFAULT_ANTHROPIC_DISCOVERY_MODEL,
    DEFAULT_GITHUB_EXTRACTION_MODEL,
    DEFAULT_GITHUB_DISCOVERY_MODEL,
    Config,
    console,
)
from knwler.models import ExtractionResult, Schema, Graph
from knwler.cli_consolidate import cli_consolidate_graphs
from dataclasses import asdict

extract_app = typer.Typer(help="Utility to manage documents.")


async def _process_file(
    file_path: Path,
    *,
    output: Optional[Path],
    config: "Config",
    no_discovery: bool,
    language: Optional[str],
    url: Optional[str],
    html_report: bool,
    gml_export: bool,
    template: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Run the full extraction pipeline on a single file."""
    # Determine this file's results directory
    if output is None:
        results_dir = Path(time.strftime("results/%Y%m%d-%H%M%S"))
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        if output.is_file():
            raise ValueError(f"Output path must be a directory, not a file: {output}")
        results_dir = output
        if overwrite:
            if results_dir.exists():
                for item in results_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    else:
                        shutil.rmtree(item)
            else:
                results_dir.mkdir(parents=True, exist_ok=True)
        else:
            results_dir.mkdir(parents=True, exist_ok=True)

    # Load text from PDF or plain text file
    if file_path.suffix.lower() == ".pdf":
        # copy the pdf to the results directory for reference and to ensure consistent caching based on file path
        cached_pdf_path = results_dir / file_path.name
        if not cached_pdf_path.exists():
            cached_pdf_path.write_bytes(file_path.read_bytes())
        extracted_text_path = results_dir / f"{file_path.stem}_extracted.txt"
        if extracted_text_path.exists():
            console.print(
                f"[green]\u2713[/green] Using cached extracted text: {extracted_text_path}"
            )
            text = extracted_text_path.read_text(encoding="utf-8", errors="ignore")
        else:
            console.print(f"[yellow]Extracting text from PDF: {file_path}[/yellow]")
            doc = fitz.open(file_path)
            text = "\n\n".join(page.get_text() for page in doc)
            extracted_text_path.write_text(text, encoding="utf-8", errors="ignore")
    else:
        text = file_path.read_text(encoding="utf-8", errors="ignore")

    # Load existing graph if output file already exists (augment mode)
    existing_data = None
    existing_result = None
    graph_json_path = results_dir / "graph.json"
    if graph_json_path.exists():
        existing_data = json.loads(graph_json_path.read_text())
        consolidated_ents = existing_data.get("graph", {}).get("entities", [])
        consolidated_rels = existing_data.get("graph", {}).get("relations", [])
        existing_result = ExtractionResult(
            entities=consolidated_ents,
            relations=consolidated_rels,
            id=existing_data.get("id", str(uuid4())),
            chunk_idx=-1,
            chunk_time=0.0,
            chunk_tokens=0,
        )

    # Language detection / setting
    if language:
        set_language(language)
        detected_lang = language
    else:
        detected_lang = await detect_language(text, config)
        set_language(detected_lang)

    lang_name = get_lang().get("name", detected_lang)

    # Show header
    _backend_labels = {
        "openai": "[cyan]OpenAI[/cyan]",
        "anthropic": "[magenta]Anthropic[/magenta]",
        "ollama": "[green]Ollama[/green]",
    }
    backend = _backend_labels.get(config.backend, "[green]Ollama[/green]")
    mode = (
        f"  \u2022  [yellow]Augmenting[/yellow]: [dim]{results_dir.name}[/dim]"
        if existing_data
        else ""
    )
    console.print(
        Panel.fit(
            f"[bold]Knwler Graph Extraction Pipeline[/bold]\n"
            f"Backend: {backend}  \u2022  File: [dim]{file_path.name}[/dim]  "
            f"\u2022  Language: [magenta]{lang_name}[/magenta]{mode}",
            border_style="blue",
        )
    )

    # Cache status
    if config.use_cache:
        cache_count = len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0
        console.print(f"[dim]Cache: {CACHE_DIR} ({cache_count} entries)[/dim]")
    else:
        console.print("[dim]Cache: disabled[/dim]")

    # ── Schema discovery ──
    console.print()
    console.rule("[bold cyan]Schema Discovery[/bold cyan]")
    console.print(f"Discovery model: [green]{config.discovery_model}[/green]")

    if no_discovery:
        schema = Schema.default()
        console.print("[yellow]Skipped (using defaults)[/yellow]")
    else:
        discovering_msg = (
            get_console_msg("discovering_schema") or "Discovering schema..."
        )
        with console.status(f"[bold green]{discovering_msg}[/bold green]"):
            schema = await discover_schema(text, config)
        console.print(f"Time: [cyan]{schema.discovery_time:.2f}s[/cyan]")

    # Merge existing schema types
    if existing_data and "schema" in existing_data:
        old_schema = existing_data["schema"]
        for et in old_schema.get("entity_types", []):
            if et.lower() not in {t.lower() for t in schema.entity_types}:
                schema.entity_types.append(et)
        for rt in old_schema.get("relation_types", []):
            if rt.lower() not in {t.lower() for t in schema.relation_types}:
                schema.relation_types.append(rt)

    console.print(f"Entity types: [green]{', '.join(schema.entity_types)}[/green]")
    console.print(f"Relation types: [green]{', '.join(schema.relation_types)}[/green]")

    # ── Chunking ──
    console.print()
    console.rule("[bold cyan]Text Chunking[/bold cyan]")
    chunks = chunk_text(text, config)
    console.print(
        f"\nChunks: [cyan]{len(chunks)}[/cyan] (~{config.max_tokens} tokens each)"
    )

    # ── Title ──
    console.print()
    console.rule("[bold cyan]Title Extraction[/bold cyan]")
    title = await extract_title(chunks, config) or file_path.stem
    console.print(f"Title: [green]{title}[/green]")

    # ── Summary ──
    console.print()
    console.rule("[bold cyan]Document Summary[/bold cyan]")
    summary = await extract_summary(chunks, config)
    if summary:
        console.print(f"Summary: [green]{summary}[/green]")
    else:
        console.print("[yellow]Summary not available[/yellow]")

    # ── Rephrase ──
    console.print()
    console.rule("[bold cyan]Chunk Rephrase[/bold cyan]")
    console.print(f"Rephrase model: [green]{config.extraction_model}[/green]")
    rephrase_t0 = time.perf_counter()
    rephrased = await rephrase_chunks(chunks, config)
    rephrase_time = time.perf_counter() - rephrase_t0
    console.print(f"Time: [cyan]{rephrase_time:.2f}s[/cyan]")

    # ── Extraction ──
    console.print()
    console.rule("[bold cyan]Extraction[/bold cyan]")
    console.print(f"Extraction model: [green]{config.extraction_model}[/green]")

    t0 = time.perf_counter()
    extraction_results: list[ExtractionResult] = await extract_all(
        chunks, schema, config, output_path=output
    )
    wall_time = time.perf_counter() - t0

    # ── Consolidation ──
    console.print()
    console.rule("[bold cyan]Consolidation[/bold cyan]")
    all_results = (
        ([existing_result] + extraction_results)
        if existing_result
        else extraction_results
    )
    consolidated, consolidation_time = await consolidate_extracted_graphs(
        all_results, config, summarize=True
    )

    # ── Community analysis ──
    console.print()
    console.rule("[bold cyan]Community Analysis[/bold cyan]")
    community_t0 = time.perf_counter()
    consolidated = await cluster_graph(consolidated, config)
    community_time = time.perf_counter() - community_t0

    # ── Stats ──
    stats = compute_stats(
        extraction_results, schema.discovery_time, wall_time, consolidation_time
    )
    stats["community_detection_time"] = round(community_time, 2)
    stats["communities"] = compute_community_stats(consolidated)
    print_stats(stats, schema, consolidated)

    # ── Save results ──
    if existing_data and "schema" in existing_data:
        old_schema = existing_data["schema"]
        entity_types = list(schema.entity_types)
        relation_types = list(schema.relation_types)

        for et in old_schema.get("entity_types", []):
            if et.lower() not in {t.lower() for t in entity_types}:
                entity_types.append(et)
        for rt in old_schema.get("relation_types", []):
            if rt.lower() not in {t.lower() for t in relation_types}:
                relation_types.append(rt)

        old_reasoning = old_schema.get("reasoning", "")
        if old_reasoning and old_reasoning != schema.reasoning:
            reasoning = f"{old_reasoning} | {schema.reasoning}".strip(" |")
        else:
            reasoning = schema.reasoning or old_reasoning
    else:
        entity_types = schema.entity_types
        relation_types = schema.relation_types
        reasoning = schema.reasoning

    stats_entry = dict(stats)
    stats_entry["run"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file": str(file_path),
        "backend": config.backend,
        "extraction_model": config.extraction_model,
        "discovery_model": config.discovery_model,
        "max_tokens": config.max_tokens,
        "overlap_tokens": config.overlap_tokens,
        "max_concurrent": config.max_concurrent,
        "use_cache": config.use_cache,
        "no_discovery": no_discovery,
    }

    if existing_data and "stats" in existing_data:
        prior_stats = existing_data.get("stats")
        if isinstance(prior_stats, list):
            stats_list = prior_stats + [stats_entry]
        else:
            stats_list = [prior_stats, stats_entry]
    else:
        stats_list = [stats_entry]
    document_id = str(uuid4())  # hash_args(title, url)
    final_chunks = (existing_data.get("chunks", []) if existing_data else []) + [
        {
            "id": r.id,
            "chunk_idx": r.chunk_idx,
            "text": chunks[r.chunk_idx],
            "rephrase": (
                rephrased[r.chunk_idx] if r.chunk_idx < len(rephrased) else ""
            ),
            "entities": r.entities,
            "relations": r.relations,
            "chunk_time": r.chunk_time,
            "chunk_tokens": r.chunk_tokens,
            "document": document_id,
        }
        for r in extraction_results
    ]
    # note that communities are not unique and can span multiple chunks, so we won't assign them to specific chunks
    output_data = {
        "id": document_id,
        "title": title,
        "summary": summary,
        "url": url,
        "language": detected_lang,
        "schema": {
            "entity_types": entity_types,
            "relation_types": relation_types,
            "reasoning": reasoning,
        },
        "stats": stats_list,
        "graph": (
            asdict(consolidated) if isinstance(consolidated, Graph) else consolidated
        ),
        "chunks": final_chunks,
    }
    graph_json_path.write_text(json.dumps(output_data, indent=2))
    console.print(
        f"\n[green]\u2713[/green] Results saved to [cyan]{graph_json_path}[/cyan]"
    )

    # GML export
    if gml_export:
        g = create_network(
            consolidated, title=title, url=url, language=lang_name, summary=summary
        )
        nx.write_gml(g, results_dir / "graph.gml")
        console.print(
            f"[green]\u2713[/green] Graph saved to "
            f"[cyan]{results_dir / 'graph.gml'}[/cyan]"
        )

    # HTML export
    if html_report:
        html_content = export_html(
            output_data,
            template=template,
        )
        html_path = results_dir / "index.html"
        html_path.write_text(html_content)
        console.print(
            f"[green]\u2713[/green] HTML report saved to [cyan]{html_path}[/cyan]"
        )

    # Save console log
    log_path = results_dir / "log.txt"
    console.save_text(str(log_path), clear=False)
    console.print(f"[green]\u2713[/green] Console log saved to [cyan]{log_path}[/cyan]")
    log_path = results_dir / "log.html"
    console.save_html(str(log_path), clear=False)
    console.print(f"[green]\u2713[/green] Console log saved to [cyan]{log_path}[/cyan]")

    # Rename results directory to include the title
    if not output:
        if title and len(title) > 30:
            new_dir_name = "_".join(title.split(" ")[:3])
        else:
            new_dir_name = title[:30]
        # sanitize directory name
        new_dir_name = re.sub(r"[^\w\- ]", "", new_dir_name).strip().replace(" ", "_")
        new_dir_path = results_dir.parent / new_dir_name
        if new_dir_path.exists():
            if overwrite:
                console.print(
                    f"[yellow]\u26a0 Overwriting existing directory: "
                    f"{new_dir_path}[/yellow]"
                )
                if new_dir_path.is_dir():
                    for item in new_dir_path.iterdir():
                        if item.is_file():
                            item.unlink()
                        else:
                            import shutil

                            shutil.rmtree(item)
            else:
                new_dir_path = results_dir.parent / f"{new_dir_name}_{int(time.time())}"
        results_dir.rename(new_dir_path)
        console.print(
            f"[green]\u2713[/green] Results directory renamed to "
            f"[cyan]{new_dir_path}[/cyan]"
        )


@extract_app.command("extract")
def extract(
    ctx: typer.Context,
    file: Annotated[
        Optional[str],
        typer.Option(
            "--file",
            "-f",
            help="Path to a text or pdf file, or a url (http/https), to extract from. If a URL is provided, the content will be fetched and extracted.",
        ),
    ] = None,
    directory: Annotated[
        Optional[Path],
        typer.Option(
            "--dir",
            "-D",
            help="Path to a directory containing text or pdf files to extract from. All files in the directory will be extracted.",
        ),
    ] = None,
    extraction_model: Annotated[
        Optional[str],
        typer.Option(
            "--extraction-model",
            "-e",
            help=(
                "Model for summary, title and graph extraction. Applies to all backends. "
                f"Defaults: Ollama={DEFAULT_OLLAMA_EXTRACTION_MODEL}, "
                f"OpenAI={DEFAULT_OPENAI_EXTRACTION_MODEL}. "
                f"Anthropic={DEFAULT_ANTHROPIC_EXTRACTION_MODEL}. "
            ),
        ),
    ] = None,
    discovery_model: Annotated[
        Optional[str],
        typer.Option(
            "--discovery-model",
            "-d",
            help=(
                "Model for schema and language discovery. Applies to all backends. "
                f"Defaults: Ollama={DEFAULT_OLLAMA_DISCOVERY_MODEL}, "
                f"OpenAI={DEFAULT_OPENAI_DISCOVERY_MODEL}. "
                f"Anthropic={DEFAULT_ANTHROPIC_DISCOVERY_MODEL}. "
            ),
        ),
    ] = None,
    concurrent: Annotated[
        int,
        typer.Option(
            "--concurrent", "-c", help="Max LLM concurrent requests, default: 10"
        ),
    ] = 10,
    no_discovery: Annotated[
        bool,
        typer.Option(
            "--no-discovery",
            help="Skip schema discovery via LLM and use the default (shallow) schema",
        ),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable LLM response caching, default: false"),
    ] = False,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            "-b",
            help="LLM backend to use: ollama (default), openai, anthropic, github.",
        ),
    ] = "ollama",
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Base URL of the LLM backend. Applies to all backends.",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Directory to save results (defaults to timestamped dir in results/)",
        ),
    ] = None,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Max tokens per chunk, default: 400"),
    ] = 400,
    html_report: Annotated[
        bool,
        typer.Option("--html-report", help="Also export an HTML report, default: true"),
    ] = True,
    gml_export: Annotated[
        bool,
        typer.Option(
            "--gml-export", help="Also export a GML graph file, default: true"
        ),
    ] = True,
    html_only: Annotated[
        bool,
        typer.Option(
            "--html-only",
            help="Only export HTML from existing an existing graph JSON file, default: false",
        ),
    ] = False,
    url: Annotated[
        Optional[str],
        typer.Option("--url", "-u", help="Origin of the document (for metadata only)"),
    ] = None,
    language: Annotated[
        Optional[str],
        typer.Option(
            "--language",
            "-l",
            help="Language to use (e.g., en, de, fr, es, nl). Auto-detects if not specified. Default: auto-detect",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Allow overwriting the output directory if it already exists (default: false)",
        ),
    ] = False,
    consolidate: Annotated[
        bool,
        typer.Option(
            "--consolidate",
            help="Whether to consolidate extracted graphs when multiple documents are processed (default: false)",
        ),
    ] = False,
    template: Annotated[
        Optional[str],
        typer.Option(
            "--template",
            help="HTML template to use for the consolidated graph report (defaults to 'columns').",
        ),
    ] = "default",
):
    """Extract knowledge graphs from text using LLMs."""
    if html_only:
        if output is None:
            raise ValueError(
                "--html-only requires --output inside which to find the existing 'graph.json' and save the HTML report"
            )
        # allow a speficic file or a directory containing 'graph.json'
        if output.is_file():
            # has to be a json file containing the graph data
            if output.suffix.lower() != ".json":
                raise ValueError(
                    f"--html-only output file must be a .json file containing the graph data: {output}"
                )
            graph_json_path = output
        else:
            output.mkdir(parents=True, exist_ok=True)
            graph_json_path = output / "graph.json"
        if not graph_json_path.exists():
            raise FileNotFoundError(f"Missing results file: {graph_json_path}")
        results_data = json.loads(graph_json_path.read_text())
        saved_lang = results_data.get("language", DEFAULT_LANGUAGE)
        set_language(saved_lang)
        html_path = export_html(
            results_data,
            graph_json_path.parent / "index.html",
            title=graph_json_path.parent.stem,
        )
        console.print(
            f"[green]\u2713[/green] HTML report saved to [cyan]{html_path}[/cyan]"
        )
        return

    # ── URL handling ──
    _tmp_file_path: Optional[Path] = None
    if file is not None and _URL_RE.match(file):
        fetched_url = file
        url = url or fetched_url
        result = asyncio.run(WebpageCollector.fetch_url(fetched_url, no_cache=no_cache))
        if result is None:
            typer.echo(f"Error: failed to fetch URL: {fetched_url}")
            return typer.Exit(1)
        _metadata, content = result
        suffix = ".pdf" if fetched_url.lower().endswith(".pdf") else ".md"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
            tmp.write(
                content if isinstance(content, bytes) else content.encode("utf-8")
            )
            _tmp_file_path = Path(tmp.name)
        file = _tmp_file_path  # type: ignore[assignment]
    elif file is not None:
        file = Path(file)  # type: ignore[assignment]

    if file is None and directory is None:
        typer.echo(ctx.get_help())
        return typer.Exit(1)

    # Config
    _valid_backends = {"ollama", "openai", "anthropic", "github"}
    backend = backend.lower()
    if backend not in _valid_backends:
        typer.echo(
            f"Error: unknown backend '{backend}'. Choose from: {', '.join(sorted(_valid_backends))}."
        )
        return typer.Exit(1)
    _default_extraction = {
        "openai": DEFAULT_OPENAI_EXTRACTION_MODEL,
        "anthropic": DEFAULT_ANTHROPIC_EXTRACTION_MODEL,
        "github": DEFAULT_GITHUB_EXTRACTION_MODEL,
        "ollama": DEFAULT_OLLAMA_EXTRACTION_MODEL,
    }
    _default_discovery = {
        "openai": DEFAULT_OPENAI_DISCOVERY_MODEL,
        "anthropic": DEFAULT_ANTHROPIC_DISCOVERY_MODEL,
        "github": DEFAULT_GITHUB_DISCOVERY_MODEL,
        "ollama": DEFAULT_OLLAMA_DISCOVERY_MODEL,
    }
    resolved_extraction_model = extraction_model or _default_extraction[backend]
    resolved_discovery = discovery_model or _default_discovery[backend]
    config = Config(
        backend=backend,
        extraction_model=resolved_extraction_model,
        discovery_model=resolved_discovery,
        max_concurrent=concurrent,
        max_tokens=max_tokens,
        use_cache=not no_cache,
        base_url=base_url,
    )

    # Build list of files to process
    if directory is not None:
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        files_to_process = sorted(
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in (".txt", ".pdf", ".md")
        )
        if not files_to_process:
            console.print(
                f"[yellow]No supported files (.txt, .pdf, .md) found in "
                f"{directory}[/yellow]"
            )
            return typer.Exit(0)
    else:
        files_to_process = [file]

    # Process each file
    for fp in files_to_process:
        if directory is not None:
            fp_output = (output / fp.stem) if output is not None else None
            console.rule(f"[bold blue]Processing: {fp.name}[/bold blue]")
        else:
            fp_output = output

        asyncio.run(
            _process_file(
                fp,
                output=fp_output,
                config=config,
                no_discovery=no_discovery,
                language=language,
                url=url,
                html_report=html_report,
                gml_export=gml_export,
                overwrite=overwrite,
                template=template,
            )
        )

        if directory is None:
            break  # single file mode, so stop after first file
        # Directory mode: log error and continue to next file
    if consolidate and len(files_to_process) > 1:
        asyncio.run(
            cli_consolidate_graphs(directory=output, output=output, config=config)
        )
    # Cleanup temp file if we fetched from a URL
    if _tmp_file_path is not None and _tmp_file_path.exists():
        _tmp_file_path.unlink()
        console.print(
            f"[green]\u2713[/green] Cleaned up temporary file: {_tmp_file_path}"
        )
