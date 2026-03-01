from knwler.consolidation import consolidate_graphs
from knwler.config import Config, console
import json
from typing import Optional
from pathlib import Path
import os


def cli_consolidate_graphs(
    directory: Optional[Path] = None,
    output: Optional[Path] = None,
    config: Optional[Config] = Config(),
):
    if directory and not directory.exists():
        console.print(
            f"[red]\u2717[/red] The specified directory [cyan]{directory}[/cyan] does not exist. Please provide a valid directory path."
        )
        return
    if output is None:
        results_dir = Path("results")
        os.makedirs(results_dir, exist_ok=True)
    else:
        os.makedirs(output, exist_ok=True)
    console.rule("[bold magenta]Consolidating Multiple Graphs[/bold magenta]")
    # search recursively for all graph.json files in the given directory (or results/ by default)
    search_dir = directory if directory else Path("results")
    # loop over dirs in search_dir and find all graph.json files
    files_to_process = []
    for dirpath, dirnames, filenames in os.walk(search_dir):
        for filename in filenames:
            if filename == "graph.json":
                files_to_process.append(Path(dirpath) / filename)
    if not files_to_process:
        console.print(
            f"[yellow]\u26a0 No graph.json files found in [cyan]{search_dir}[/cyan] for consolidation.[/yellow]"
        )
        return

    # Load all individual graph.json files
    all_graphs = []
    for graph_json_path in files_to_process:

        if graph_json_path.exists():
            data = json.loads(graph_json_path.read_text())
            all_graphs.append(data)
        else:
            console.print(
                f"[yellow]\u26a0 Missing graph.json for {graph_json_path.name}, skipping consolidation for this file.[/yellow]"
            )
    console.print(
        f"[green]\u2713[/green] Found [cyan]{len(all_graphs)}[/cyan] graph(s) to consolidate from [cyan]{search_dir}[/cyan]."
    )
    if all_graphs:
        # note that clustering is also applied at the consolidation level to group similar entities together across documents, which is useful when merging a large amount of graphs towards a database ingestion
        consolidated = consolidate_graphs(all_graphs, cluster=True, config=config)
        consolidated_path = (
            (output / "consolidated_graph.json")
            if output
            else (Path("results") / "consolidated_graph.json")
        )
        consolidated_path.write_text(json.dumps(consolidated, indent=2))
        console.print(
            f"[green]\u2713[/green] Consolidated graph saved to [cyan]{consolidated_path}[/cyan]"
        )
    else:
        console.print(
            f"[yellow]\u26a0 No valid graph data found in the graph.json files for consolidation.[/yellow]"
        )
