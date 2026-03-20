"""
Extraction statistics computation and display.
"""

from statistics import mean, median

from rich.table import Table

from knwler.config import console
from knwler.models import ExtractionResult, Schema, Graph, ClusteredGraph

from dataclasses import asdict


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def compute_stats(
    results: list[ExtractionResult],
    schema_time: float,
    wall_time: float,
    consolidation_time: float = 0.0,
) -> dict:
    """Compute extraction statistics."""
    times = [r.chunk_time for r in results]
    tokens = [r.chunk_tokens for r in results]
    entities = [r.entities_count for r in results]
    relations = [r.relations_count for r in results]

    total_time = sum(times)
    total_tokens = sum(tokens)

    return {
        "num_chunks": len(results),
        "schema_discovery_time": round(schema_time, 2),
        "extraction_wall_time": round(wall_time, 2),
        "consolidation_time": round(consolidation_time, 2),
        "total_cpu_time": round(total_time, 2),
        "total_time": round(schema_time + wall_time + consolidation_time, 2),
        "parallelism": round(total_time / wall_time, 2) if wall_time > 0 else 0,
        "throughput_tps": round(total_tokens / wall_time, 1) if wall_time > 0 else 0,
        "time": {
            "min": round(min(times), 2),
            "max": round(max(times), 2),
            "avg": round(mean(times), 2),
            "median": round(median(times), 2),
        },
        "tokens": {
            "min": min(tokens),
            "max": max(tokens),
            "avg": round(mean(tokens)),
            "total": total_tokens,
        },
        "entities": {"total": sum(entities), "avg": round(mean(entities), 1)},
        "relations": {"total": sum(relations), "avg": round(mean(relations), 1)},
    }


def compute_community_stats(consolidated: dict | ClusteredGraph) -> dict:
    """Compute community detection statistics."""
    if isinstance(consolidated, ClusteredGraph):
        consolidated = asdict(consolidated)  # type: ignore
    communities = consolidated.get("communities", [])
    sizes = [len(c.get("members", [])) for c in communities]

    if not sizes:
        return {
            "count": 0,
            "singletons": 0,
            "largest": 0,
            "size": {"min": 0, "max": 0, "avg": 0.0, "median": 0.0},
        }

    return {
        "count": len(sizes),
        "singletons": sum(1 for s in sizes if s == 1),
        "largest": max(sizes),
        "size": {
            "min": min(sizes),
            "max": max(sizes),
            "avg": round(mean(sizes), 1),
            "median": round(median(sizes), 1),
        },
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_stats(
    stats: dict,
    schema: Schema,
    consolidated: dict | ClusteredGraph | None = None,
    _console=None,
):
    """Print formatted statistics using rich tables."""
    if _console is None:
        _console = console
    _console.print()

    table = Table(
        title="Extraction Results", show_header=True, header_style="bold cyan"
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row(
        "Schema",
        f"{len(schema.entity_types)} entity types, "
        f"{len(schema.relation_types)} relation types",
    )
    table.add_row("Discovery time", f"{stats['schema_discovery_time']:.2f}s")
    table.add_row("Chunks", str(stats["num_chunks"]))
    table.add_row("Extraction time", f"{stats['extraction_wall_time']:.2f}s")
    table.add_row("Consolidation time", f"{stats['consolidation_time']:.2f}s")
    if "community_detection_time" in stats:
        table.add_row("Community time", f"{stats['community_detection_time']:.2f}s")
    table.add_row("CPU time", f"{stats['total_cpu_time']:.2f}s")
    table.add_row("Parallelism", f"[green]{stats['parallelism']:.1f}x[/green]")

    t = stats["time"]
    table.add_row(
        "Time/chunk",
        f"min={t['min']:.2f}s, max={t['max']:.2f}s, avg={t['avg']:.2f}s",
    )

    tk = stats["tokens"]
    table.add_row("Tokens/chunk", f"min={tk['min']}, max={tk['max']}, avg={tk['avg']}")
    table.add_row(
        "Throughput", f"[green]{stats['throughput_tps']:.1f}[/green] tokens/sec"
    )

    e, r = stats["entities"], stats["relations"]
    table.add_row("Entities", f"{e['total']} total ({e['avg']:.1f}/chunk)")
    table.add_row("Relations", f"{r['total']} total ({r['avg']:.1f}/chunk)")

    if consolidated:
        if isinstance(consolidated, ClusteredGraph):
            consolidated = asdict(consolidated)  # type: ignore
        table.add_row(
            "Consolidated",
            f"[bold green]{len(consolidated['entities'])}[/bold green] entities, "
            f"[bold green]{len(consolidated['relations'])}[/bold green] relations",
        )
        if "communities" in consolidated:
            cstats = stats.get("communities", {})
            size = cstats.get("size", {})
            table.add_row(
                "Communities",
                f"{cstats.get('count', 0)} total, "
                f"{cstats.get('singletons', 0)} singletons",
            )
            table.add_row(
                "Community size",
                f"min={size.get('min', 0)}, max={size.get('max', 0)}, "
                f"avg={size.get('avg', 0):.1f}, median={size.get('median', 0):.1f}",
            )

    table.add_row("", "")
    table.add_row(
        "[bold]TOTAL TIME[/bold]",
        f"[bold cyan]{stats['total_time']:.2f}s[/bold cyan]",
    )

    _console.print(table)
