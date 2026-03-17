"""
Benchmark report generator.

Loads all benchmark_results_*.json files from the output directory (or a
specific file passed on the command line), computes the Knowledge Yield Score
(KYS), and renders a self-contained HTML report via Jinja2.

Usage
-----
    python benchmark/report.py                          # latest results file
    python benchmark/report.py path/to/results.json    # specific file
    python benchmark/report.py --all                   # merge all result files
"""

from __future__ import annotations

import json
import math
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
TEMPLATE_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT / "benchmark" / "output"


# ── Backend colour palette (Chart.js rgba) ─────────────────────────────────
BACKEND_COLORS = {
    "ollama": "rgba(38, 198, 218, 0.75)",
    "openai": "rgba(63, 81, 181, 0.75)",
    "anthropic": "rgba(239, 83, 80, 0.75)",
}
DEFAULT_COLOR = "rgba(150, 150, 150, 0.75)"


# ── Composite metric ─────────────────────────────────────────────────────────
def compute_kys(results: list[dict]) -> list[dict]:
    """
    Knowledge Yield Score (KYS) — geometric mean of two normalised sub-scores:

        graph_size   = num_entities + num_relations
        total_time   = schema_time + rephrase_time + extraction_time + consolidation_time

        quality_norm = graph_size / max(graph_size)          → richness of output  [0,1]
        speed_norm   = min(total_time) / total_time          → pipeline efficiency [0,1]
        KYS          = sqrt(quality_norm × speed_norm)       → geometric mean

    A high KYS requires *both* a large knowledge graph *and* a fast pipeline.
    Pure throughput (knowledge_rate = graph_size / total_time) is also included
    as a complementary absolute metric.
    """
    valid = [r for r in results if r.get("error") is None] or []
    if not valid:
        print("No valid benchmark runs found in the results.")
        return []

    max_graph = max(r["num_entities"] + r["num_relations"] for r in valid)
    min_time = min(
        r["schema_time"]
        + r["rephrase_time"]
        + r["extraction_time"]
        + r["consolidation_time"]
        for r in valid
    )

    enriched = []
    for r in valid:
        graph_size = r["num_entities"] + r["num_relations"]
        total_time = (
            r["schema_time"]
            + r["rephrase_time"]
            + r["extraction_time"]
            + r["consolidation_time"]
        )

        quality_norm = graph_size / max_graph if max_graph > 0 else 0.0
        speed_norm = min_time / total_time if total_time > 0 else 0.0
        kys = math.sqrt(quality_norm * speed_norm)

        enriched.append(
            {
                **r,
                "graph_size": graph_size,
                "total_time": round(total_time, 2),
                "quality_norm": round(quality_norm, 4),
                "speed_norm": round(speed_norm, 4),
                "kys": round(kys, 4),
                "knowledge_rate": (
                    round(graph_size / total_time, 4) if total_time > 0 else 0.0
                ),
            }
        )

    # Sort by KYS descending (ties broken by knowledge_rate)
    enriched.sort(key=lambda x: (x["kys"], x["knowledge_rate"]), reverse=True)
    return enriched


# ── Load results ─────────────────────────────────────────────────────────────
def load_results(path: Path | None = None, merge_all: bool = False) -> list[dict]:
    if merge_all:
        files = sorted(OUTPUT_DIR.glob("benchmark_results_*.json"))
        if not files:
            raise FileNotFoundError(
                f"No benchmark_results_*.json found in {OUTPUT_DIR}"
            )
        seen = set()
        merged: list[dict] = []
        for f in files:
            for entry in json.loads(f.read_text(encoding="utf-8")):
                key = (
                    entry["backend"],
                    entry["discovery_model"],
                    entry["extraction_model"],
                )
                if key not in seen:
                    seen.add(key)
                    merged.append(entry)
        return merged

    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    # Default: most recent results file
    files = sorted(OUTPUT_DIR.glob("benchmark_results_*.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark_results_*.json found in {OUTPUT_DIR}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


# ── Render ────────────────────────────────────────────────────────────────────
def render_report(runs: list[dict], output_path: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("report_template.html")

    # Per-chart arrays (already sorted by KYS)
    def label(r: dict) -> str:
        return f"{r['discovery_model']} / {r['extraction_model']}"

    chart_labels = [label(r) for r in runs]
    chart_kys = [r["kys"] for r in runs]
    chart_schema_times = [r["schema_time"] for r in runs]
    chart_rephrase_times = [r["rephrase_time"] for r in runs]
    chart_extraction_times = [r["extraction_time"] for r in runs]
    chart_consolidation_times = [r["consolidation_time"] for r in runs]
    chart_entities = [r["num_entities"] for r in runs]
    chart_relations = [r["num_relations"] for r in runs]
    chart_rates = [r["knowledge_rate"] for r in runs]
    chart_backend_colors = [
        BACKEND_COLORS.get(r["backend"], DEFAULT_COLOR) for r in runs
    ]

    ctx = {
        "runs": runs,
        "backends": sorted({r["backend"] for r in runs}),
        "document_name": "HumanRights.pdf",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "avg_knowledge_rate": (
            round(sum(r["knowledge_rate"] for r in runs) / len(runs), 2)
            if runs
            else 0.0
        ),
        "total_entities": sum(r["num_entities"] for r in runs) if runs else 0,
        "max_graph_size": max(r["graph_size"] for r in runs) if runs else 0,
        "min_total_time": min(r["total_time"] for r in runs) if runs else 0,
        # chart data
        "chart_labels": chart_labels,
        "chart_kys": chart_kys,
        "chart_schema_times": chart_schema_times,
        "chart_rephrase_times": chart_rephrase_times,
        "chart_extraction_times": chart_extraction_times,
        "chart_consolidation_times": chart_consolidation_times,
        "chart_entities": chart_entities,
        "chart_relations": chart_relations,
        "chart_rates": chart_rates,
        "chart_backend_colors": chart_backend_colors,
    }

    html = template.render(**ctx)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved → {output_path}")


# ── Public API for programmatic use ─────────────────────────────────────────
def generate_report(
    results: list[dict],
    report_path: Path | None = None,
    open_browser: bool = True,
) -> Path:
    """Compute KYS, render the HTML report, and optionally open it in a browser.

    Parameters
    ----------
    results:
        Raw benchmark result dicts (as produced by ``benchmark/run.py``).
    report_path:
        Destination for the HTML file.  Defaults to
        ``benchmark/output/benchmark_report.html``.
    open_browser:
        When *True* the report is opened in the system default browser.

    Returns
    -------
    Path
        Absolute path to the written HTML file.
    """
    if not results or all(r.get("error") for r in results):
        print("No benchmark results provided for report generation.")
        return Path()
    runs = compute_kys(results)
    dest = report_path or (OUTPUT_DIR / "benchmark_report.html")
    render_report(runs, dest)
    if open_browser:
        webbrowser.open(dest.as_uri())
    return dest


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    args = sys.argv[1:]
    merge_all = "--all" in args
    path_arg = next((a for a in args if not a.startswith("--")), None)
    source_path = Path(path_arg) if path_arg else None

    raw = load_results(path=source_path, merge_all=merge_all)

    report_name = "benchmark_report.html"
    if source_path:
        report_name = source_path.stem + "_report.html"

    generate_report(raw, report_path=OUTPUT_DIR / report_name, open_browser=True)


if __name__ == "__main__":
    main()
