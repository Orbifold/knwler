"""
Extra extraction utilities: rephrase, title, summary.
"""

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from knwler.config import Config, console, null_console
from knwler.language import get_console_msg, get_prompt
from knwler.llm import llm_generate, parse_json_response
from knwler.models import Chunk


# ---------------------------------------------------------------------------
# Rephrase
# ---------------------------------------------------------------------------
async def rephrase_chunks(
    chunks: list[str], config: Config, _console=None
) -> list[str]:
    """Rephrase each chunk in simple language for UI display."""
    if _console is None:
        _console = console
    rephrased: list[str] = []
    progress_msg = get_console_msg("rephrasing") or "Rephrasing..."

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"[cyan]{progress_msg}", total=len(chunks))

        for chunk in chunks:
            prompt = get_prompt("rephrase_chunk", chunk=chunk)

            if not prompt:
                prompt = (
                    "Rewrite the following text so that it is easy to understand. "
                    "Keep all the key facts and names. Convey the information "
                    "clearly and concisely.\n\n"
                    f'TEXT:\n"""{chunk}"""\n\n'
                    'Return JSON:\n{\n  "rephrase": "..."\n}'
                )
            response = await llm_generate(prompt, config, model=config.extraction_model)
            parsed = parse_json_response(response)
            rephrased.append(parsed.get("rephrase", chunk))
            progress.update(task, advance=1)

    return rephrased


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
async def extract_title(
    chunks: list[str] | list[Chunk], config: Config, max_chunks: int = 3
) -> str:
    """Extract a short document title from the first few chunks."""
    sample = "\n\n".join(
        c.text if isinstance(c, Chunk) else c for c in chunks[:max_chunks]
    )

    prompt = get_prompt("extract_title", sample=sample)
    if not prompt:
        prompt = (
            "Create a short, clear document title based on the text below. "
            "Return only a JSON object.\n\n"
            f'TEXT:\n"""{sample}"""\n\n'
            'Return JSON:\n{\n  "title": "..."\n}'
        )
    response = await llm_generate(prompt, config, model=config.extraction_model)
    parsed = parse_json_response(response)
    return (parsed.get("title") or "").strip()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
async def extract_summary(
    chunks: list[str] | list[Chunk], config: Config, max_chunks: int = 3
) -> str:
    """Summarize the document based on the first few chunks."""
    sample = "\n\n".join(
        c.text if isinstance(c, Chunk) else c for c in chunks[:max_chunks]
    )

    prompt = get_prompt("extract_summary", sample=sample)

    if not prompt:
        prompt = (
            "Summarize the document below in 3-5 sentences. "
            "Focus on the main topic and key points. Return only JSON.\n\n"
            f'TEXT:\n"""{sample}"""\n\n'
            'Return JSON:\n{\n  "summary": "..."\n}'
        )
    response = await llm_generate(prompt, config, model=config.extraction_model)
    parsed = parse_json_response(response)
    return (parsed.get("summary") or "").strip()
