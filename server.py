"""
Knwler FastAPI Server
=====================
Wraps the extraction pipeline from main.py as a local HTTP + WebSocket API.
Designed to be launched by Tauri as a sidecar process.

On startup, finds a free port, prints ``PORT:<num>`` to stdout, then serves
the API on ``127.0.0.1:<port>``.

Uses TaskIQ with InMemoryBroker for background task management.
"""

import asyncio
import json
import socket
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import taskiq_fastapi
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from taskiq import InMemoryBroker

from main import (
    Config,
    load_languages,
    render_report_html,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# TaskIQ broker (in-memory, single-process – perfect for a desktop app)
# ---------------------------------------------------------------------------
broker = InMemoryBroker()
taskiq_fastapi.init(broker, "server:app")

# In-process job state – all keyed by the UUID returned as job_id.
job_progress: dict[str, dict] = {}
job_results: dict[str, dict] = {}
job_errors: dict[str, str] = {}
job_websockets: dict[str, list[WebSocket]] = {}


# ---------------------------------------------------------------------------
# TaskIQ task
# ---------------------------------------------------------------------------
@broker.task
async def extraction_task(
    file_path_str: str,
    config_dict: dict,
    language: str | None,
    no_discovery: bool,
    url: str | None,
    task_id: str,
) -> dict:
    """Run the extraction pipeline as a TaskIQ background task."""
    file_path = Path(file_path_str)

    # Reconstruct Config from dict (TaskIQ serializes arguments)
    config = Config(**config_dict)

    job_progress[task_id] = {"stage": "", "current": 0, "total": 0}

    def sync_progress(stage: str, current: int, total: int):
        """Sync callback called from within the running event loop."""
        job_progress[task_id] = {
            "stage": stage,
            "current": current,
            "total": total, 
        }
        # Schedule the async broadcast without blocking (already in the loop)
        asyncio.ensure_future(_broadcast_progress(task_id, stage, current, total))

    try:
        result = await run_pipeline(
            file_path=file_path,
            config=config,
            language=language,
            no_discovery=no_discovery,
            url=url,
            progress_callback=sync_progress,
        )

        job_results[task_id] = result

        # Notify WebSocket clients of completion
        for ws in list(job_websockets.get(task_id, [])):
            try:
                await ws.send_json({"type": "completed"})
            except Exception:
                pass

        return result

    except Exception as exc:
        job_errors[task_id] = str(exc)
        raise

    finally:
        # Clean up temp file
        file_path.unlink(missing_ok=True)


async def _broadcast_progress(task_id: str, stage: str, current: int, total: int):
    for ws in list(job_websockets.get(task_id, [])):
        try:
            await ws.send_json(
                {"type": "progress", "stage": stage, "current": current, "total": total}
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# App lifespan (starts/stops the broker)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_languages()
    if not broker.is_worker_process:
        await broker.startup()
    yield
    if not broker.is_worker_process:
        await broker.shutdown()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Knwler Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract")
async def start_extraction(
    file: UploadFile = File(...),
    use_openai: bool = Form(False),
    openai_api_key: str = Form(""),
    openai_base_url: str = Form("https://api.openai.com/v1"),
    extraction_model: str = Form(""),
    discovery_model: str = Form(""),
    max_concurrent: int = Form(8),
    max_tokens: int = Form(400),
    no_discovery: bool = Form(False),
    language: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
):
    """Start an extraction job. Returns a task ID immediately."""
    # Save uploaded file to a temp location
    suffix = Path(file.filename or "doc.txt").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    # Build config as a plain dict (serializable for TaskIQ)
    config = Config(
        use_openai=use_openai,
        openai_api_key=openai_api_key or "",
        openai_base_url=openai_base_url,
        max_concurrent=max_concurrent,
        max_tokens=max_tokens,
    )
    if extraction_model:
        if use_openai:
            config.openai_extraction_model = extraction_model
        else:
            config.ollama_extraction_model = extraction_model
    if discovery_model:
        if use_openai:
            config.openai_discovery_model = discovery_model
        else:
            config.ollama_discovery_model = discovery_model

    # Serialize Config to dict (TaskIQ requires JSON-serializable arguments)
    from dataclasses import asdict

    config_dict = asdict(config)

    # Generate a stable job ID before kicking so the task can use it immediately
    job_id = str(uuid.uuid4())

    # Kick the TaskIQ task exactly once
    await extraction_task.kiq(
        file_path_str=str(tmp.name),
        config_dict=config_dict,
        language=language,
        no_discovery=no_discovery,
        url=url,
        task_id=job_id,
    )
    print(f"Started job {job_id}", file=sys.stderr, flush=True)
    return {"job_id": job_id}


@app.get("/jobs/{task_id}/status")
async def job_status(task_id: str):
    progress = job_progress.get(task_id, {"stage": "", "current": 0, "total": 0})

    if task_id in job_errors:
        return {"status": "failed", "progress": progress, "error": job_errors[task_id]}
    if task_id in job_results:
        return {"status": "completed", "progress": progress, "error": None}
    if task_id in job_progress:
        return {"status": "running", "progress": progress, "error": None}
    return {"status": "pending", "progress": progress, "error": None}


@app.get("/jobs/{task_id}/result")
async def job_result(task_id: str):
    if task_id in job_errors:
        return JSONResponse(status_code=500, content={"error": job_errors[task_id]})
    if task_id not in job_results:
        return JSONResponse(status_code=400, content={"error": "Job not complete"})
    return job_results[task_id]


@app.get("/jobs/{task_id}/report")
async def job_report(task_id: str):
    """Return a fully rendered, self-contained HTML report page."""
    if task_id in job_errors:
        return JSONResponse(status_code=500, content={"error": job_errors[task_id]})
    if task_id not in job_results:
        return JSONResponse(status_code=400, content={"error": "Job not complete"})

    data = job_results[task_id]
    html_content = render_report_html(data, title=data.get("title", "Knowledge Graph"))
    return HTMLResponse(content=html_content)


@app.websocket("/ws/jobs/{task_id}")
async def job_ws(websocket: WebSocket, task_id: str):
    """Real-time progress stream for a job."""
    await websocket.accept()
    job_websockets.setdefault(task_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        sockets = job_websockets.get(task_id, [])
        if websocket in sockets:
            sockets.remove(websocket)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=0, help="Port to listen on (0 = auto)"
    )
    args = parser.parse_args()

    port = args.port or _find_free_port()
    print(f"PORT:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
