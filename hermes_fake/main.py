from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from typing import Any, AsyncIterator, Mapping

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


app = FastAPI(title="Deterministic Hermes Upstream")
mode = os.getenv("FAKE_HERMES_MODE", "normal")
runs: dict[str, dict[str, Any]] = {}
counters: dict[str, int] = defaultdict(int)


def _frame(event: str, payload: Mapping[str, Any], *, terminated: bool = True) -> str:
    suffix = "\n\n" if terminated else ""
    return f"event: {event}\ndata: {json.dumps(dict(payload), separators=(',', ':'))}\n{suffix}"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug/state")
async def debug_state() -> dict[str, Any]:
    return {"mode": mode, "counters": dict(counters), "runs": runs}


@app.post("/debug/mode")
async def set_mode(payload: Mapping[str, Any]) -> dict[str, str]:
    global mode
    mode = str(payload.get("mode") or "normal")
    return {"mode": mode}


@app.post("/v1/runs")
async def start(payload: Mapping[str, Any], request: Request) -> dict[str, Any]:
    counters["starts"] += 1
    number = counters["starts"]
    run_id = f"fake-hermes-run-{number}"
    session_id = f"fake-hermes-session-{number}"
    runs[run_id] = {
        "run_id": run_id,
        "session_id": session_id,
        "status": "running",
        "payload": dict(payload),
        "headers": {"x-hermes-session-id": request.headers.get("x-hermes-session-id"), "x-hermes-run-id": request.headers.get("x-hermes-run-id")},
    }
    return {"run_id": run_id, "session_id": session_id}


async def _events(run_id: str) -> AsyncIterator[str]:
    record = runs[run_id]
    counters["event_streams"] += 1
    if mode == "delayed":
        await asyncio.sleep(1)
    yield _frame("message.delta", {"event_id": f"{run_id}:progress-1", "delta": "deterministic "})
    if mode == "missing_terminal":
        return
    if mode == "unterminated":
        yield _frame("run.completed", {"event_id": f"{run_id}:terminal", "output": "deterministic result"}, terminated=False)
    else:
        yield _frame("run.completed", {"event_id": f"{run_id}:terminal", "output": "deterministic result"})
    record["status"] = "completed"


@app.get("/v1/runs/{run_id}/events")
async def events(run_id: str, request: Request) -> StreamingResponse:
    runs[run_id]["headers"] = {
        "x-hermes-session-id": request.headers.get("x-hermes-session-id"),
        "x-hermes-run-id": request.headers.get("x-hermes-run-id"),
    }
    return StreamingResponse(_events(run_id), media_type="text/event-stream")


@app.post("/v1/runs/{run_id}/stop")
async def stop(run_id: str) -> dict[str, Any]:
    counters["stops"] += 1
    record = runs[run_id]
    record["status"] = "cancelled"
    return {"run_id": run_id, "session_id": record["session_id"], "status": "cancelled"}


@app.get("/v1/runs/{run_id}")
async def inspect(run_id: str) -> dict[str, Any]:
    counters["inspects"] += 1
    return dict(runs[run_id])
