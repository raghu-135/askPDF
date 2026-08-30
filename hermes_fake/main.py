from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from typing import Any, AsyncIterator, Mapping

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


app = FastAPI(title="Deterministic Hermes Upstream")
mode = os.getenv("FAKE_HERMES_MODE", "normal")
runs: dict[str, dict[str, Any]] = {}
counters: dict[str, int] = defaultdict(int)
mcp_invocations: list[dict[str, Any]] = []


async def _mcp_request(url: str, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            url,
            headers={"accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": counters["mcp_requests"] + 1, "method": method, "params": dict(params)},
        )
        response.raise_for_status()
        counters["mcp_requests"] += 1
        return dict(response.json())


async def _invoke_document_evidence(record: dict[str, Any]) -> dict[str, Any]:
    payload = record["payload"]
    server = (payload.get("mcp_servers") or {}).get("askpdf") or {}
    url = str(server.get("url") or "")
    allowed_contracts = list((server.get("tools") or {}).get("include") or [])
    listed = await _mcp_request(url, "tools/list", {})
    tools = ((listed.get("result") or {}).get("tools") or [])
    selected = next(
        item
        for item in tools
        if ((item.get("_meta") or {}).get("com.askpdf/contract-id") == "document_evidence")
    )
    if "document_evidence" not in allowed_contracts:
        raise RuntimeError("document_evidence was not enabled for the deterministic Hermes run")
    metadata = payload.get("metadata") or {}
    runtime_context = {
        "thread_id": metadata.get("askpdf_thread_id"),
        "run_id": metadata.get("askpdf_run_id"),
        "tool_call_id": f"{record['run_id']}:document-evidence",
        "mcp_request_id": f"{record['run_id']}:mcp",
    }
    called = await _mcp_request(
        url,
        "tools/call",
        {
            "name": selected["name"],
            "arguments": {"query": "deterministic Hermes document evidence", "max_results": 3},
            "_meta": {"com.askpdf/runtime-context": runtime_context},
        },
    )
    result = called.get("result") or {}
    invocation = {
        "contract_id": "document_evidence",
        "tool_name": selected["name"],
        "run_id": runtime_context["run_id"],
        "thread_id": runtime_context["thread_id"],
        "request": {"arguments": {"query": "deterministic Hermes document evidence", "max_results": 3}, "runtime_context": runtime_context},
        "is_error": bool(result.get("isError")),
        "trace": (result.get("structuredContent") or {}).get("trace") or {},
    }
    mcp_invocations.append(invocation)
    counters["mcp_tool_calls"] += 1
    return invocation


def _frame(event: str, payload: Mapping[str, Any], *, terminated: bool = True) -> str:
    suffix = "\n\n" if terminated else ""
    return f"event: {event}\ndata: {json.dumps(dict(payload), separators=(',', ':'))}\n{suffix}"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug/state")
async def debug_state() -> dict[str, Any]:
    return {"mode": mode, "counters": dict(counters), "runs": runs, "mcp_invocations": mcp_invocations}


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
    if mode == "malformed_json":
        yield "event: message.delta\ndata: {not-json}\n\n"
        return
    if mode == "invalid_shape":
        yield "event: message.delta\ndata: [1,2,3]\n\n"
        return
    invocation = await _invoke_document_evidence(record)
    yield _frame("message.delta", {"event_id": f"{run_id}:progress-1", "delta": "deterministic "})
    if mode == "missing_terminal":
        return
    if mode == "unterminated":
        yield _frame("run.completed", {"event_id": f"{run_id}:terminal", "output": f"deterministic result from {invocation['contract_id']}"}, terminated=False)
    else:
        yield _frame("run.completed", {"event_id": f"{run_id}:terminal", "output": f"deterministic result from {invocation['contract_id']}"})
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
