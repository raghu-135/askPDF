"""Neutral runtime-protocol gateway for the Hermes HTTP/SSE API.

This service deliberately does not import Hermes, LangGraph, or rag-service
application modules. Hermes owns its session/run state behind HERMES_API_URL;
this gateway only translates between the neutral wire contract and Hermes' API.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


WIRE_VERSION = 1


def _envelope(*, status: str, result: Mapping[str, Any] | None = None, error: Mapping[str, Any] | None = None, request_id: str | None = None) -> dict[str, Any]:
    return {
        "contract_version": WIRE_VERSION,
        "request_id": request_id,
        "status": status,
        "result": dict(result or {}),
        "error": dict(error or {}),
        "runtime_metadata": {"framework": "hermes", "builder_id": "hermes_agent"},
    }


def _error(code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
    return {"code": code, "safe_message": message, "retryable": retryable, "details": {}}


def _neutral_event(run_id: str, sequence: int, kind: str, payload: Mapping[str, Any] | None = None, *, terminal: bool = False) -> dict[str, Any]:
    return {
        "event_id": f"{run_id}:{sequence}",
        "run_id": run_id,
        "sequence": sequence,
        "kind": kind,
        "payload": dict(payload or {}),
        "terminal": terminal,
        "contract_version": WIRE_VERSION,
    }


def _sse(event: Mapping[str, Any], result: Mapping[str, Any] | None = None) -> str:
    import json

    payload = {"event": dict(event)}
    if result is not None:
        payload["result"] = dict(result)
    return f"id: {event['event_id']}\nevent: {event['kind']}\ndata: {json.dumps(payload, separators=(',', ':'), default=str)}\n\n"


def _hermes_event_kind(event_name: str, payload: Mapping[str, Any]) -> str:
    name = event_name.lower()
    if name in {"message.delta", "message_delta", "content.delta"}:
        return "output.delta"
    if name in {"message.complete", "message_complete", "response.completed"}:
        return "output.completed"
    if name in {"tool.start", "tool.started", "tool_call.started"}:
        return "tool.started"
    if name in {"tool.complete", "tool.completed", "tool_call.completed"}:
        return "tool.completed"
    if name in {"run.completed", "completed", "done"}:
        return "run.completed"
    if name in {"run.failed", "failed", "error"}:
        return "run.failed"
    if name in {"run.cancelled", "run.canceled", "cancelled", "canceled"}:
        return "run.cancelled"
    if name.startswith("session"):
        return "runtime.session_started"
    return "runtime.event"


def create_app() -> FastAPI:
    state: dict[str, Any] = {"active": set(), "draining": False}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["draining"] = False
        yield
        state["draining"] = True

    app = FastAPI(title="AskPDF Hermes Runtime", version=os.getenv("HERMES_RUNTIME_VERSION", "hermes-gateway-1"), lifespan=lifespan)

    def upstream_url() -> str:
        return os.getenv("HERMES_API_URL", "http://hermes-agent:8000").rstrip("/")

    def timeout() -> httpx.Timeout:
        return httpx.Timeout(float(os.getenv("HERMES_RUNTIME_READ_TIMEOUT_SECONDS", "30")), connect=5, write=10)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "hermes-runtime"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(upstream_url() + "/health")
            ready = response.status_code < 500
        except Exception:
            ready = False
        return JSONResponse({"status": "ok" if ready else "not_ready", "checks": {"hermes": {"status": "ok" if ready else "failed"}}}, status_code=200 if ready else 503)

    @app.get("/v1/capabilities")
    async def capabilities(request: Request) -> dict[str, Any]:
        return _envelope(
            status="ok",
            request_id=request.headers.get("x-request-id"),
            result={"capabilities": {
                "streaming": True,
                "resume": False,
                "cancellation": True,
                "inspection": True,
                "continuation_cleanup": True,
                "task_execution": False,
                "native_checkpoints": False,
                "runtime_version": os.getenv("HERMES_RUNTIME_VERSION", "hermes-gateway-1"),
                "contract_version": WIRE_VERSION,
            }},
        )

    @app.post("/v1/validate")
    async def validate(payload: Mapping[str, Any], request: Request) -> dict[str, Any]:
        definition = payload.get("definition") or {}
        spec = payload.get("spec") or {}
        issues = []
        if definition.get("framework") != "hermes" or definition.get("builder_id") != "hermes_agent":
            issues.append({"code": "invalid_runtime_identity", "message": "Hermes runtime requires framework=hermes and builder_id=hermes_agent"})
        if not isinstance(spec.get("config"), Mapping):
            issues.append({"code": "missing_config", "message": "Hermes spec requires config"})
        return _envelope(status="ok", request_id=request.headers.get("x-request-id"), result={"validation": {"valid": not issues, "issues": issues, "normalized_spec": spec if not issues else None, "runtime_metadata": {"framework": "hermes", "builder_id": "hermes_agent"}}})

    async def stream_run(payload: Mapping[str, Any], request: Request, *, expected_run_id: str | None = None) -> AsyncIterator[str]:
        neutral_request = payload.get("request") or {}
        run_id = str(neutral_request.get("run_id") or "")
        if expected_run_id and run_id != expected_run_id:
            raise HTTPException(status_code=400, detail="run_id does not match request path")
        if state["draining"]:
            raise HTTPException(status_code=503, detail="runtime is draining")
        options = dict(neutral_request.get("options") or {})
        input_data = dict(neutral_request.get("input") or {})
        context = dict(payload.get("context") or {})
        question = str(input_data.get("question") or context.get("request_payload", {}).get("question") or "")
        config = dict(context.get("resolved_spec") or {}).get("config") or {}
        upstream_payload = {
            "input": {"messages": [{"role": "user", "content": question}]},
            "model": options.get("llm_model") or config.get("model") or os.getenv("HERMES_MODEL", ""),
            "stream": True,
            "metadata": {"askpdf_run_id": run_id, "askpdf_definition_id": neutral_request.get("definition_id")},
        }
        headers = {}
        token = os.getenv("HERMES_API_TOKEN")
        if token:
            headers["authorization"] = f"Bearer {token}"
        session_id = (neutral_request.get("continuation") or {}).get("payload", {}).get("session_id")
        if session_id:
            headers["X-Hermes-Session-Id"] = str(session_id)
        sequence = 1
        yield _sse(_neutral_event(run_id, sequence, "run.started", {"framework": "hermes"}))
        sequence += 1
        try:
            async with httpx.AsyncClient(timeout=timeout()) as client:
                response = await client.post(upstream_url() + "/v1/runs", headers=headers, json=upstream_payload)
                response.raise_for_status()
                start = response.json()
                upstream_run_id = str(start.get("run_id") or start.get("id") or run_id)
                headers["X-Hermes-Run-Id"] = upstream_run_id
                yield _sse(_neutral_event(run_id, sequence, "runtime.session_started", {"session_id": session_id, "upstream_run_id": upstream_run_id}))
                sequence += 1
                async with client.stream("GET", upstream_url() + f"/v1/runs/{upstream_run_id}/events", headers=headers) as events_response:
                    events_response.raise_for_status()
                    event_name = "message"
                    data: list[str] = []
                    async for line in events_response.aiter_lines():
                        if line == "":
                            if data:
                                import json
                                raw = json.loads("\n".join(data))
                                event_payload = raw if isinstance(raw, Mapping) else {"value": raw}
                                kind = _hermes_event_kind(event_name, event_payload)
                                terminal = kind in {"run.completed", "run.failed", "run.cancelled"}
                                event = _neutral_event(run_id, sequence, kind, event_payload, terminal=terminal)
                                result = None
                                if terminal:
                                    status = "completed" if kind == "run.completed" else "cancelled" if kind == "run.cancelled" else "failed"
                                    result = {"status": status, "output": event_payload.get("output") or event_payload.get("content"), "runtime_metadata": {"session_id": session_id, "upstream_run_id": upstream_run_id}, "error": event_payload.get("error")}
                                yield _sse(event, result)
                                sequence += 1
                            event_name, data = "message", []
                            continue
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                        elif line.startswith("data:"):
                            data.append(line[5:].lstrip())
                    if not data:
                        event = _neutral_event(run_id, sequence, "run.completed", {"output": ""}, terminal=True)
                        yield _sse(event, {"status": "completed", "output": "", "runtime_metadata": {"session_id": session_id}})
        except httpx.HTTPError as exc:
            event = _neutral_event(run_id, sequence, "run.failed", {"error": _error("hermes_upstream_error", "Hermes runtime is unavailable", retryable=True)}, terminal=True)
            yield _sse(event, {"status": "failed", "error": _error("hermes_upstream_error", str(exc), retryable=True)})
        finally:
            state["active"].discard(run_id)

    @app.post("/v1/runs/start")
    async def start(payload: Mapping[str, Any], request: Request) -> StreamingResponse:
        return StreamingResponse(stream_run(payload, request), media_type="text/event-stream")

    @app.post("/v1/runs/{run_id}/resume")
    async def resume(run_id: str) -> JSONResponse:
        return JSONResponse(_envelope(status="failed", error=_error("runtime_capability_unsupported", "Hermes resume is not enabled")), status_code=409)

    @app.post("/v1/runs/{run_id}/continue")
    async def continue_run(run_id: str) -> JSONResponse:
        return JSONResponse(_envelope(status="failed", error=_error("runtime_capability_unsupported", "Hermes continuation is not enabled")), status_code=409)

    @app.post("/v1/runs/{run_id}/cancel")
    async def cancel(run_id: str, request: Request) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(upstream_url() + f"/v1/runs/{run_id}/stop")
                response.raise_for_status()
            return _envelope(status="ok", request_id=request.headers.get("x-request-id"), result={"run_id": run_id, "status": "cancellation_requested"})
        except httpx.HTTPError as exc:
            return _envelope(status="failed", error=_error("hermes_cancel_failed", str(exc), retryable=True), request_id=request.headers.get("x-request-id"))

    @app.post("/v1/runs/{run_id}/inspect")
    async def inspect(run_id: str, request: Request) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(upstream_url() + f"/v1/runs/{run_id}")
                response.raise_for_status()
            return _envelope(status="ok", request_id=request.headers.get("x-request-id"), result=dict(response.json()))
        except httpx.HTTPError as exc:
            return _envelope(status="failed", error=_error("hermes_inspect_failed", str(exc), retryable=True), request_id=request.headers.get("x-request-id"))

    @app.delete("/v1/continuations/{binding_id}")
    async def delete_continuation(binding_id: str, request: Request) -> dict[str, Any]:
        return _envelope(status="ok", request_id=request.headers.get("x-request-id"), result={"binding_id": binding_id, "status": "released"})

    return app
