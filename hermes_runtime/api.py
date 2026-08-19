"""Neutral runtime-protocol gateway for the Hermes HTTP/SSE API.

This service deliberately does not import Hermes, LangGraph, or rag-service
application modules. Hermes owns its session/run state behind HERMES_API_URL;
this gateway only translates between the neutral wire contract and Hermes' API.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from hermes_runtime.execution_store import (
    HermesExecutionConflictError,
    HermesExecutionStore,
)
from hermes_runtime.compatibility import HERMES_REVISION, HERMES_TERMINAL_EVENTS


WIRE_VERSION = 1
logger = logging.getLogger(__name__)


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


def _neutral_event(run_id: str, sequence: int, kind: str, payload: Mapping[str, Any] | None = None, *, event_id: str | None = None, source_event_id: str | None = None, terminal: bool = False, continuation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    event = {
        "event_id": event_id or f"{run_id}:{sequence}",
        "run_id": run_id,
        "sequence": sequence,
        "kind": kind,
        "payload": dict(payload or {}),
        "terminal": terminal,
        "contract_version": WIRE_VERSION,
    }
    if continuation is not None:
        event["continuation"] = dict(continuation)
    if source_event_id is not None:
        event["source_event_id"] = str(source_event_id)
    return event


def _sse(event: Mapping[str, Any], result: Mapping[str, Any] | None = None) -> str:
    payload = {"event": dict(event)}
    if result is not None:
        payload["result"] = dict(result)
    return f"id: {event['event_id']}\nevent: {event['kind']}\ndata: {json.dumps(payload, separators=(',', ':'), default=str)}\n\n"


def _recovery_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a durable record and restore its upstream binding into the request."""
    payload = copy.deepcopy(dict(record.get("payload") or {}))
    request = dict(payload.get("request") or {})
    continuation = record.get("continuation")
    if isinstance(continuation, Mapping):
        request["continuation"] = copy.deepcopy(dict(continuation))
    payload["request"] = request
    return payload


def _response_session_id(value: Mapping[str, Any]) -> str | None:
    """Extract a Hermes session ID from supported start-response shapes."""
    direct = value.get("session_id")
    if direct:
        return str(direct)
    session = value.get("session")
    if isinstance(session, Mapping) and session.get("id"):
        return str(session["id"])
    return None


def _hermes_event_kind(event_name: str, payload: Mapping[str, Any]) -> str:
    name = str(payload.get("event") or event_name).lower()
    if name == "message.delta":
        return "output.delta"
    if name == "tool.started":
        return "tool.started"
    if name == "tool.completed":
        return "tool.completed"
    if name == "reasoning.available":
        return "reasoning.available"
    if name == "approval.request":
        return "approval.request"
    if name == "approval.responded":
        return "approval.responded"
    if name == "run.steered":
        return "run.steered"
    if name == "run.completed":
        return "run.completed"
    if name == "run.failed":
        return "run.failed"
    if name == "run.cancelled":
        return "run.cancelled"
    if name in {"subagent.start", "subagent.complete"}:
        return name
    return "runtime.event"


def create_app() -> FastAPI:
    hermes_api_url = os.getenv("HERMES_API_URL", "").strip()
    if not hermes_api_url:
        raise RuntimeError(
            "HERMES_API_URL is required for the Hermes runtime"
        )
    storage_backend = os.getenv("HERMES_RUNTIME_STORAGE_BACKEND", "file").strip().lower()
    worker_count = int(os.getenv("HERMES_RUNTIME_WORKERS", os.getenv("WEB_CONCURRENCY", "1")))
    if storage_backend != "file":
        raise RuntimeError("Hermes PostgreSQL execution storage is not enabled")
    if worker_count > 1:
        raise RuntimeError("Hermes file execution storage supports one worker only")
    store = HermesExecutionStore()
    state: dict[str, Any] = {
        "active": {},
        "draining": False,
        "store": store,
        "storage_backend": storage_backend,
        "worker_count": worker_count,
        "storage_healthy": True,
    }
    start_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["draining"] = False
        # Recovered records remain inspectable. Their upstream bindings are
        # intentionally not discarded when the gateway process restarts.
        for run_id, record in list(store.records.items()):
            if record.get("status") in {"queued", "running"} and record.get("payload"):
                state["active"][run_id] = asyncio.create_task(
                    _background_run(_recovery_payload(record), None),
                    name=f"hermes-runtime-recovery-{run_id}",
                )
        yield
        state["draining"] = True

    app = FastAPI(title="AskPDF Hermes Runtime", version=os.getenv("HERMES_RUNTIME_VERSION", "hermes-gateway-1"), lifespan=lifespan)

    def upstream_url() -> str:
        return hermes_api_url.rstrip("/")

    def upstream_headers(session_id: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        token = os.getenv("HERMES_API_TOKEN")
        if token:
            headers["authorization"] = f"Bearer {token}"
        if session_id:
            headers["X-Hermes-Session-Id"] = str(session_id)
        return headers

    def timeout(max_seconds: float | None = None) -> httpx.Timeout:
        read_timeout = float(os.getenv("HERMES_RUNTIME_READ_TIMEOUT_SECONDS", "30"))
        if max_seconds is not None:
            read_timeout = min(read_timeout, max_seconds)
        return httpx.Timeout(read_timeout, connect=5, write=10)

    def configured_mcp_tools() -> set[str]:
        return {item.strip() for item in os.getenv("HERMES_MCP_ALLOWED_TOOLS", "document_evidence,thread_conversation_history,clarify_intent").split(",") if item.strip()}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "hermes-runtime"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        hermes_ready = False
        mcp_required = os.getenv("ASKPDF_MCP_REQUIRED", "true").lower() in {"1", "true", "yes", "on"}
        mcp_ready = not mcp_required
        mcp_checked = False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(upstream_url() + "/health")
                hermes_ready = 200 <= response.status_code < 400
                if hermes_ready and mcp_required:
                    mcp_url = os.getenv("ASKPDF_MCP_HEALTH_URL", "").strip()
                    if mcp_url:
                        mcp_checked = True
                        mcp_response = await client.get(mcp_url)
                        mcp_ready = 200 <= mcp_response.status_code < 400
        except Exception:
            pass
        if not state["storage_healthy"]:
            state["storage_healthy"] = store.probe()
        storage_ready = bool(state["storage_healthy"])
        ready = hermes_ready and mcp_ready and storage_ready
        return JSONResponse(
            {
                "status": "ok" if ready else "not_ready",
                "checks": {
                    "hermes": {
                        "status": "ok" if hermes_ready else "failed",
                        "storage_backend": state["storage_backend"],
                        "worker_count": state["worker_count"],
                    },
                    "mcp": {
                        "status": "ok" if mcp_ready else ("failed" if mcp_checked else "not_checked"),
                        "required": mcp_required,
                    },
                    "storage": {
                        "status": "ok" if storage_ready else "failed",
                        "backend": state["storage_backend"],
                    },
                },
            },
            status_code=200 if ready else 503,
        )

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
                "continuation_cleanup": False,
                "task_execution": False,
                "native_checkpoints": False,
                "approval_response": True,
                "steering": True,
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
        if spec.get("schema_version") != 2:
            issues.append({"code": "unsupported_schema_version", "message": "Hermes definitions must use schema_version 2"})
        if spec.get("definition_version") != 1:
            issues.append({"code": "unsupported_definition_version", "message": "Hermes definitions must use definition_version 1"})
        config = spec.get("config")
        if not isinstance(config, Mapping):
            issues.append({"code": "missing_config", "message": "Hermes spec requires config"})
        else:
            if config.get("mcp_server") != "askpdf":
                issues.append({"code": "unsupported_mcp_server", "message": "Hermes runtime requires mcp_server=askpdf"})
            if not set(config.get("allowed_tool_ids") or []).issubset(configured_mcp_tools()):
                issues.append({"code": "unsupported_tool_allowlist", "message": "Hermes tool allowlist exceeds the configured MCP catalog"})
        return _envelope(status="ok", request_id=request.headers.get("x-request-id"), result={"validation": {"valid": not issues, "issues": issues, "normalized_spec": spec if not issues else None, "runtime_metadata": {"framework": "hermes", "builder_id": "hermes_agent", "hermes_revision": HERMES_REVISION, "mcp_server": "askpdf", "allowed_tool_ids": sorted(configured_mcp_tools())}}})

    def _binding(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        continuation = payload.get("continuation") or (payload.get("request") or {}).get("continuation")
        if not isinstance(continuation, Mapping):
            return None
        if continuation.get("binding_type") != "hermes_session":
            return None
        return continuation

    def _upstream_run_id(run_id: str, payload: Mapping[str, Any]) -> str:
        binding = _binding(payload)
        value = (binding or {}).get("payload") if binding else None
        upstream = value.get("upstream_run_id") if isinstance(value, Mapping) else None
        if not upstream:
            raise HTTPException(status_code=409, detail=_error("runtime_binding_missing", "Hermes upstream run binding is not available"))
        return str(upstream)

    async def stream_run(payload: Mapping[str, Any], request: Request, *, expected_run_id: str | None = None) -> AsyncIterator[str]:
        neutral_request = payload.get("request") or {}
        run_id = str(neutral_request.get("run_id") or "")
        if expected_run_id and run_id != expected_run_id:
            raise HTTPException(status_code=400, detail="run_id does not match request path")
        if state["draining"]:
            raise HTTPException(status_code=503, detail="runtime is draining")
        options = dict(neutral_request.get("options") or {})
        input_data = dict(neutral_request.get("input") or {})
        continuation = neutral_request.get("continuation") if isinstance(neutral_request.get("continuation"), Mapping) else None
        context = dict(payload.get("context") or {})
        question = str(input_data.get("question") or context.get("request_payload", {}).get("question") or "")
        config = dict(context.get("resolved_spec") or {}).get("config") or {}
        max_events = max(1, int(config.get("max_event_count") or 200))
        max_output_chars = max(1, int(config.get("max_output_chars") or 12000))
        max_duration_seconds = max(1, int(config.get("max_duration_seconds") or 300))
        deadline = time.monotonic() + max_duration_seconds
        system_prompt = str(config.get("system_prompt") or "").strip()
        task_context = input_data.get("task_context")
        context_token = str(input_data.get("mcp_execution_context_token") or "").strip()
        if isinstance(task_context, Mapping):
            question = question + "\n\naskPDF task context:\n" + json.dumps(task_context, sort_keys=True, ensure_ascii=False)
        if context_token:
            system_prompt += (
                "\n\nFor every askPDF MCP tool call, pass this exact value in the "
                "_askpdf_context_token argument: " + context_token
            )
        upstream_payload = {
            "input": question,
            "instructions": system_prompt or None,
            "model": options.get("llm_model") or config.get("model") or "",
            "provider": options.get("llm_provider") or config.get("provider") or "custom",
            "metadata": {
                "askpdf_run_id": run_id,
                "askpdf_thread_id": neutral_request.get("thread_id"),
                "askpdf_definition_id": neutral_request.get("definition_id"),
                "askpdf_profile_id": (dict(context.get("resolved_spec") or {}).get("managed_profile") or {}).get("profile_id"),
            },
        }
        upstream_payload = {key: value for key, value in upstream_payload.items() if value not in (None, "")}
        headers = upstream_headers()
        session_id = (neutral_request.get("continuation") or {}).get("payload", {}).get("session_id")
        if session_id:
            headers["X-Hermes-Session-Id"] = str(session_id)
        sequence = store.next_sequence(run_id) if os.getenv("HERMES_RUNTIME_EVENT_ID_MODE", "durable").strip().lower() == "durable" else 1
        output_chars = 0
        terminal_seen = False

        def process_frame(frame_event_name: str, frame_data: list[str]) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
            nonlocal output_chars, sequence, terminal_seen, session_id
            try:
                raw = json.loads("\n".join(frame_data))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=502,
                    detail=_error("hermes_upstream_protocol_error", "Hermes emitted malformed event JSON"),
                ) from exc
            if not isinstance(raw, Mapping):
                raise HTTPException(
                    status_code=502,
                    detail=_error("hermes_upstream_protocol_error", "Hermes emitted an invalid event envelope"),
                )
            event_payload = raw
            upstream_event_name = str(event_payload.get("event") or frame_event_name)
            source_event_id = event_payload.get("event_id") or event_payload.get("id") or event_payload.get("sequence")
            if source_event_id is not None:
                source_event_id = f"{upstream_event_name}:{source_event_id}"
            kind = _hermes_event_kind(upstream_event_name, event_payload)
            if kind == "runtime.event":
                event_payload = {"upstream_event": upstream_event_name, "data": dict(event_payload)}
            terminal = kind in HERMES_TERMINAL_EVENTS
            if kind == "output.delta":
                output_chars += len(str(event_payload.get("content") or event_payload.get("delta") or event_payload.get("text") or ""))
                if output_chars > max_output_chars:
                    raise HTTPException(status_code=409, detail=_error("runtime_limit_exceeded", "Hermes output exceeded the configured limit"))
            if sequence > max_events:
                raise HTTPException(status_code=409, detail=_error("runtime_limit_exceeded", "Hermes emitted too many events"))
            event = _neutral_event(run_id, sequence, kind, event_payload, source_event_id=source_event_id, terminal=terminal, continuation=continuation)
            result = None
            if terminal:
                terminal_seen = True
                status = "completed" if kind == "run.completed" else "cancelled" if kind == "run.cancelled" else "failed"
                output = event_payload.get("output") or event_payload.get("content")
                if output is not None:
                    output = str(output)[:max_output_chars]
                result = {
                    "status": status,
                    "output": output,
                    "runtime_metadata": {
                        "session_id": session_id,
                        "upstream_run_id": upstream_run_id,
                        "mcp_server": config.get("mcp_server"),
                        "allowed_tool_ids": list(config.get("allowed_tool_ids") or []),
                    },
                    "continuation": continuation,
                    "error": event_payload.get("error"),
                }
            sequence += 1
            operation_event = None
            if terminal:
                operation_event = _neutral_event(
                    run_id,
                    sequence,
                    "operation.failed" if kind == "run.failed" else "operation.completed",
                    {
                        "operation_id": "hermes_session",
                        "operation_type": "agent_session",
                        "operation_label": "Hermes Agent",
                        "visit_index": 1,
                        "status": status,
                        "error": event_payload.get("error"),
                    },
                    continuation=continuation,
                )
                sequence += 1
                event["sequence"] = sequence
                event["event_id"] = f"{run_id}:{sequence}"
                sequence += 1
            return event, result, operation_event

        try:
            async with httpx.AsyncClient(timeout=timeout(max_duration_seconds)) as client:
                existing_binding = (neutral_request.get("continuation") or {}).get("payload") or {}
                upstream_run_id = str(existing_binding.get("upstream_run_id") or "")
                if not upstream_run_id:
                    response = await client.post(upstream_url() + "/v1/runs", headers=headers, json=upstream_payload)
                    response.raise_for_status()
                    start = response.json()
                    response_session_id = _response_session_id(start)
                    if not session_id and response_session_id:
                        session_id = response_session_id
                    upstream_run_id = str(start.get("run_id") or start.get("id") or "")
                if not upstream_run_id:
                    raise HTTPException(status_code=502, detail=_error("runtime_protocol_error", "Hermes did not return an upstream run ID"))
                if not session_id:
                    status_response = await client.get(upstream_url() + f"/v1/runs/{upstream_run_id}", headers=headers)
                    status_response.raise_for_status()
                    status_payload = status_response.json()
                    if isinstance(status_payload, Mapping):
                        session_id = _response_session_id(status_payload)
                if session_id:
                    headers["X-Hermes-Session-Id"] = str(session_id)
                headers["X-Hermes-Run-Id"] = upstream_run_id
                continuation = {
                    "binding_type": "hermes_session",
                    "binding_version": 1,
                    "runtime_version": HERMES_REVISION,
                    "payload": {"session_id": session_id, "upstream_run_id": upstream_run_id},
                }
                yield _sse(_neutral_event(run_id, sequence, "run.started", {"framework": "hermes"}, continuation=continuation))
                sequence += 1
                yield _sse(_neutral_event(run_id, sequence, "operation.started", {
                    "operation_id": "hermes_session",
                    "operation_type": "agent_session",
                    "operation_label": "Hermes Agent",
                    "visit_index": 1,
                    "session_id": session_id,
                }, continuation=continuation))
                sequence += 1
                async with client.stream("GET", upstream_url() + f"/v1/runs/{upstream_run_id}/events", headers=headers) as events_response:
                    events_response.raise_for_status()
                    event_name = "message"
                    data: list[str] = []
                    async for line in events_response.aiter_lines():
                        if time.monotonic() >= deadline:
                            raise HTTPException(status_code=409, detail=_error("runtime_limit_exceeded", "Hermes execution exceeded the configured duration"))
                        if line == "":
                            if data:
                                event, result, operation_event = process_frame(event_name, data)
                                if operation_event is not None:
                                    yield _sse(operation_event)
                                yield _sse(event, result)
                                if terminal_seen:
                                    return
                            event_name, data = "message", []
                            continue
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                        elif line.startswith("data:"):
                            data.append(line[5:].lstrip())
                    if data and not terminal_seen:
                        event, result, operation_event = process_frame(event_name, data)
                        if operation_event is not None:
                            yield _sse(operation_event)
                        yield _sse(event, result)
                        if terminal_seen:
                            return
                    if not terminal_seen:
                        error = _error("hermes_upstream_protocol_error", "Hermes closed the event stream without a terminal event", retryable=True)
                        event = _neutral_event(run_id, sequence, "run.failed", {"error": error}, terminal=True, continuation=continuation)
                        terminal_seen = True
                        yield _sse(event, {"status": "failed", "error": error, "continuation": continuation})
        except HTTPException as exc:
            if terminal_seen:
                return
            detail = exc.detail if isinstance(exc.detail, Mapping) else _error("hermes_runtime_error", str(exc.detail))
            error = detail if isinstance(detail, Mapping) and detail.get("code") else _error("hermes_runtime_error", str(exc.detail))
            event = _neutral_event(run_id, sequence, "run.failed", {"error": error}, terminal=True, continuation=continuation)
            terminal_seen = True
            yield _sse(event, {"status": "failed", "error": error, "continuation": continuation})
        except httpx.HTTPError as exc:
            if terminal_seen:
                return
            event = _neutral_event(run_id, sequence, "run.failed", {"error": _error("hermes_upstream_error", "Hermes runtime is unavailable", retryable=True)}, terminal=True, continuation=continuation)
            terminal_seen = True
            yield _sse(event, {"status": "failed", "error": _error("hermes_upstream_error", str(exc), retryable=True), "continuation": continuation})
    async def _background_run(payload: Mapping[str, Any], request: Request | None) -> None:
        run_id = str((payload.get("request") or {}).get("run_id") or "")
        try:
            async for frame in stream_run(payload, request):
                terminal_status = None
                if "event: run.completed" in frame:
                    terminal_status = "completed"
                elif "event: run.failed" in frame:
                    terminal_status = "failed"
                elif "event: run.cancelled" in frame:
                    terminal_status = "cancelled"
                if terminal_status:
                    state["store"].finalize(run_id, frame, status=terminal_status)
                else:
                    state["store"].append(run_id, frame)
        except Exception:
            logger.exception("Hermes background execution failed | run_id=%s", run_id)
            record = state["store"].records.setdefault(run_id, {"run_id": run_id, "events": []})
            if record.get("status") not in {"completed", "failed", "cancelled"} and not record.get("terminal_event_id"):
                sequence = state["store"].next_sequence(run_id)
                continuation = record.get("continuation")
                error = _error(
                    "hermes_gateway_internal_error",
                    "Hermes gateway failed while processing the run",
                    retryable=True,
                )
                event = _neutral_event(
                    run_id,
                    sequence,
                    "run.failed",
                    {"error": error},
                    terminal=True,
                    continuation=continuation if isinstance(continuation, Mapping) else None,
                )
                frame = _sse(event, {"status": "failed", "error": error, "continuation": continuation})
                try:
                    state["store"].finalize(run_id, frame, status="failed")
                except Exception:
                    state["storage_healthy"] = False
                    logger.critical("Hermes terminal failure could not be persisted | run_id=%s", run_id, exc_info=True)
                    # Keep current subscribers finite even though durability is unavailable.
                    try:
                        state["store"].fail_in_memory(run_id, frame)
                    except Exception:
                        record["status"] = "failed"
                        logger.critical("Hermes in-memory terminal frame could not be recorded | run_id=%s", run_id, exc_info=True)
        finally:
            state["active"].pop(run_id, None)

    async def _subscribe(run_id: str, after_event_id: str | None = None) -> AsyncIterator[str]:
        sent = after_event_id
        while True:
            frames = state["store"].frames_after(run_id, sent)
            if frames:
                for frame in frames:
                    yield frame
                    sent = next((line[3:].strip() for line in frame.splitlines() if line.startswith("id:")), sent)
                    if any(line in frame for line in ("event: run.completed", "event: run.failed", "event: run.cancelled")):
                        return
            record = state["store"].records.get(run_id)
            if record is None:
                return
            if record.get("status") in {"completed", "failed", "cancelled"}:
                return
            task = state["active"].get(run_id)
            if task is None and record.get("status") in {"queued", "running"}:
                error = _error(
                    "hermes_gateway_internal_error",
                    "Hermes gateway execution stopped unexpectedly",
                    retryable=True,
                )
                sequence = state["store"].next_sequence(run_id)
                frame = _sse(
                    _neutral_event(run_id, sequence, "run.failed", {"error": error}, terminal=True),
                    {"status": "failed", "error": error},
                )
                try:
                    state["store"].finalize(run_id, frame, status="failed")
                except Exception:
                    state["storage_healthy"] = False
                    state["store"].fail_in_memory(run_id, frame)
                yield frame
                return
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.1)

    @app.post("/v1/runs/start")
    async def start(payload: Mapping[str, Any], request: Request) -> StreamingResponse:
        run_id = str((payload.get("request") or {}).get("run_id") or "")
        if not run_id:
            raise HTTPException(status_code=400, detail=_error("runtime_protocol_error", "run_id is required"))
        async with start_lock:
            try:
                record = state["store"].create(run_id, payload)
            except HermesExecutionConflictError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=_error(
                        "runtime_operation_conflict",
                        str(exc),
                        retryable=False,
                    ),
                ) from exc
            task = state["active"].get(run_id)
            if record.get("status") == "queued" and (task is None or task.done()):
                state["store"].update(run_id, status="running")
                task = asyncio.create_task(_background_run(payload, request), name=f"hermes-runtime-{run_id}")
                state["active"][run_id] = task
        return StreamingResponse(_subscribe(run_id), media_type="text/event-stream")

    @app.get("/v1/runs/{run_id}/events")
    async def events(run_id: str, after_event_id: str | None = None) -> StreamingResponse:
        if run_id not in state["store"].records:
            raise HTTPException(status_code=404, detail="runtime run not found")
        return StreamingResponse(_subscribe(run_id, after_event_id), media_type="text/event-stream")

    @app.post("/v1/runs/{run_id}/resume")
    async def resume(run_id: str) -> JSONResponse:
        return JSONResponse(_envelope(status="failed", error=_error("runtime_capability_unsupported", "Hermes resume is not enabled")), status_code=409)

    @app.post("/v1/runs/{run_id}/continue")
    async def continue_run(run_id: str) -> JSONResponse:
        return JSONResponse(_envelope(status="failed", error=_error("runtime_capability_unsupported", "Hermes continuation is not enabled")), status_code=409)

    @app.post("/v1/runs/{run_id}/cancel")
    async def cancel(run_id: str, request: Request, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            payload = payload or {}
            upstream_run_id = _upstream_run_id(run_id, payload)
            binding = _binding(payload)
            session_id = ((binding or {}).get("payload") or {}).get("session_id")
            headers = upstream_headers(session_id)
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(upstream_url() + f"/v1/runs/{upstream_run_id}/stop", headers=headers)
                response.raise_for_status()
            return _envelope(status="ok", request_id=request.headers.get("x-request-id"), result={"run_id": run_id, "upstream_run_id": upstream_run_id, "status": "cancellation_requested"})
        except httpx.HTTPError as exc:
            return _envelope(status="failed", error=_error("hermes_cancel_failed", str(exc), retryable=True), request_id=request.headers.get("x-request-id"))

    @app.post("/v1/runs/{run_id}/inspect")
    async def inspect(run_id: str, request: Request, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            payload = payload or {}
            upstream_run_id = _upstream_run_id(run_id, payload)
            binding = _binding(payload)
            session_id = ((binding or {}).get("payload") or {}).get("session_id")
            headers = upstream_headers(session_id)
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(upstream_url() + f"/v1/runs/{upstream_run_id}", headers=headers)
                response.raise_for_status()
            return _envelope(status="ok", request_id=request.headers.get("x-request-id"), result={**dict(response.json()), "run_id": run_id, "upstream_run_id": upstream_run_id})
        except httpx.HTTPError as exc:
            return _envelope(status="failed", error=_error("hermes_inspect_failed", str(exc), retryable=True), request_id=request.headers.get("x-request-id"))

    async def _forward_control(run_id: str, request: Request, payload: Mapping[str, Any], operation: str, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            upstream_run_id = _upstream_run_id(run_id, payload)
            binding = _binding(payload)
            session_id = ((binding or {}).get("payload") or {}).get("session_id")
            headers = upstream_headers(session_id)
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(upstream_url() + f"/v1/runs/{upstream_run_id}/{operation}", headers=headers, json=dict(body))
                response.raise_for_status()
            result = response.json()
            return _envelope(status="ok", request_id=request.headers.get("x-request-id"), result={**dict(result), "run_id": run_id, "upstream_run_id": upstream_run_id})
        except httpx.HTTPError as exc:
            return _envelope(status="failed", error=_error(f"hermes_{operation}_failed", str(exc), retryable=True), request_id=request.headers.get("x-request-id"))

    @app.post("/v1/runs/{run_id}/approval")
    async def approval(run_id: str, request: Request, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = payload.get("response") or {}
        choice = str(response.get("choice") or "").strip().lower()
        if choice not in {"once", "session", "always", "deny"}:
            raise HTTPException(status_code=400, detail=_error("invalid_approval_choice", "Approval choice must be once, session, always, or deny"))
        return await _forward_control(run_id, request, payload, "approval", {"choice": choice, "resolve_all": bool(response.get("resolve_all"))})

    @app.post("/v1/runs/{run_id}/steer")
    async def steer(run_id: str, request: Request, payload: Mapping[str, Any]) -> dict[str, Any]:
        steering = payload.get("steering") or {}
        text = str(steering.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail=_error("invalid_steer_input", "Steering text is required"))
        return await _forward_control(run_id, request, payload, "steer", {"input": text})

    @app.delete("/v1/continuations/{binding_id}")
    async def delete_continuation(binding_id: str, request: Request) -> dict[str, Any]:
        return JSONResponse(_envelope(status="failed", request_id=request.headers.get("x-request-id"), error=_error("runtime_capability_unsupported", "Hermes does not expose safe durable session deletion")), status_code=409)

    return app
