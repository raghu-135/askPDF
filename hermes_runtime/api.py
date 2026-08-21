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
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from hermes_runtime.execution_store import (
    HermesExecutionConflictError,
    HermesExecutionStore,
)
from hermes_runtime.compatibility import HERMES_REVISION, HERMES_TERMINAL_EVENTS
from hermes_runtime.profile_manager import (
    RunProfile,
    RunProfileManager,
    configured_context_length,
    configured_provider,
    validate_provider_context,
)


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


def _error(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "safe_message": message,
        "retryable": retryable,
        "details": dict(details or {}),
    }


def _upstream_timeout(max_seconds: float | None = None) -> httpx.Timeout:
    """Use the task budget for streams and the configured limit for short calls."""
    read_timeout = (
        float(max_seconds)
        if max_seconds is not None
        else float(os.getenv("HERMES_RUNTIME_READ_TIMEOUT_SECONDS", "30"))
    )
    return httpx.Timeout(read_timeout, connect=5, write=10)


def _rendered_model_context_length() -> int:
    config_path = Path(os.getenv("HERMES_RENDERED_CONFIG_PATH", "/opt/data/config.yaml"))
    match = re.search(r"(?m)^\s{2}context_length:\s*([0-9]+)\s*$", config_path.read_text())
    if not match:
        raise RuntimeError("Hermes rendered config has no concrete model.context_length")
    return int(match.group(1))


class _HermesEventBudget:
    """Bound semantic events and streamed text independently.

    Hermes commonly emits one message.delta per token. Those chunks are
    bounded by output characters and must not consume the lifecycle-event
    budget used for tools, reasoning, approvals, and other state changes.
    """

    def __init__(self, *, max_lifecycle_events: int, max_output_chars: int) -> None:
        self.max_lifecycle_events = max_lifecycle_events
        self.max_output_chars = max_output_chars
        self.max_raw_frames = max_lifecycle_events + max_output_chars
        self.lifecycle_events = 0
        self.output_chars = 0
        self.raw_frames = 0

    def observe(self, kind: str, output_delta: str = "") -> None:
        self.raw_frames += 1
        if self.raw_frames > self.max_raw_frames:
            raise HTTPException(
                status_code=409,
                detail=_error(
                    "runtime_limit_exceeded",
                    "Hermes emitted too many stream frames",
                    details=self.details(),
                ),
            )
        if kind == "output.delta" and output_delta:
            self.output_chars += len(output_delta)
            if self.output_chars > self.max_output_chars:
                raise HTTPException(
                    status_code=409,
                    detail=_error(
                        "runtime_limit_exceeded",
                        "Hermes output exceeded the configured limit",
                        details=self.details(),
                    ),
                )
            return
        # Empty deltas consume this budget so a peer cannot evade both limits.
        self.lifecycle_events += 1
        if self.lifecycle_events > self.max_lifecycle_events:
            raise HTTPException(
                status_code=409,
                detail=_error(
                    "runtime_limit_exceeded",
                    "Hermes emitted too many lifecycle events",
                    details=self.details(),
                ),
            )

    def details(self) -> dict[str, int]:
        return {
            "lifecycle_event_count": self.lifecycle_events,
            "output_char_count": self.output_chars,
            "raw_frame_count": self.raw_frames,
        }


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
    if name == "tool.failed":
        return "tool.failed"
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


def _normalized_tool_payload(kind: str, payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Normalize pinned-Hermes tool events without retaining argument values."""

    data = dict(payload)
    tool_name = str(data.get("tool_name") or data.get("tool") or data.get("name") or "").strip()
    call_id = str(
        data.get("tool_call_id")
        or data.get("call_id")
        or data.get("request_id")
        or data.get("id")
        or ""
    ).strip()
    arguments = data.get("arguments") or data.get("args") or data.get("input")
    argument_names = sorted(str(key) for key in arguments) if isinstance(arguments, Mapping) else []
    error = data.get("error")
    ok = data.get("ok")
    if ok is None and kind == "tool.completed":
        ok = not bool(error)
    if kind == "tool.completed" and ok is False:
        kind = "tool.failed"
    normalized = {
        **data,
        "tool_name": tool_name or None,
        "tool_call_id": call_id or None,
        "request_id": str(data.get("request_id") or call_id or "") or None,
        "provided_argument_names": argument_names,
        "ok": bool(ok) if ok is not None else None,
        "duration_ms": data.get("duration_ms") or data.get("elapsed_ms"),
        "result_count": data.get("result_count") or data.get("source_count") or 0,
        "source": data.get("source") or "hermes",
        "error": error,
    }
    for key in ("arguments", "args", "input"):
        normalized.pop(key, None)
    return kind, normalized


async def _request_upstream_stop(
    hermes_api_url: str,
    runtime_profile: str,
    upstream_run_id: str,
    headers: Mapping[str, str],
) -> dict[str, Any]:
    """Request a cooperative stop for one exact profile-scoped Hermes run.

    Pinned Hermes acknowledges this operation with ``status=stopping``.  That
    acknowledgement is not terminal: the synchronous agent/provider worker may
    still be running until it observes the hard-interrupt flag.
    """

    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.post(
            hermes_api_url.rstrip("/") + f"/p/{runtime_profile}/v1/runs/{upstream_run_id}/stop",
            headers=dict(headers),
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise httpx.DecodingError(
                "Hermes stop response was not valid JSON", request=response.request
            ) from exc
        if not isinstance(payload, Mapping):
            raise httpx.DecodingError("Hermes stop response was not an object", request=response.request)
        return dict(payload)


async def _confirm_upstream_stop(
    hermes_api_url: str,
    runtime_profile: str,
    upstream_run_id: str,
    headers: Mapping[str, str],
    *,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float | None = None,
) -> dict[str, Any]:
    """Wait for Hermes' cooperative stop to reach an actual terminal state."""

    timeout = max(
        0.0,
        float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("HERMES_STOP_CONFIRM_TIMEOUT_SECONDS", "15")
        ),
    )
    interval = max(
        0.01,
        float(
            poll_interval_seconds
            if poll_interval_seconds is not None
            else os.getenv("HERMES_STOP_POLL_INTERVAL_SECONDS", "0.25")
        ),
    )
    status_url = (
        hermes_api_url.rstrip("/")
        + f"/p/{runtime_profile}/v1/runs/{upstream_run_id}"
    )
    deadline = time.monotonic() + timeout
    last_status = "stopping"
    async with httpx.AsyncClient(timeout=5) as client:
        while True:
            response = await client.get(status_url, headers=dict(headers))
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise httpx.DecodingError(
                    "Hermes run status was not valid JSON", request=response.request
                ) from exc
            if not isinstance(payload, Mapping):
                raise httpx.DecodingError("Hermes run status was not an object", request=response.request)
            last_status = str(payload.get("status") or "unknown").lower()
            if last_status in {"cancelled", "completed", "failed"}:
                return {
                    "confirmed": True,
                    "status": last_status,
                    "last_event": payload.get("last_event"),
                }
            if time.monotonic() >= deadline:
                return {"confirmed": False, "status": last_status}
            await asyncio.sleep(min(interval, max(0.0, deadline - time.monotonic())))


async def _stop_and_confirm_upstream_run(
    hermes_api_url: str,
    runtime_profile: str,
    upstream_run_id: str,
    headers: Mapping[str, str],
) -> dict[str, Any]:
    acknowledgement = await _request_upstream_stop(
        hermes_api_url, runtime_profile, upstream_run_id, headers
    )
    confirmation = await _confirm_upstream_stop(
        hermes_api_url, runtime_profile, upstream_run_id, headers
    )
    return {
        **confirmation,
        "acknowledged_status": str(acknowledgement.get("status") or "unknown"),
    }


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
    profile_manager = RunProfileManager()
    state: dict[str, Any] = {
        "active": {},
        "draining": False,
        "store": store,
        "storage_backend": storage_backend,
        "worker_count": worker_count,
        "storage_healthy": True,
        "profile_manager": profile_manager,
    }
    start_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["draining"] = False
        profile_max_age = int(os.getenv("HERMES_RUN_PROFILE_MAX_AGE_SECONDS", "86400"))
        profile_manager.sweep_stale(max_age_seconds=profile_max_age)
        async def sweep_profiles() -> None:
            interval = max(60, int(os.getenv("HERMES_RUN_PROFILE_SWEEP_INTERVAL_SECONDS", "3600")))
            while True:
                await asyncio.sleep(interval)
                profile_manager.sweep_stale(max_age_seconds=profile_max_age)
        profile_sweeper = asyncio.create_task(sweep_profiles(), name="hermes-profile-sweeper")
        # Recovered records remain inspectable. Their upstream bindings are
        # intentionally not discarded when the gateway process restarts.
        for run_id, record in list(store.records.items()):
            if record.get("status") in {"queued", "running"} and record.get("payload"):
                state["active"][run_id] = asyncio.create_task(
                    _background_run(_recovery_payload(record), None),
                    name=f"hermes-runtime-recovery-{run_id}",
                )
        try:
            yield
        finally:
            state["draining"] = True
            profile_sweeper.cancel()
            try:
                await profile_sweeper
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="AskPDF Hermes Runtime", version=os.getenv("HERMES_RUNTIME_VERSION", "hermes-gateway-1"), lifespan=lifespan)

    def upstream_url() -> str:
        return hermes_api_url.rstrip("/")

    def profile_upstream_url(profile: str | None = None) -> str:
        base = upstream_url()
        return f"{base}/p/{profile}" if profile else base

    def upstream_headers(session_id: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        token = os.getenv("HERMES_API_TOKEN")
        if token:
            headers["authorization"] = f"Bearer {token}"
        if session_id:
            headers["X-Hermes-Session-Id"] = str(session_id)
        return headers

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
        try:
            context_length = configured_context_length()
            provider = configured_provider()
            validate_provider_context(provider, context_length)
            rendered_context_length = _rendered_model_context_length()
            context_ready = rendered_context_length == context_length
        except (OSError, RuntimeError, ValueError):
            context_length = None
            rendered_context_length = None
            context_ready = False
        ready = hermes_ready and mcp_ready and storage_ready and context_ready
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
                    "model_context": {
                        "status": "ok" if context_ready else "failed",
                        "configured_context_length": context_length,
                        "rendered_context_length": rendered_context_length,
                        "provider": provider if context_length is not None else None,
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
        if spec.get("definition_version") not in {1, 2}:
            issues.append({"code": "unsupported_definition_version", "message": "Hermes definitions must use definition_version 1 or 2"})
        config = spec.get("config")
        if not isinstance(config, Mapping):
            issues.append({"code": "missing_config", "message": "Hermes spec requires config"})
        else:
            if config.get("mcp_server") != "askpdf":
                issues.append({"code": "unsupported_mcp_server", "message": "Hermes runtime requires mcp_server=askpdf"})
        allowed_tool_ids = list(config.get("allowed_tool_ids") or []) if isinstance(config, Mapping) else []
        return _envelope(status="ok", request_id=request.headers.get("x-request-id"), result={"validation": {"valid": not issues, "issues": issues, "normalized_spec": spec if not issues else None, "runtime_metadata": {"framework": "hermes", "builder_id": "hermes_agent", "hermes_revision": HERMES_REVISION, "mcp_server": "askpdf", "allowed_tool_ids": sorted(allowed_tool_ids)}}})

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
        resolved_spec = dict(context.get("resolved_spec") or {})
        definition_version = int(resolved_spec.get("definition_version") or 1)
        managed_profile = resolved_spec.get("managed_profile") or {}
        config = resolved_spec.get("config") or {}
        managed_mcp = managed_profile.get("mcp") or {}
        runtime_profile = str(managed_mcp.get("runtime_profile") or "").strip()
        managed_limits = managed_profile.get("limits") or {}
        effective_limits = managed_limits if definition_version >= 2 else config
        max_events = max(1, int(effective_limits.get("max_event_count") or 200))
        max_output_chars = max(1, int(effective_limits.get("max_output_chars") or 12000))
        max_duration_seconds = max(1, int(effective_limits.get("max_duration_seconds") or 300))
        deadline = time.monotonic() + max_duration_seconds
        system_prompt = str((managed_profile.get("instructions") if definition_version >= 2 else config.get("system_prompt")) or "").strip()
        task_context = input_data.get("task_context")
        context_token = str(input_data.get("mcp_execution_context_token") or "").strip()
        if isinstance(task_context, Mapping):
            question = question + "\n\naskPDF task context:\n" + json.dumps(task_context, sort_keys=True, ensure_ascii=False)
        upstream_payload = {
            "input": question,
            "instructions": system_prompt or None,
            "model": ((managed_profile.get("model_policy") or {}).get("model") if definition_version >= 2 else options.get("llm_model") or config.get("model")) or "",
            "provider": ((managed_profile.get("model_policy") or {}).get("provider") if definition_version >= 2 else options.get("llm_provider") or config.get("provider")) or "custom",
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
        event_budget = _HermesEventBudget(
            max_lifecycle_events=max_events,
            max_output_chars=max_output_chars,
        )
        terminal_seen = False
        retain_profile = False
        tool_activity = {
            "started": 0,
            "completed": 0,
            "failed": 0,
            "evidence_result_count": 0,
            "last_tool_name": None,
            "last_tool_call_id": None,
        }

        def process_frame(frame_event_name: str, frame_data: list[str]) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
            nonlocal sequence, terminal_seen, session_id
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
            elif kind in {"tool.started", "tool.completed", "tool.failed"}:
                kind, event_payload = _normalized_tool_payload(kind, event_payload)
                if kind == "tool.started":
                    tool_activity["started"] += 1
                elif kind == "tool.completed":
                    tool_activity["completed"] += 1
                    tool_activity["evidence_result_count"] += max(0, int(event_payload.get("result_count") or 0))
                else:
                    tool_activity["failed"] += 1
                tool_activity["last_tool_name"] = event_payload.get("tool_name")
                tool_activity["last_tool_call_id"] = event_payload.get("tool_call_id")
            terminal = kind in HERMES_TERMINAL_EVENTS
            output_delta = str(event_payload.get("content") or event_payload.get("delta") or event_payload.get("text") or "") if kind == "output.delta" else ""
            event_budget.observe(kind, output_delta)
            event = _neutral_event(run_id, sequence, kind, event_payload, source_event_id=source_event_id, terminal=terminal, continuation=continuation)
            result = None
            if kind == "approval.request":
                interrupt_id = str(event_payload.get("approval_id") or event_payload.get("id") or f"hermes-approval-{sequence}")
                result = {
                    "status": "awaiting_human",
                    "pending_interrupt": {
                        "interrupt_id": interrupt_id,
                        "type": "hermes_approval",
                        "title": str(event_payload.get("title") or "Hermes tool approval required"),
                        "description": str(event_payload.get("command") or event_payload.get("description") or ""),
                        "allowed_actions": ["approve", "reject"],
                        "runtime_approval_choices": ["once", "session", "always", "deny"],
                        "checkpoint_resume": True,
                        "runtime_payload": dict(event_payload),
                    },
                    "continuation": continuation,
                }
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
                        "mcp_server": managed_mcp.get("server") if definition_version >= 2 else config.get("mcp_server"),
                        "allowed_tool_ids": list(managed_mcp.get("allowed_tool_ids") or []) if definition_version >= 2 else list(config.get("allowed_tool_ids") or []),
                        "policy_fingerprint": managed_profile.get("profile_id"),
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

        binding_payload = ((neutral_request.get("continuation") or {}).get("payload") or {})
        execution_profile = str(binding_payload.get("runtime_profile") or "")
        run_profile: RunProfile | None = None
        try:
            if definition_version >= 2 and not profile_manager.is_reusable(execution_profile):
                run_profile = profile_manager.create(
                    run_id=run_id,
                    managed_profile=managed_profile,
                    context_token=context_token,
                )
                execution_profile = run_profile.name
                if not profile_manager.verify(run_profile):
                    raise HTTPException(status_code=502, detail=_error(
                        "hermes_profile_preflight_failed",
                        "Hermes run profile could not be verified",
                        retryable=True,
                        details={"stage": "profile_identity", "reason": "fingerprint_mismatch", "profile_digest": run_profile.config_fingerprint},
                    ))
            elif not execution_profile:
                # Version-1 runs predate task-scoped MCP credentials. Preserve
                # their frozen static profile instead of changing them in place.
                execution_profile = runtime_profile
            async with httpx.AsyncClient(timeout=_upstream_timeout(max_duration_seconds)) as client:
                if run_profile is not None:
                    preflight_url = profile_upstream_url(execution_profile) + "/v1/toolsets"
                    try:
                        preflight = await client.get(preflight_url, headers=headers)
                        preflight.raise_for_status()
                        toolset_payload = preflight.json()
                        activation = (
                            toolset_payload.get("askpdf_runtime_profile")
                            if isinstance(toolset_payload, Mapping)
                            else None
                        )
                        if not isinstance(activation, Mapping):
                            raise ValueError("profile activation metadata missing")
                        registered_tools = {
                            str(value) for value in activation.get("registered_tools") or []
                        }
                        expected_registered = {
                            f"mcp__{run_profile.mcp_server_name}__{tool_name}"
                            for tool_name in run_profile.expected_tools
                        }
                        activation_mismatches = []
                        if str(activation.get("name") or "") != run_profile.name:
                            activation_mismatches.append("profile_name")
                        if str(activation.get("config_fingerprint") or "") != run_profile.activation_fingerprint:
                            activation_mismatches.append("config_fingerprint")
                        if set(str(value) for value in activation.get("mcp_server_names") or []) != {run_profile.mcp_server_name}:
                            activation_mismatches.append("mcp_server_names")
                        header_digests = activation.get("mcp_context_header_sha256") or {}
                        if (
                            not isinstance(header_digests, Mapping)
                            or str(header_digests.get(run_profile.mcp_server_name) or "") != run_profile.token_digest
                        ):
                            activation_mismatches.append("mcp_context_header")
                        askpdf_registered = {
                            value for value in registered_tools
                            if value.startswith(f"mcp__{run_profile.mcp_server_name}__")
                        }
                        if askpdf_registered != expected_registered:
                            activation_mismatches.append("registered_tools")
                        if activation_mismatches:
                            logger.warning(
                                "Hermes profile activation mismatch profile=%s fields=%s expected_tools=%s registered_tools=%s",
                                run_profile.name,
                                activation_mismatches,
                                sorted(expected_registered),
                                sorted(registered_tools),
                            )
                            raise ValueError("profile activation identity or tools mismatch")
                    except (httpx.HTTPError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        status_code = getattr(getattr(exc, "response", None), "status_code", None)
                        raise HTTPException(status_code=502, detail=_error(
                            "hermes_profile_preflight_failed",
                            "Hermes run profile could not activate its MCP tools",
                            retryable=True,
                            details={
                                "stage": "tool_discovery",
                                "reason": "activation_mismatch" if isinstance(exc, ValueError) else "transport_or_protocol",
                                "http_status": status_code,
                                "profile_digest": run_profile.config_fingerprint,
                                "expected_tools": list(run_profile.expected_tools),
                            },
                        )) from exc
                    try:
                        mcp_response = await client.get(
                            "http://rag-service:8000/internal/hermes-mcp/preflight",
                            headers={
                                "x-askpdf-execution-context": context_token,
                                "x-askpdf-expected-run-id": run_id,
                                "x-askpdf-expected-thread-id": str(neutral_request.get("thread_id") or ""),
                                "x-askpdf-expected-task-id": str(neutral_request.get("task_id") or ""),
                            },
                        )
                        mcp_response.raise_for_status()
                        mcp_payload = mcp_response.json()
                        if not isinstance(mcp_payload, Mapping) or mcp_payload.get("status") != "ok" or str(mcp_payload.get("run_id") or "") != run_id:
                            raise ValueError("MCP preflight returned an error")
                    except (httpx.HTTPError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        status_code = getattr(getattr(exc, "response", None), "status_code", None)
                        raise HTTPException(status_code=502, detail=_error(
                            "hermes_profile_preflight_failed",
                            "Hermes run profile could not activate its MCP tools",
                            retryable=True,
                            details={"stage": "mcp_context", "reason": "context_rejected", "http_status": status_code, "profile_digest": run_profile.config_fingerprint},
                        )) from exc
                    yield _sse(_neutral_event(run_id, sequence, "operation.completed", {
                        "operation_id": "hermes_profile_preflight", "operation_type": "runtime_preflight",
                        "operation_label": "Hermes profile and MCP preflight", "visit_index": 1,
                        "profile_digest": run_profile.config_fingerprint, "token_digest": run_profile.token_digest,
                        "token_expires_at": run_profile.token_expires_at, "tool_count": len(run_profile.expected_tools),
                    }))
                    sequence += 1
                existing_binding = (neutral_request.get("continuation") or {}).get("payload") or {}
                upstream_run_id = str(existing_binding.get("upstream_run_id") or "")
                if not upstream_run_id:
                    response = await client.post(profile_upstream_url(execution_profile) + "/v1/runs", headers=headers, json=upstream_payload)
                    response.raise_for_status()
                    start = response.json()
                    response_session_id = _response_session_id(start)
                    if not session_id and response_session_id:
                        session_id = response_session_id
                    upstream_run_id = str(start.get("run_id") or start.get("id") or "")
                if not upstream_run_id:
                    raise HTTPException(status_code=502, detail=_error("runtime_protocol_error", "Hermes did not return an upstream run ID"))
                if not session_id:
                    status_response = await client.get(profile_upstream_url(execution_profile) + f"/v1/runs/{upstream_run_id}", headers=headers)
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
                    "payload": {
                        "session_id": session_id,
                        "upstream_run_id": upstream_run_id,
                        "runtime_profile": execution_profile,
                        "policy_profile": runtime_profile,
                        **(run_profile.continuation_metadata() if run_profile is not None else {}),
                    },
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
                async with client.stream("GET", profile_upstream_url(execution_profile) + f"/v1/runs/{upstream_run_id}/events", headers=headers) as events_response:
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
                                if result is not None and result.get("status") == "awaiting_human":
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
                        if result is not None and result.get("status") == "awaiting_human":
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
            if error.get("code") == "runtime_limit_exceeded":
                stop_status = "not_attempted"
                stop_error_type = None
                if upstream_run_id and execution_profile:
                    try:
                        stop_result = await _stop_and_confirm_upstream_run(
                            hermes_api_url,
                            execution_profile,
                            upstream_run_id,
                            headers,
                        )
                        stop_status = str(stop_result["status"])
                        if not stop_result["confirmed"]:
                            stop_status = "unconfirmed"
                            retain_profile = True
                    except httpx.HTTPError as stop_exc:
                        stop_status = "failed"
                        stop_error_type = type(stop_exc).__name__
                        retain_profile = True
                details = dict(error.get("details") or {})
                details.update({
                    "configured_limit_seconds": max_duration_seconds,
                    "elapsed_ms": round((time.monotonic() - (deadline - max_duration_seconds)) * 1000, 2),
                    "upstream_stop_status": stop_status,
                    "tool_activity": dict(tool_activity),
                })
                if stop_error_type:
                    details["upstream_stop_error_type"] = stop_error_type
                error = {**dict(error), "details": details}
                logger.warning(
                    "Hermes runtime limit reached | run_id=%s upstream_run_id=%s profile=%s details=%s",
                    run_id,
                    upstream_run_id or None,
                    runtime_profile,
                    dict(error.get("details") or {}),
                )
            event = _neutral_event(run_id, sequence, "run.failed", {"error": error}, terminal=True, continuation=continuation)
            terminal_seen = True
            yield _sse(event, {"status": "failed", "error": error, "continuation": continuation})
        except httpx.HTTPError as exc:
            if terminal_seen:
                return
            phase = "event_stream" if upstream_run_id else "run_start"
            is_timeout = isinstance(exc, httpx.TimeoutException)
            safe_message = (
                "Hermes did not produce an event before the task execution timeout"
                if is_timeout
                else "Hermes runtime is unavailable"
            )
            details: dict[str, Any] = {
                "phase": phase,
                "error_type": type(exc).__name__,
            }
            if is_timeout and upstream_run_id and execution_profile:
                try:
                    stop_result = await _stop_and_confirm_upstream_run(
                        hermes_api_url,
                        execution_profile,
                        upstream_run_id,
                        headers,
                    )
                    details["upstream_stop_status"] = (
                        stop_result["status"]
                        if stop_result["confirmed"]
                        else "unconfirmed"
                    )
                    if not stop_result["confirmed"]:
                        retain_profile = True
                except httpx.HTTPError as stop_exc:
                    details["upstream_stop_status"] = "failed"
                    details["upstream_stop_error_type"] = type(stop_exc).__name__
                    retain_profile = True
            error = _error(
                "hermes_upstream_timeout" if is_timeout else "hermes_upstream_error",
                safe_message,
                retryable=True,
                details=details,
            )
            logger.warning(
                "Hermes upstream request failed | run_id=%s upstream_run_id=%s profile=%s phase=%s error_type=%s",
                run_id,
                upstream_run_id or None,
                runtime_profile,
                phase,
                type(exc).__name__,
            )
            event = _neutral_event(run_id, sequence, "run.failed", {"error": error}, terminal=True, continuation=continuation)
            terminal_seen = True
            yield _sse(event, {"status": "failed", "error": error, "continuation": continuation})
        finally:
            # Approval is a resumable boundary. Keep the credential-bearing
            # profile active until a terminal result, cancellation, or expiry.
            if terminal_seen and not retain_profile:
                profile_manager.retire(run_profile or execution_profile)
    async def _background_run(payload: Mapping[str, Any], request: Request | None) -> None:
        run_id = str((payload.get("request") or {}).get("run_id") or "")
        try:
            async for frame in stream_run(payload, request):
                terminal_status = None
                if '"status":"awaiting_human"' in frame:
                    terminal_status = "awaiting_human"
                elif "event: run.completed" in frame:
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
            if record.get("status") in {"completed", "failed", "cancelled", "awaiting_human"}:
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
    async def continue_run(run_id: str, payload: Mapping[str, Any], request: Request) -> StreamingResponse:
        record = state["store"].records.get(run_id)
        if record is None or record.get("status") != "awaiting_human":
            raise HTTPException(status_code=409, detail=_error("runtime_continuation_unavailable", "Hermes run is not awaiting continuation"))
        continuation = _binding(payload)
        profile = str((((continuation or {}).get("payload") or {}).get("runtime_profile") or ""))
        if not profile_manager.is_reusable(profile):
            raise HTTPException(status_code=409, detail=_error("runtime_profile_expired", "Hermes continuation profile has expired"))
        state["store"].update(run_id, status="running", payload=dict(payload))

        async def continued() -> AsyncIterator[str]:
            async for frame in stream_run(payload, request, expected_run_id=run_id):
                status = None
                if '"status":"awaiting_human"' in frame:
                    status = "awaiting_human"
                elif "event: run.completed" in frame:
                    status = "completed"
                elif "event: run.failed" in frame:
                    status = "failed"
                elif "event: run.cancelled" in frame:
                    status = "cancelled"
                if status:
                    state["store"].finalize(run_id, frame, status=status)
                else:
                    state["store"].append(run_id, frame)
                yield frame
        return StreamingResponse(continued(), media_type="text/event-stream")

    @app.post("/v1/runs/{run_id}/cancel")
    async def cancel(run_id: str, request: Request, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            payload = payload or {}
            upstream_run_id = _upstream_run_id(run_id, payload)
            binding = _binding(payload)
            session_id = ((binding or {}).get("payload") or {}).get("session_id")
            runtime_profile = ((binding or {}).get("payload") or {}).get("runtime_profile")
            headers = upstream_headers(session_id)
            stop_result = await _stop_and_confirm_upstream_run(
                hermes_api_url,
                str(runtime_profile or ""),
                upstream_run_id,
                headers,
            )
            if not stop_result["confirmed"]:
                return _envelope(
                    status="failed",
                    error=_error(
                        "hermes_stop_unconfirmed",
                        "Hermes accepted cancellation but the upstream run is still stopping",
                        retryable=True,
                        details={
                            "run_id": run_id,
                            "upstream_run_id": upstream_run_id,
                            "upstream_status": stop_result["status"],
                            "acknowledged_status": stop_result["acknowledged_status"],
                        },
                    ),
                    request_id=request.headers.get("x-request-id"),
                )
            profile_manager.retire(str(runtime_profile or ""))
            return _envelope(
                status="ok",
                request_id=request.headers.get("x-request-id"),
                result={
                    "run_id": run_id,
                    "upstream_run_id": upstream_run_id,
                    "status": "cancelled" if stop_result["status"] == "cancelled" else "already_terminal",
                    "upstream_status": stop_result["status"],
                },
            )
        except httpx.HTTPError as exc:
            return _envelope(status="failed", error=_error("hermes_cancel_failed", str(exc), retryable=True), request_id=request.headers.get("x-request-id"))

    @app.post("/v1/runs/{run_id}/inspect")
    async def inspect(run_id: str, request: Request, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            payload = payload or {}
            upstream_run_id = _upstream_run_id(run_id, payload)
            binding = _binding(payload)
            session_id = ((binding or {}).get("payload") or {}).get("session_id")
            runtime_profile = ((binding or {}).get("payload") or {}).get("runtime_profile")
            headers = upstream_headers(session_id)
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(profile_upstream_url(runtime_profile) + f"/v1/runs/{upstream_run_id}", headers=headers)
                response.raise_for_status()
            return _envelope(status="ok", request_id=request.headers.get("x-request-id"), result={**dict(response.json()), "run_id": run_id, "upstream_run_id": upstream_run_id})
        except httpx.HTTPError as exc:
            return _envelope(status="failed", error=_error("hermes_inspect_failed", str(exc), retryable=True), request_id=request.headers.get("x-request-id"))

    async def _forward_control(run_id: str, request: Request, payload: Mapping[str, Any], operation: str, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            upstream_run_id = _upstream_run_id(run_id, payload)
            binding = _binding(payload)
            session_id = ((binding or {}).get("payload") or {}).get("session_id")
            runtime_profile = ((binding or {}).get("payload") or {}).get("runtime_profile")
            headers = upstream_headers(session_id)
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(profile_upstream_url(runtime_profile) + f"/v1/runs/{upstream_run_id}/{operation}", headers=headers, json=dict(body))
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
