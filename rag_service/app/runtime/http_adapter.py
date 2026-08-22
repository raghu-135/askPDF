"""Shared HTTP/SSE client for separately deployable runtimes.

The transport is framework-neutral; concrete adapters override only identity,
endpoint configuration, and framework-specific capability/result translation.
"""

from __future__ import annotations

import os
import hashlib
import json
import inspect
import asyncio
from typing import Any, Mapping

import httpx

from app.runtime.adapter import AgentRuntimeAdapter, AgentRuntimeEventSink, RuntimeExecutionContext
from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    RuntimeCapabilities,
    RuntimeValidationResult,
)
from app.runtime.errors import RuntimeError
from app.runtime.transport import (
    capabilities_from_dict,
    event_from_dict,
    result_from_dict,
    sse_encode,
    validation_from_dict,
    iter_sse,
)


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items() if str(key) != "request"}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if hasattr(value, "model_dump"):
        return _safe_json(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return _safe_json(vars(value))
    return str(value)


def context_to_dict(context: RuntimeExecutionContext) -> dict[str, Any]:
    """Serialize only execution inputs; repositories and writers never cross the wire."""

    request_payload = _safe_json(context.request)
    if isinstance(request_payload, Mapping):
        request_payload = dict(request_payload)
        # The external runtime must select its MCP-backed, persistence-free
        # context loader. This is transport metadata, not product state.
        request_payload.setdefault("runtime_execution_mode", True)

    return {
        "embedding_model": context.embedding_model,
        # Task execution historically carried request-only fields (objective,
        # limits, model selection, and tool policy) on the legacy request
        # object. Preserve that input explicitly across the HTTP boundary;
        # it is execution input, not a product repository or writer.
        "request_payload": request_payload,
        "resolved_spec": _safe_json(context.resolved_spec),
        "agent_run_context": _safe_json(context.agent_run_context),
        "task_id": context.task_id,
        "task_worker_id": context.task_worker_id,
    }


def _structured_runtime_error(payload: Any) -> tuple[dict[str, Any], Mapping[str, Any]] | None:
    """Extract a neutral runtime error from either supported HTTP shape."""
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    if not isinstance(error, Mapping):
        detail = payload.get("detail")
        error = detail if isinstance(detail, Mapping) else None
    if not isinstance(error, Mapping) or not error.get("code"):
        return None
    return dict(error), payload


def _raise_structured_runtime_error(payload: Any) -> None:
    decoded = _structured_runtime_error(payload)
    if decoded is None:
        return
    error, envelope = decoded
    raise RuntimeError(
        code=str(error.get("code") or "runtime_failed"),
        safe_message=str(error.get("safe_message") or error.get("message") or "Agent runtime failed"),
        retryable=bool(error.get("retryable")),
        details=dict(error.get("details") or {}),
        runtime_metadata=dict(envelope.get("runtime_metadata") or {}),
    )


class HttpRuntimeAdapter(AgentRuntimeAdapter):
    framework = "langgraph"
    builder_id = "langgraph_graph"

    def __init__(
        self,
        base_url: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("LANGGRAPH_RUNTIME_URL", "http://langgraph-runtime:8100")).rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._timeout = httpx.Timeout(
            read_timeout or float(os.getenv("AGENT_RUNTIME_READ_TIMEOUT_SECONDS", "30")),
            connect=connect_timeout or float(os.getenv("AGENT_RUNTIME_CONNECT_TIMEOUT_SECONDS", "5")),
            write=float(os.getenv("AGENT_RUNTIME_WRITE_TIMEOUT_SECONDS", "10")),
        )
        self._execution_timeout = float(os.getenv("AGENT_RUNTIME_EXECUTION_TIMEOUT_SECONDS", "3600"))

    async def _client_for_request(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self, request: AgentRuntimeRequest | None = None) -> dict[str, str]:
        headers = {"accept": "application/json", "x-runtime-contract-version": str(request.contract_version if request else 1)}
        if request is not None:
            headers["x-agent-run-id"] = request.run_id
            if request.trace_id:
                headers["x-request-id"] = request.trace_id
            traceparent = request.options.get("traceparent")
            if traceparent:
                headers["traceparent"] = str(traceparent)
            tracestate = request.options.get("tracestate")
            if tracestate:
                headers["tracestate"] = str(tracestate)
            if request.authentication.get("token"):
                headers["authorization"] = str(request.authentication["token"])
            if request.permissions:
                headers["x-agent-permissions"] = json.dumps(dict(request.permissions), separators=(",", ":"))
        token = os.getenv("LANGGRAPH_RUNTIME_TOKEN")
        if token:
            headers["authorization"] = f"Bearer {token}"
        return headers

    async def _json(self, method: str, path: str, *, request: AgentRuntimeRequest | None = None, **kwargs: Any) -> Any:
        try:
            response = await (await self._client_for_request()).request(method, self.base_url + path, headers=self._headers(request), **kwargs)
        except httpx.TimeoutException as exc:
            raise RuntimeError.from_exception(exc, code="runtime_timeout", retryable=True, safe_message="Agent runtime timed out") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError.from_exception(exc, code="runtime_transport_error", retryable=True, safe_message="Agent runtime is unavailable") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            if response.status_code >= 400:
                raise RuntimeError.from_exception(exc, code="runtime_transport_error", retryable=True, safe_message="Agent runtime is unavailable") from exc
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned invalid JSON") from exc
        _raise_structured_runtime_error(payload)
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError.from_exception(exc, code="runtime_transport_error", retryable=True, safe_message="Agent runtime is unavailable") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned an invalid response")
        if int(payload.get("contract_version") or 1) != 1:
            raise RuntimeError("runtime_contract_unsupported", "Agent runtime contract version is unsupported")
        if payload.get("error"):
            error = payload["error"]
            raise RuntimeError(
                code=str(error.get("code") or "runtime_failed"),
                safe_message=str(error.get("safe_message") or "Agent runtime failed"),
                retryable=bool(error.get("retryable")),
                details=dict(error.get("details") or {}),
                runtime_metadata=dict(payload.get("runtime_metadata") or {}),
            )
        return payload.get("result") if "result" in payload else payload

    async def capabilities(self, definition: AgentDefinition) -> RuntimeCapabilities:
        value = await self._json("GET", "/v1/capabilities")
        try:
            return capabilities_from_dict(value.get("capabilities") or value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned malformed capabilities") from exc

    async def validate(self, definition: AgentDefinition, spec: Mapping[str, Any], *, options: Mapping[str, Any] | None = None) -> RuntimeValidationResult:
        value = await self._json("POST", "/v1/validate", json={"definition": definition.to_dict(), "spec": _safe_json(spec), "options": _safe_json(options or {})})
        return validation_from_dict(value.get("validation") or value)

    async def _stream(self, path: str, request: AgentRuntimeRequest, *, context: RuntimeExecutionContext, payload: Mapping[str, Any] | None, event_sink: AgentRuntimeEventSink | None) -> AgentRuntimeResult:
        body = {"request": request.to_dict(), "context": context_to_dict(context), **dict(payload or {})}
        seen: dict[str, str] = {}
        terminal: AgentRuntimeResult | None = None
        terminal_hash: str | None = None
        terminal_event_id: str | None = None
        last_sequence = 0
        last_event_id: str | None = None

        async def consume(method: str, stream_path: str, *, replay: bool = False) -> None:
            nonlocal terminal, terminal_hash, terminal_event_id, last_sequence
            nonlocal last_event_id
            client = await self._client_for_request()
            headers = {**self._headers(request), "accept": "text/event-stream"}
            params: dict[str, Any] | None = None
            if replay:
                params = {"after_sequence": last_sequence}
                if last_event_id:
                    headers["last-event-id"] = last_event_id
            kwargs: dict[str, Any] = {"headers": headers}
            if method == "POST":
                kwargs["json"] = body
            if params:
                kwargs["params"] = params
            async with client.stream(method, self.base_url + stream_path, **kwargs) as response:
                if response.status_code >= 400:
                    try:
                        failure = await response.aread()
                        envelope = json.loads(failure)
                    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                        envelope = {}
                    _raise_structured_runtime_error(envelope)
                response.raise_for_status()
                async for _name, item in iter_sse(response):
                    envelope = item["data"]
                    if not isinstance(envelope, Mapping):
                        raise RuntimeError("runtime_protocol_error", "Agent runtime returned a malformed event envelope")
                    event_payload = envelope.get("event") or envelope
                    if not isinstance(event_payload, Mapping):
                        raise RuntimeError("runtime_protocol_error", "Agent runtime returned a malformed event")
                    event = event_from_dict(event_payload)
                    if event.run_id != request.run_id:
                        raise RuntimeError("runtime_protocol_error", "Agent runtime returned a mismatched run ID")
                    if event.contract_version != request.contract_version:
                        raise RuntimeError("runtime_contract_unsupported", "Agent runtime event contract version is unsupported")
                    event_hash = hashlib.sha256(json.dumps(event.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
                    if event.event_id in seen:
                        if seen[event.event_id] != event_hash:
                            raise RuntimeError("runtime_protocol_error", "Agent runtime returned conflicting duplicate event IDs")
                        continue
                    seen[event.event_id] = event_hash
                    if event.sequence <= last_sequence:
                        raise RuntimeError("runtime_protocol_error", "Agent runtime event sequence is not monotonic")
                    last_sequence = event.sequence
                    last_event_id = event.event_id
                    if event_sink is not None:
                        emit = getattr(event_sink, "emit", None)
                        if emit is not None:
                            await self._emit_to_sink(emit, event)
                        if event.continuation is not None:
                            persist_binding = getattr(event_sink, "persist_runtime_binding", None)
                            if persist_binding is not None:
                                await persist_binding(request.run_id, event.continuation)
                    if envelope.get("result") is not None:
                        resumable_boundary = (
                            isinstance(envelope.get("result"), Mapping)
                            and str(envelope["result"].get("status") or "") in {"awaiting_human", "paused"}
                        )
                        if not event.terminal and not resumable_boundary:
                            raise RuntimeError("runtime_protocol_error", "Agent runtime attached a result to a nonterminal event")
                        if terminal_event_id is not None and event.event_id != terminal_event_id:
                            raise RuntimeError("runtime_protocol_error", "Agent runtime returned more than one terminal result")
                        candidate = result_from_dict(envelope["result"])
                        candidate_hash = hashlib.sha256(json.dumps(candidate.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
                        if terminal_hash is not None and candidate_hash != terminal_hash:
                            raise RuntimeError("runtime_protocol_error", "Agent runtime returned conflicting terminal results")
                        terminal = candidate
                        terminal_hash = candidate_hash
                        terminal_event_id = event.event_id

        reconnect_attempts = max(0, int(os.getenv("AGENT_RUNTIME_RECONNECT_MAX_ATTEMPTS", "5")))
        reconnect_backoff = max(0.01, float(os.getenv("AGENT_RUNTIME_RECONNECT_BACKOFF_SECONDS", "0.25")))
        reconnect_deadline = min(
            self._execution_timeout,
            max(1.0, float(os.getenv("AGENT_RUNTIME_RECONNECT_DEADLINE_SECONDS", "30"))),
        )
        reconnect_started = asyncio.get_running_loop().time()
        reconnect_count = 0
        try:
            try:
                await asyncio.wait_for(consume("POST", path), timeout=self._execution_timeout)
            except httpx.HTTPError:
                # The runtime may have committed the execution even though
                # the subscriber connection failed. Fall through to durable
                # replay before surfacing a transport error.
                pass
            while terminal is None:
                if reconnect_count >= reconnect_attempts or asyncio.get_running_loop().time() - reconnect_started >= reconnect_deadline:
                    raise RuntimeError("runtime_stream_error", "Agent runtime stream ended before a terminal result", retryable=True)
                await asyncio.sleep(min(reconnect_backoff * (2 ** reconnect_count), 5.0))
                reconnect_count += 1
                try:
                    await asyncio.wait_for(
                        consume("GET", f"/v1/runs/{request.run_id}/events", replay=True),
                        timeout=max(1.0, reconnect_deadline),
                    )
                except httpx.HTTPStatusError as exc:
                    # The initial POST may have been lost before the runtime
                    # committed its durable record. Retry the same idempotent
                    # operation once the replay endpoint reports not-found.
                    if exc.response.status_code == 404 and last_sequence == 0:
                        await asyncio.wait_for(consume("POST", path), timeout=self._execution_timeout)
                    else:
                        raise
        except RuntimeError:
            raise
        except asyncio.TimeoutError as exc:
            raise RuntimeError("runtime_execution_timeout", "Agent runtime execution timed out", retryable=True) from exc
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned malformed SSE data") from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError.from_exception(exc, code="runtime_stream_timeout", retryable=True, safe_message="Agent runtime stream timed out") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError.from_exception(exc, code="runtime_stream_error", retryable=True, safe_message="Agent runtime stream failed") from exc
        if terminal is None:
            raise RuntimeError("runtime_protocol_error", "Agent runtime stream ended without a terminal result")
        return terminal

    @staticmethod
    async def _emit_to_sink(emit: Any, event: AgentRuntimeEvent) -> None:
        """Bridge neutral events to the existing legacy SSE sink when needed."""

        owner = getattr(emit, "__self__", None)
        emit_runtime_event = getattr(owner, "emit_runtime_event", None)
        if emit_runtime_event is not None:
            await emit_runtime_event(event)
            return

        try:
            parameters = inspect.signature(emit).parameters.values()
            positional = [
                parameter
                for parameter in parameters
                if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            accepts_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
        except (TypeError, ValueError):
            positional = []
            accepts_varargs = False

        if accepts_varargs or len(positional) >= 2:
            await emit(event.kind, dict(event.payload))
        else:
            await emit(event)

    async def start(self, request: AgentRuntimeRequest, *, context: RuntimeExecutionContext, event_sink: AgentRuntimeEventSink | None = None) -> AgentRuntimeResult:
        return await self._stream("/v1/runs/start", request, context=context, payload=None, event_sink=event_sink)

    async def resume(self, request: AgentRuntimeRequest, *, interrupt: Mapping[str, Any], context: RuntimeExecutionContext, event_sink: AgentRuntimeEventSink | None = None) -> AgentRuntimeResult:
        return await self._stream("/v1/runs/%s/resume" % request.run_id, request, context=context, payload={"interrupt": _safe_json(interrupt)}, event_sink=event_sink)

    async def continue_run(self, request: AgentRuntimeRequest, *, context: RuntimeExecutionContext, event_sink: AgentRuntimeEventSink | None = None) -> AgentRuntimeResult | None:
        result = await self._stream("/v1/runs/%s/continue" % request.run_id, request, context=context, payload=None, event_sink=event_sink)
        if result.status == "no_continuation":
            return None
        return result

    async def cancel(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        value = await self._json("POST", "/v1/runs/%s/cancel" % request.run_id, request=request, json={"request": request.to_dict()})
        return dict(value) if isinstance(value, Mapping) else {"result": value}

    async def inspect_state(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        value = await self._json("POST", "/v1/runs/%s/inspect" % request.run_id, request=request, json={"request": request.to_dict()})
        return dict(value or {}) if isinstance(value, Mapping) else {}

    async def project_trace(self, events: list[Mapping[str, Any]], *, run_id: str, context: RuntimeExecutionContext | None = None) -> list[AgentRuntimeEvent]:
        return [event_from_dict(event) for event in events]


class HttpLangGraphRuntimeAdapter(HttpRuntimeAdapter):
    """LangGraph-specific HTTP operations layered on the neutral transport."""

    framework = "langgraph"
    builder_id = "langgraph_graph"

    async def delete_continuation(self, continuation: Any) -> Any:
        binding_id = str(continuation.payload.get("binding_id") or continuation.payload.get("checkpoint_thread_id") or "")
        return await self._json("DELETE", "/v1/continuations/%s" % binding_id, json={"continuation": continuation.to_dict()})
