"""Shared HTTP/SSE client for separately deployable runtimes.

The transport is framework-neutral; concrete adapters override only identity,
endpoint configuration, and framework-specific capability/result translation.
"""

from __future__ import annotations

import os
import hashlib
import json
import asyncio
import time
from dataclasses import replace
from typing import Any, Mapping

import httpx

from app.runtime.adapter import AgentRuntimeAdapter, AgentRuntimeEventSink, RuntimeInvocationContext
from app.runtime.budgets import deep_agent_budgets
from app.runtime.catalog import definition_metadata_from_spec
from runtime_protocol.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    RuntimeCapabilities,
    RuntimeCourseCorrection,
    RuntimeCourseCorrectionReceipt,
    RuntimeOperationId,
    RuntimeValidationResult,
)
from runtime_protocol.errors import RuntimeError
from app.runtime.operational_limits import required_positive_float, required_positive_int
from runtime_protocol.transport import (
    capabilities_from_dict,
    course_correction_receipt_from_dict,
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
    raise TypeError(f"runtime wire values must be JSON-compatible, got {type(value).__name__}")


def context_to_dict(context: RuntimeInvocationContext) -> dict[str, Any]:
    """Serialize only execution inputs; repositories and writers never cross the wire."""

    request_payload = _safe_json(context.request_payload)
    if isinstance(request_payload, Mapping):
        request_payload = dict(request_payload)
        # The external runtime must select its MCP-backed, persistence-free
        # context loader. This is transport metadata, not product state.
        request_payload.setdefault("runtime_execution_mode", True)

    return {
        "embedding_model": context.embedding_model,
        # Objective, limits, model selection, and tool policy are explicit
        # execution inputs, not product repositories or writers.
        "request_payload": request_payload,
        "resolved_spec": _safe_json(context.resolved_spec),
        "agent_run_context": _safe_json(context.agent_run_context),
        "task_id": context.task_id,
        "task_worker_id": context.task_worker_id,
        "task_context": context.task_context.to_dict() if context.task_context is not None else None,
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


class RuntimeTransportConnector:
    """HTTP/SSE transport connector shared by concrete runtime adapters.

    This class owns only transport mechanics. Framework adapters select
    endpoints and translate the neutral product contracts around it.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        framework: str = "langgraph",
        authorization_env: str = "LANGGRAPH_RUNTIME_TOKEN",
        authorization_envs: tuple[str, ...] | None = None,
        visualization_id: str | None = None,
        replay_by_event_id: bool = False,
    ) -> None:
        self.framework = framework
        self.authorization_env = authorization_env
        self.authorization_envs = authorization_envs or (authorization_env,)
        self.visualization_id = visualization_id
        self.replay_by_event_id = replay_by_event_id
        configured_base_url = base_url or os.getenv("LANGGRAPH_RUNTIME_URL", "").strip()
        if not configured_base_url:
            raise RuntimeError("runtime_configuration_invalid", "LANGGRAPH_RUNTIME_URL is required for the external runtime")
        self.base_url = configured_base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._timeout = httpx.Timeout(
            read_timeout or required_positive_float("AGENT_RUNTIME_READ_TIMEOUT_SECONDS"),
            connect=connect_timeout or required_positive_float("AGENT_RUNTIME_CONNECT_TIMEOUT_SECONDS"),
            write=required_positive_float("AGENT_RUNTIME_WRITE_TIMEOUT_SECONDS"),
        )
        self._execution_timeout = float(deep_agent_budgets(self.framework)["max_duration_seconds"])
        self._reconnect_attempts = required_positive_int("AGENT_RUNTIME_RECONNECT_MAX_ATTEMPTS")
        self._reconnect_backoff = required_positive_float("AGENT_RUNTIME_RECONNECT_BACKOFF_SECONDS")
        self._reconnect_deadline = min(
            self._execution_timeout,
            required_positive_float("AGENT_RUNTIME_RECONNECT_DEADLINE_SECONDS"),
        )
        self._output_delta_flush_seconds = required_positive_float("AGENT_RUNTIME_OUTPUT_DELTA_FLUSH_SECONDS")
        self._output_delta_flush_bytes = required_positive_int("AGENT_RUNTIME_OUTPUT_DELTA_FLUSH_BYTES")
        if self.framework == "langgraph" and not any(os.getenv(name, "").strip() for name in self.authorization_envs):
            raise RuntimeError("runtime_configuration_invalid", "LANGGRAPH_RUNTIME_TOKEN is required for the external LangGraph runtime")

    async def _client_for_request(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _replay_params(self, *, last_sequence: int, last_event_id: str | None) -> dict[str, Any]:
        """Encode the runtime transport's durable replay cursor."""
        return {"after_event_id": last_event_id} if self.replay_by_event_id and last_event_id else {"after_sequence": last_sequence}

    def _headers(self, request: AgentRuntimeRequest | None = None) -> dict[str, str]:
        headers = {"accept": "application/json"}
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
        for env_name in self.authorization_envs:
            token = os.getenv(env_name)
            if token:
                headers["authorization"] = f"Bearer {token}"
                break
        if self.framework == "langgraph" and "authorization" not in headers:
            raise RuntimeError("runtime_configuration_invalid", "LANGGRAPH_RUNTIME_TOKEN is required for the external LangGraph runtime")
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
            # A plain JSON ``detail`` is an explicit runtime rejection, not a
            # connectivity failure. Preserve it so callers can fix the
            # definition/configuration instead of retrying an available runtime.
            detail = payload.get("detail") if isinstance(payload, Mapping) else None
            if response.status_code < 500 and isinstance(detail, str) and detail.strip():
                raise RuntimeError(
                    code="runtime_request_rejected",
                    safe_message=detail[:2000],
                    retryable=False,
                    details={"status_code": response.status_code},
                ) from exc
            raise RuntimeError.from_exception(exc, code="runtime_transport_error", retryable=True, safe_message="Agent runtime is unavailable") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned an invalid response")
        if payload.get("error"):
            error = payload["error"]
            raise RuntimeError(
                code=str(error.get("code") or "runtime_failed"),
                safe_message=str(error.get("safe_message") or "Agent runtime failed"),
                retryable=bool(error.get("retryable")),
                details=dict(error.get("details") or {}),
                runtime_metadata=dict(payload.get("runtime_metadata") or {}),
            )
        if "result" not in payload or not isinstance(payload["result"], Mapping):
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned an invalid response envelope")
        return payload["result"]

    async def _stream(self, path: str, request: AgentRuntimeRequest, *, context: RuntimeInvocationContext, payload: Mapping[str, Any] | None, event_sink: AgentRuntimeEventSink | None) -> AgentRuntimeResult:
        resolved_spec = context.resolved_spec if isinstance(context.resolved_spec, Mapping) else {}
        runtime = resolved_spec.get("runtime") if isinstance(resolved_spec.get("runtime"), Mapping) else {}
        features = runtime.get("features") if isinstance(runtime.get("features"), Mapping) else {}
        definition = AgentDefinition(
            definition_id=request.definition_id,
            framework=request.framework,
            builder_id=request.builder_id,
            capabilities=dict(features),
            definition_metadata=definition_metadata_from_spec(resolved_spec),
        )
        operation_id = str(request.options.get("idempotency_key") or "").strip()
        if not operation_id:
            operation_seed = {
                "path": path,
                "run_id": request.run_id,
                "input": request.input,
                "interrupt": dict((payload or {}).get("interrupt") or {}),
                "continuation": request.continuation.to_dict() if request.continuation else None,
            }
            operation_id = "runtime-operation:" + hashlib.sha256(
                json.dumps(operation_seed, sort_keys=True, separators=(",", ":"), default=str).encode()
            ).hexdigest()
        body = {
            "definition": definition.to_dict(),
            "request": request.to_dict(),
            "context": context_to_dict(context),
            "operation_id": operation_id,
            **dict(payload or {}),
        }
        seen: dict[str, str] = {}
        terminal: AgentRuntimeResult | None = None
        terminal_hash: str | None = None
        terminal_event_id: str | None = None
        terminal_event_seen = False
        last_sequence = 0
        last_event_id: str | None = None
        pending_deltas: list[AgentRuntimeEvent] = []
        pending_delta_bytes = 0
        pending_delta_started: float | None = None
        delta_flush_task: asyncio.Task[None] | None = None
        delta_lock = asyncio.Lock()
        last_persisted_binding_hash: str | None = None

        async def flush_output_deltas() -> None:
            nonlocal pending_delta_bytes, pending_delta_started, delta_flush_task
            timer = delta_flush_task
            current = asyncio.current_task()
            if timer is not current:
                delta_flush_task = None
            if timer is not None and timer is not current and not timer.done():
                timer.cancel()
                await asyncio.gather(timer, return_exceptions=True)
            async with delta_lock:
                if not pending_deltas:
                    return
                first, last = pending_deltas[0], pending_deltas[-1]
                text = "".join(
                    str(event.payload.get("delta") or event.payload.get("content") or event.payload.get("text") or "")
                    for event in pending_deltas
                )
                source_ids = [event.event_id for event in pending_deltas]
                source_metadata = dict(first.source_metadata or {})
                source_metadata.update({
                    "first_source_sequence": first.sequence,
                    "last_source_sequence": last.sequence,
                    "first_source_event_id": first.event_id,
                    "last_source_event_id": last.event_id,
                    "source_event_ids": source_ids,
                    "chunk_count": len(pending_deltas),
                })
                if self.visualization_id:
                    source_metadata.setdefault("visualization_id", self.visualization_id)
                coalesced = AgentRuntimeEvent(
                    event_id=f"coalesced:{first.event_id}:{last.event_id}",
                    run_id=last.run_id,
                    sequence=last.sequence,
                    kind="output.delta",
                    attempt=last.attempt,
                    payload={"delta": text, "chunk_count": len(pending_deltas)},
                    occurred_at=last.occurred_at,
                    trace_id=last.trace_id or first.trace_id,
                    source_metadata=source_metadata,
                    continuation=last.continuation,
                )
                pending_deltas.clear()
                pending_delta_bytes = 0
                pending_delta_started = None
                if event_sink is not None:
                    await event_sink.emit_runtime_event(coalesced)
            if timer is current:
                delta_flush_task = None

        async def emit_product_event(event: AgentRuntimeEvent) -> None:
            nonlocal pending_delta_bytes, pending_delta_started, delta_flush_task
            if event.kind != "output.delta":
                await flush_output_deltas()
                if event_sink is not None and not event.terminal:
                    await event_sink.emit_runtime_event(event)
                return
            async with delta_lock:
                if pending_delta_started is None:
                    pending_delta_started = time.monotonic()

                    async def flush_after_interval() -> None:
                        await asyncio.sleep(self._output_delta_flush_seconds)
                        await flush_output_deltas()

                    delta_flush_task = asyncio.create_task(
                        flush_after_interval(), name=f"runtime-delta-flush-{request.run_id}"
                    )
                pending_deltas.append(event)
                pending_delta_bytes += len(
                    str(event.payload.get("delta") or event.payload.get("content") or event.payload.get("text") or "").encode("utf-8")
                )
                should_flush = (
                    pending_delta_bytes >= self._output_delta_flush_bytes
                    or time.monotonic() - pending_delta_started >= self._output_delta_flush_seconds
                )
            if should_flush:
                await flush_output_deltas()

        async def consume(method: str, stream_path: str, *, replay: bool = False) -> None:
            nonlocal terminal, terminal_hash, terminal_event_id, terminal_event_seen, last_sequence
            nonlocal last_event_id, last_persisted_binding_hash
            client = await self._client_for_request()
            headers = {**self._headers(request), "accept": "text/event-stream"}
            params: dict[str, Any] | None = None
            if replay:
                await flush_output_deltas()
                params = self._replay_params(
                    last_sequence=last_sequence,
                    last_event_id=last_event_id,
                )
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
                    if (
                        response.status_code < 500
                        and isinstance(envelope, Mapping)
                        and isinstance(envelope.get("detail"), str)
                        and envelope["detail"].strip()
                    ):
                        raise RuntimeError(
                            code="runtime_request_rejected",
                            safe_message=envelope["detail"][:2000],
                            retryable=False,
                            details={"status_code": response.status_code},
                        )
                response.raise_for_status()
                async for _name, item in iter_sse(response):
                    envelope = item["data"]
                    if not isinstance(envelope, Mapping):
                        raise RuntimeError("runtime_protocol_error", "Agent runtime returned a malformed event envelope")
                    event_payload = envelope.get("event")
                    if not isinstance(event_payload, Mapping):
                        raise RuntimeError("runtime_protocol_error", "Agent runtime returned a malformed event")
                    if _name != str(event_payload.get("kind") or ""):
                        raise RuntimeError("runtime_protocol_error", "Agent runtime event name does not match its kind")
                    event_data = dict(event_payload)
                    source_metadata = dict(event_data.get("source_metadata") or {})
                    source_metadata.setdefault("framework", self.framework)
                    source_metadata.setdefault("source_event", str(event_data.get("kind") or _name))
                    if self.visualization_id:
                        source_metadata.setdefault("visualization_id", self.visualization_id)
                    event_data["source_metadata"] = source_metadata
                    event = event_from_dict(event_data)
                    if event.run_id != request.run_id:
                        raise RuntimeError("runtime_protocol_error", "Agent runtime returned a mismatched run ID")
                    event_hash = hashlib.sha256(json.dumps(event.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
                    if event.event_id in seen:
                        if seen[event.event_id] != event_hash:
                            raise RuntimeError("runtime_protocol_error", "Agent runtime returned conflicting duplicate event IDs")
                        continue
                    if terminal_event_seen:
                        raise RuntimeError("runtime_protocol_error", "Agent runtime returned an event after its terminal event")
                    seen[event.event_id] = event_hash
                    if event.sequence <= last_sequence:
                        raise RuntimeError("runtime_protocol_error", "Agent runtime event sequence is not monotonic")
                    if event.terminal:
                        if terminal_event_seen:
                            raise RuntimeError("runtime_protocol_error", "Agent runtime returned more than one terminal event")
                        terminal_event_seen = True
                        if envelope.get("result") is None:
                            raise RuntimeError("runtime_protocol_error", "Agent runtime terminal event did not include a result")
                    last_sequence = event.sequence
                    last_event_id = event.event_id
                    if event_sink is not None:
                        await emit_product_event(event)
                        if event.continuation is not None:
                            binding_hash = hashlib.sha256(
                                json.dumps(event.continuation.to_dict(), sort_keys=True, default=str).encode()
                            ).hexdigest()
                            if binding_hash != last_persisted_binding_hash:
                                persist_binding = getattr(event_sink, "persist_runtime_binding", None)
                                if persist_binding is not None:
                                    await persist_binding(request.run_id, event.continuation)
                                last_persisted_binding_hash = binding_hash
                    if envelope.get("result") is not None:
                        resumable_boundary = (
                            isinstance(envelope.get("result"), Mapping)
                            and str(envelope["result"].get("status") or "") in {"awaiting_human", "paused"}
                            and event.kind in {"approval.requested", "interrupt.requested", "run.paused"}
                        )
                        if not event.terminal and not resumable_boundary:
                            raise RuntimeError("runtime_protocol_error", "Agent runtime attached a result to an invalid nonterminal event")
                        if terminal_event_id is not None and event.event_id != terminal_event_id:
                            raise RuntimeError("runtime_protocol_error", "Agent runtime returned more than one terminal result")
                        candidate = result_from_dict(envelope["result"])
                        candidate_hash = hashlib.sha256(json.dumps(candidate.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
                        if terminal_hash is not None and candidate_hash != terminal_hash:
                            raise RuntimeError("runtime_protocol_error", "Agent runtime returned conflicting terminal results")
                        terminal = candidate
                        terminal_hash = candidate_hash
                        terminal_event_id = event.event_id

        reconnect_started: float | None = None
        reconnect_count = 0
        try:
            try:
                await asyncio.wait_for(consume("POST", path), timeout=self._execution_timeout)
            except httpx.HTTPStatusError:
                # Deterministic HTTP admission failures are not ambiguous
                # stream disconnects and must not trigger replay or a second
                # start request.
                raise
            except httpx.HTTPError:
                # The runtime may have committed the execution even though
                # the subscriber connection failed. Fall through to durable
                # replay before surfacing a transport error.
                pass
            while terminal is None:
                if reconnect_started is None:
                    reconnect_started = asyncio.get_running_loop().time()
                if reconnect_count > 0 and (
                    reconnect_count >= self._reconnect_attempts
                    or asyncio.get_running_loop().time() - reconnect_started >= self._reconnect_deadline
                ):
                    raise RuntimeError("runtime_stream_error", "Agent runtime stream ended before a terminal result", retryable=True)
                await asyncio.sleep(min(self._reconnect_backoff * (2 ** reconnect_count), self._reconnect_deadline))
                reconnect_count += 1
                replay_start_sequence = last_sequence
                try:
                    # HTTP connect/read timeouts detect a stalled replay. Do not
                    # impose the reconnect deadline on a healthy event stream:
                    # a large durable backlog may legitimately take longer to
                    # validate and project than reconnect establishment.
                    await consume("GET", f"/v1/runs/{request.run_id}/events", replay=True)
                    if last_sequence > replay_start_sequence:
                        reconnect_started = None
                        reconnect_count = 0
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
        except httpx.HTTPStatusError as exc:
            raise RuntimeError.from_exception(
                exc,
                code="runtime_transport_error",
                retryable=False,
                safe_message="Agent runtime rejected the request",
                details={"status_code": exc.response.status_code},
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError.from_exception(exc, code="runtime_stream_timeout", retryable=True, safe_message="Agent runtime stream timed out") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError.from_exception(exc, code="runtime_stream_error", retryable=True, safe_message="Agent runtime stream failed") from exc
        finally:
            if delta_flush_task is not None and not delta_flush_task.done():
                delta_flush_task.cancel()
                await asyncio.gather(delta_flush_task, return_exceptions=True)
            cleanup_task = asyncio.create_task(
                flush_output_deltas(), name=f"runtime-delta-final-flush-{request.run_id}"
            )
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                # Complete the bounded persistence cleanup before propagating
                # caller cancellation; never orphan a writer using a DB session.
                await cleanup_task
                raise
        if terminal is None:
            raise RuntimeError("runtime_protocol_error", "Agent runtime stream ended without a terminal result")
        return terminal

class HttpLangGraphRuntimeAdapter(AgentRuntimeAdapter):
    """LangGraph adapter composed with the neutral HTTP transport."""

    # The external runtime exposes a durable pause request protocol backed by
    # its execution store and LangGraph checkpointer.
    supports_task_pause = True
    supports_external_task_pause = True
    framework = "langgraph"
    builder_id = "langgraph_graph"
    implemented_operations = frozenset({
        RuntimeOperationId.RUN_START,
        RuntimeOperationId.RUN_CANCEL,
        RuntimeOperationId.RUN_RESUME,
        RuntimeOperationId.RUN_INSPECT_STATE,
        RuntimeOperationId.RUN_CONTINUATION_CLEANUP,
        RuntimeOperationId.TASK_COURSE_CORRECTION_SUBMIT,
        RuntimeOperationId.TRACE_PROJECT,
    })

    def __init__(self, base_url: str | None = None, **kwargs: Any) -> None:
        self.transport = RuntimeTransportConnector(
            base_url=base_url,
            framework=self.framework,
            authorization_env="LANGGRAPH_RUNTIME_TOKEN",
            visualization_id="langgraph.session",
            **kwargs,
        )

    async def prepare_request(
        self,
        request: AgentRuntimeRequest,
        *,
        context: RuntimeInvocationContext,
    ) -> AgentRuntimeRequest:
        from app.mcp.execution_context_token import issue_execution_context_token
        from app.mcp.registry import MCP_TOOL_DEFINITIONS
        from app.tools.context import ToolInvocationContext

        spec = dict(context.resolved_spec or {})
        config = dict(spec.get("config") or {})
        configured_tool_ids = {
            str(value) for value in config.get("allowed_tool_ids") or [] if value
        }
        # Workflow specs expose stable, framework-neutral contract IDs, while
        # the MCP authorization boundary validates canonical MCP tool names.
        # Expand the grant here, where the control plane owns both registries,
        # instead of teaching the external runtime about product policy.
        allowed_tools = sorted(
            name
            for name, definition in MCP_TOOL_DEFINITIONS.items()
            if name in configured_tool_ids
            or definition.registry_contract_id in configured_tool_ids
        )
        task_context = context.task_context
        task_id = str(request.task_id or request.run_id)
        limits = dict(task_context.limits or {}) if task_context is not None else {}
        ttl_seconds = max(3600, int(limits.get("max_active_runtime_ms", 3_600_000)) // 1000)
        token = issue_execution_context_token(
            ToolInvocationContext(
                thread_id=request.thread_id,
                run_id=request.run_id,
                embedding_model=context.embedding_model,
                context_window=int(config.get("context_window") or 32_768),
                use_web_search=bool(config.get("use_web_search")),
                use_reranker=True,
                extensions={"task_id": task_id, "llm_model": config.get("llm_model")},
            ),
            task_id=task_id,
            allowed_tools=allowed_tools,
            ttl_seconds=ttl_seconds,
        )
        return replace(
            request,
            input={
                **dict(request.input),
                "mcp_execution_context_token": token,
                # Admission must check the canonical MCP grant, not every
                # framework-neutral tool contract in the workflow. Some
                # contracts (for example ``clarify_intent``) are implemented
                # inside the graph and are intentionally absent from MCP.
                "mcp_allowed_tool_ids": allowed_tools,
            },
        )

    async def aclose(self) -> None:
        await self.transport.aclose()

    async def capabilities(self, definition: AgentDefinition) -> RuntimeCapabilities:
        value = await self.transport._json("POST", "/v1/capabilities", json={"definition": definition.to_dict()})
        return capabilities_from_dict(value["capabilities"])

    async def deployment_capabilities(self) -> RuntimeCapabilities:
        value = await self.transport._json("GET", "/v1/capabilities")
        return capabilities_from_dict(value["capabilities"])

    async def validate(self, definition: AgentDefinition, spec: Mapping[str, Any], *, options: Mapping[str, Any] | None = None) -> RuntimeValidationResult:
        value = await self.transport._json("POST", "/v1/validate", json={"definition": definition.to_dict(), "spec": _safe_json(spec), "options": _safe_json(options or {})})
        return validation_from_dict(value["validation"])

    async def resolve_definition(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        thread_settings: Mapping[str, Any],
        request_overrides: Mapping[str, Any],
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        value = await self.transport._json(
            "POST",
            "/v1/resolve",
            json={
                "definition": definition.to_dict(),
                "spec": _safe_json(spec),
                "thread_settings": _safe_json(thread_settings),
                "request_overrides": _safe_json(request_overrides),
                "options": _safe_json(options or {}),
            },
        )
        resolved = value.get("resolved_spec")
        if not isinstance(resolved, Mapping):
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned an invalid resolved definition")
        return dict(resolved)

    async def builder_catalog(self, definition: AgentDefinition) -> Mapping[str, Any]:
        value = await self.transport._json(
            "POST", "/v1/catalog", json={"definition": definition.to_dict()}
        )
        catalog = value.get("catalog")
        if not isinstance(catalog, Mapping):
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned an invalid builder catalog")
        return dict(catalog)

    async def start(self, request: AgentRuntimeRequest, *, context: RuntimeInvocationContext, event_sink: AgentRuntimeEventSink | None = None) -> AgentRuntimeResult:
        return await self.transport._stream("/v1/runs/start", request, context=context, payload=None, event_sink=event_sink)

    async def resume(self, request: AgentRuntimeRequest, *, interrupt: Mapping[str, Any], context: RuntimeInvocationContext, event_sink: AgentRuntimeEventSink | None = None) -> AgentRuntimeResult:
        return await self.transport._stream(f"/v1/runs/{request.run_id}/resume", request, context=context, payload={"interrupt": _safe_json(interrupt)}, event_sink=event_sink)

    async def continue_run(self, request: AgentRuntimeRequest, *, context: RuntimeInvocationContext, event_sink: AgentRuntimeEventSink | None = None) -> AgentRuntimeResult | None:
        result = await self.transport._stream(
            f"/v1/runs/{request.run_id}/continue",
            request,
            context=context,
            payload=None,
            event_sink=event_sink,
        )
        return None if result.status == "no_continuation" else result

    async def cancel(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        value = await self.transport._json("POST", f"/v1/runs/{request.run_id}/cancel", request=request, json={"request": request.to_dict()})
        return dict(value) if isinstance(value, Mapping) else {"result": value}

    async def pause(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        value = await self.transport._json("POST", f"/v1/runs/{request.run_id}/pause", request=request, json={"request": request.to_dict()})
        return dict(value) if isinstance(value, Mapping) else {"result": value}

    async def submit_course_correction(
        self,
        request: AgentRuntimeRequest,
        correction: RuntimeCourseCorrection,
    ) -> RuntimeCourseCorrectionReceipt:
        value = await self.transport._json(
            "POST",
            f"/v1/runs/{request.run_id}/course-corrections",
            request=request,
            json={"request": request.to_dict(), "correction": correction.to_dict()},
        )
        if not isinstance(value, Mapping):
            raise RuntimeError("runtime_protocol_error", "Runtime returned an invalid correction receipt")
        return course_correction_receipt_from_dict(value)

    async def inspect_state(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        value = await self.transport._json("POST", f"/v1/runs/{request.run_id}/inspect", request=request, json={"request": request.to_dict()})
        return dict(value or {}) if isinstance(value, Mapping) else {}

    async def project_trace(self, events: list[Mapping[str, Any]], *, run_id: str, context: RuntimeInvocationContext | None = None) -> list[AgentRuntimeEvent]:
        projected = []
        for event in events:
            value = dict(event)
            source_metadata = dict(value.get("source_metadata") or {})
            source_metadata.setdefault("framework", self.framework)
            source_metadata.setdefault("source_event", str(value.get("kind") or "runtime.event"))
            source_metadata.setdefault("visualization_id", "langgraph.session")
            value["source_metadata"] = source_metadata
            projected.append(event_from_dict(value))
        return projected

    async def delete_continuation(self, continuation: Any) -> Any:
        binding_id = str(continuation.payload.get("binding_id") or "")
        if not binding_id:
            raise RuntimeError("runtime_binding_invalid", "Runtime continuation is missing its opaque binding ID")
        return await self.transport._json("DELETE", f"/v1/continuations/{binding_id}", json={"continuation": continuation.to_dict()})
