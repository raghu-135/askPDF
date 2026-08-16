from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncIterator, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import AgentRuntimeEvent, AgentRuntimeResult
from app.runtime.errors import RuntimeError
from app.runtime.transport import (
    WIRE_VERSION,
    definition_from_dict,
    event_from_dict,
    request_from_dict,
    result_from_dict,
    sse_encode,
    json_envelope,
)


logger = logging.getLogger(__name__)


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{str(key): _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _context(payload: Mapping[str, Any], request: Any, *, cancellation_checker: Any = None) -> RuntimeExecutionContext:
    value = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
    run_context = dict(value.get("agent_run_context") or {})
    if isinstance(run_context.get("run"), Mapping):
        run_context["run"] = _namespace(run_context["run"])
    resolved_spec = dict(value.get("resolved_spec") or {})
    if resolved_spec and run_context.get("run") is not None:
        # Continuation code consumes the legacy run-shaped object. Keep its
        # graph snapshot synchronized with the explicit neutral context field
        # rather than relying on ORM/model serialization details.
        run_context["run"].resolved_spec_json = resolved_spec
    request_payload = value.get("request_payload")
    request_values = {
        **(dict(request_payload) if isinstance(request_payload, Mapping) else {}),
        **dict(request.input or {}),
        **dict(request.options or {}),
    }
    # Only the top-level request needs attribute access for the legacy node
    # functions. Preserve nested options as ordinary dict/list values because
    # they may be copied into LangGraph state and must remain checkpointable.
    return RuntimeExecutionContext(
        request=SimpleNamespace(**request_values),
        embedding_model=value.get("embedding_model") or request.options.get("embedding_model"),
        resolved_spec=resolved_spec,
        agent_run_context=run_context,
        task_id=value.get("task_id") or request.task_id,
        task_worker_id=value.get("task_worker_id"),
        cancellation_checker=cancellation_checker,
    )


class _QueueSink:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.queue: asyncio.Queue[AgentRuntimeEvent | None] = asyncio.Queue()
        self.sequence = 0

    async def emit(self, *args: Any) -> None:
        if len(args) == 1 and isinstance(args[0], AgentRuntimeEvent):
            event = args[0]
        else:
            kind = str(args[0]) if args else "runtime.event"
            payload = dict(args[1] or {}) if len(args) > 1 and isinstance(args[1], Mapping) else {}
            self.sequence += 1
            event = AgentRuntimeEvent(
                event_id=str(payload.get("event_id") or f"{self.run_id}:{self.sequence}"),
                run_id=self.run_id,
                sequence=self.sequence,
                kind=kind,
                payload=payload,
                terminal=kind in {"run.completed", "run.failed", "run.cancelled", "run.terminal"},
            )
        self.sequence = max(self.sequence, event.sequence)
        await self.queue.put(event)


def create_app() -> FastAPI:
    runtime_state: dict[str, Any] = {
        "draining": False,
        "active": {},
        "readiness": {},
    }

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime_state["draining"] = False
        yield
        runtime_state["draining"] = True
        grace = max(1.0, float(os.getenv("AGENT_RUNTIME_SHUTDOWN_GRACE_SECONDS", "30")))
        deadline = time.monotonic() + grace
        while runtime_state["active"] and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        for task in list(runtime_state["active"].values()):
            task.cancel()

    app = FastAPI(
        title="AskPDF LangGraph Runtime",
        version=os.getenv("RUNTIME_PROVIDER_VERSION", "1"),
        lifespan=lifespan,
    )
    adapter: Any = None
    cancellation_events: dict[str, asyncio.Event] = {}

    def get_adapter() -> Any:
        """Load legacy execution code only when an execution operation is called."""
        nonlocal adapter
        if adapter is None:
            from app.runtime.langgraph_adapter import LangGraphRuntimeAdapter

            adapter = LangGraphRuntimeAdapter()
        return adapter

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "langgraph-runtime"}

    async def _probe(url: str, timeout: float) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
            return {"status": "ok" if response.status_code < 500 else "failed", "http_status": response.status_code}
        except Exception as exc:
            return {"status": "failed", "error": type(exc).__name__}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        if runtime_state["draining"]:
            return JSONResponse({"status": "draining"}, status_code=503)
        checks: dict[str, Any] = {}
        if os.getenv("ASKPDF_AGENT_CHECKPOINTER", "memory").strip().lower() != "postgres":
            checks["checkpoint_store"] = {"status": "ok", "backend": "memory"}
        else:
            try:
                from app.agent_workflows.checkpointing import open_agent_checkpointer

                async with open_agent_checkpointer():
                    pass
                checks["checkpoint_store"] = {"status": "ok", "backend": "postgres"}
            except Exception as exc:
                checks["checkpoint_store"] = {"status": "failed", "error": type(exc).__name__}
        mcp_url = os.getenv("MCP_LOOPBACK_URL", "").strip()
        if mcp_url:
            checks["mcp"] = await _probe(mcp_url, float(os.getenv("AGENT_RUNTIME_MCP_READY_TIMEOUT_SECONDS", "5")))
        else:
            checks["mcp"] = {"status": "not_configured"}
        provider_url = os.getenv("LLM_API_URL", "").strip()
        if provider_url:
            provider_base = provider_url.rstrip("/")
            models_url = provider_base + "/models" if provider_base.endswith("/v1") else provider_base + "/v1/models"
            checks["provider"] = await _probe(models_url, float(os.getenv("AGENT_RUNTIME_PROVIDER_READY_TIMEOUT_SECONDS", "10")))
        else:
            checks["provider"] = {"status": "not_configured"}
        healthy = all(value.get("status") == "ok" for value in checks.values())
        runtime_state["readiness"] = checks
        return JSONResponse({"status": "ok" if healthy else "not_ready", "checks": checks}, status_code=200 if healthy else 503)

    @app.get("/v1/capabilities")
    async def capabilities(request: Request) -> dict[str, Any]:
        return json_envelope(
            status="ok",
            request_id=request.headers.get("x-request-id"),
            result={"capabilities": {
                "streaming": True,
                "resume": True,
                "cancellation": True,
                "inspection": True,
                "continuation_cleanup": True,
                "task_execution": True,
                "native_checkpoints": True,
                "runtime_version": os.getenv("RUNTIME_PROVIDER_VERSION", "1"),
                "contract_version": WIRE_VERSION,
            }},
            runtime_metadata={"framework": "langgraph", "builder_id": "langgraph_graph"},
        )

    @app.post("/v1/validate")
    async def validate(payload: Mapping[str, Any], request: Request) -> dict[str, Any]:
        try:
            definition = definition_from_dict(payload["definition"])
            value = await get_adapter().validate(definition, payload.get("spec") or {}, options=payload.get("options") or {})
            return json_envelope(status="ok", request_id=request.headers.get("x-request-id"), result={"validation": value.to_dict()})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def stream_operation(payload: Mapping[str, Any], operation: str, expected_run_id: str | None = None) -> AsyncIterator[str]:
        request = request_from_dict(payload["request"])
        if expected_run_id and request.run_id != expected_run_id:
            raise HTTPException(status_code=400, detail="run_id does not match request path")
        if runtime_state["draining"]:
            raise HTTPException(status_code=503, detail="runtime is draining")
        cancellation_event = cancellation_events.setdefault(request.run_id, asyncio.Event())

        async def cancellation_probe() -> bool:
            return cancellation_event.is_set()

        context = _context(payload, request, cancellation_checker=cancellation_probe)
        sink = _QueueSink(request.run_id)
        task = asyncio.create_task(
            getattr(get_adapter(), operation)(
                request,
                **({"interrupt": payload.get("interrupt") or {}} if operation == "resume" else {}),
                context=context,
                event_sink=sink,
            )
        )
        runtime_state["active"][request.run_id] = task
        execution_timeout = max(1.0, float(os.getenv("AGENT_RUNTIME_EXECUTION_TIMEOUT_SECONDS", "3600")))
        deadline = time.monotonic() + execution_timeout
        try:
            while True:
                if task.done() and sink.queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(sink.queue.get(), timeout=min(10.0, max(0.1, deadline - time.monotonic())))
                except asyncio.TimeoutError:
                    if time.monotonic() >= deadline:
                        task.cancel()
                        error = RuntimeError("runtime_execution_timeout", "Agent runtime execution timed out", retryable=True)
                        terminal = AgentRuntimeEvent(event_id=f"{request.run_id}:terminal", run_id=request.run_id, sequence=sink.sequence + 1, kind="run.failed", terminal=True, payload={"error": error.to_dict()})
                        yield sse_encode(terminal, result=AgentRuntimeResult(status="failed", error=error.to_dict()))
                        return
                    # Keep the HTTP stream alive while a model or MCP call is
                    # executing and no runtime event has been produced yet.
                    yield ": keep-alive\n\n"
                    continue
                if event is None:
                    continue
                yield sse_encode(event)
            try:
                result = task.result()
            except RuntimeError as exc:
                logger.exception("LangGraph runtime failed with a mapped runtime error | run_id=%s", request.run_id)
                event = AgentRuntimeEvent(event_id=f"{request.run_id}:terminal", run_id=request.run_id, sequence=sink.sequence + 1, kind="run.failed", terminal=True, payload={"error": exc.to_dict()})
                yield sse_encode(event, result=AgentRuntimeResult(status="failed", error=exc.to_dict()))
                return
            except Exception as exc:
                logger.exception("LangGraph runtime execution failed | run_id=%s", request.run_id)
                error = RuntimeError.from_exception(exc, code="runtime_execution_failed", retryable=False, safe_message="Agent runtime execution failed")
                event = AgentRuntimeEvent(event_id=f"{request.run_id}:terminal", run_id=request.run_id, sequence=sink.sequence + 1, kind="run.failed", terminal=True, payload={"error": error.to_dict()})
                yield sse_encode(event, result=AgentRuntimeResult(status="failed", error=error.to_dict()))
                return
            if result is None:
                # A continuation probe is allowed to find no resumable
                # checkpoint. The control plane uses this result to start the
                # task as a new execution on the same persisted run.
                result = AgentRuntimeResult(
                    status="no_continuation",
                    runtime_metadata={"continuation_available": False},
                )
            result = result if isinstance(result, AgentRuntimeResult) else result_from_dict(result)
            terminal_kind = (
                "run.continuation_empty"
                if result.status == "no_continuation"
                else "run.cancelled"
                if result.status == "cancelled"
                else "run.failed"
                if result.status == "failed"
                else "run.completed"
            )
            terminal = AgentRuntimeEvent(event_id=f"{request.run_id}:terminal", run_id=request.run_id, sequence=sink.sequence + 1, kind=terminal_kind, terminal=True, payload={"status": result.status})
            yield sse_encode(terminal, result=result)
        finally:
            runtime_state["active"].pop(request.run_id, None)
            cancellation_events.pop(request.run_id, None)
            if not task.done():
                task.cancel()

    @app.post("/v1/runs/start")
    async def start(payload: Mapping[str, Any]) -> StreamingResponse:
        return StreamingResponse(stream_operation(payload, "start"), media_type="text/event-stream")

    @app.post("/v1/runs/{run_id}/resume")
    async def resume(run_id: str, payload: Mapping[str, Any]) -> StreamingResponse:
        return StreamingResponse(stream_operation(payload, "resume", run_id), media_type="text/event-stream")

    @app.post("/v1/runs/{run_id}/continue")
    async def continue_run(run_id: str, payload: Mapping[str, Any]) -> StreamingResponse:
        return StreamingResponse(stream_operation(payload, "continue_run", run_id), media_type="text/event-stream")

    @app.post("/v1/runs/{run_id}/cancel")
    async def cancel(run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = request_from_dict(payload["request"])
        if request.run_id != run_id:
            raise HTTPException(status_code=400, detail="run_id does not match request path")
        cancellation_events.setdefault(request.run_id, asyncio.Event()).set()
        return json_envelope(status="ok", request_id=request.headers.get("x-request-id"), result={"run_id": request.run_id, "status": "cancellation_requested"})

    @app.post("/v1/runs/{run_id}/inspect")
    async def inspect(run_id: str, payload: Mapping[str, Any], request_context: Request) -> dict[str, Any]:
        request = request_from_dict(payload["request"])
        if request.run_id != run_id:
            raise HTTPException(status_code=400, detail="run_id does not match request path")
        return json_envelope(status="ok", request_id=request_context.headers.get("x-request-id"), result=dict(await get_adapter().inspect(request)))

    @app.delete("/v1/continuations/{binding_id}")
    async def delete_continuation(binding_id: str, payload: Mapping[str, Any], request: Request) -> dict[str, Any]:
        from app.runtime.transport import _binding

        continuation = _binding(payload.get("continuation"))
        result = await get_adapter().delete_continuation(continuation)
        return json_envelope(status="ok", request_id=request.headers.get("x-request-id"), result=result if isinstance(result, Mapping) else {"value": result})

    return app
