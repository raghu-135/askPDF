from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
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
from runtime_service.execution_store import ExecutionStore


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
        self.queue: asyncio.Queue[AgentRuntimeEvent | None] = asyncio.Queue(maxsize=max(100, int(os.getenv("AGENT_RUNTIME_EVENT_BUFFER_SIZE", "1000"))))
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
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(event)


def create_app(*, execution_store: ExecutionStore | None = None) -> FastAPI:
    runtime_state: dict[str, Any] = {
        "draining": False,
        "active": {},
        "readiness": {},
    }
    execution_store = execution_store or ExecutionStore()
    # Serialize the durable-record check and task registration.  Without this
    # small critical section, two subscribers can both observe no active task
    # and start the same execution.
    execution_start_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime_state["draining"] = False
        await execution_store.initialize()
        # Re-claim durable nonterminal executions after a runtime restart.
        # The worker is defined below, but FastAPI enters lifespan only after
        # create_app has finished constructing the complete application.
        for record in await execution_store.nonterminal():
            try:
                recovered_request = request_from_dict(record.request)
                runtime_state["active"][record.run_id] = asyncio.create_task(
                    _execute_operation(record.payload, record.operation, recovered_request, attempt=record.attempt),
                    name=f"agent-runtime-recovery-{record.run_id}",
                )
            except Exception:
                logger.exception("Unable to recover runtime execution | run_id=%s", record.run_id)
        yield
        runtime_state["draining"] = True
        grace = max(1.0, float(os.getenv("AGENT_RUNTIME_SHUTDOWN_GRACE_SECONDS", "30")))
        deadline = time.monotonic() + grace
        while runtime_state["active"] and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        for task in list(runtime_state["active"].values()):
            task.cancel()
        await execution_store.close()

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

    async def _execute_operation(payload: Mapping[str, Any], operation: str, request: Any, *, attempt: int) -> None:
        """Run independently of any HTTP subscriber and journal every event."""
        run_id = request.run_id
        cancellation_event = asyncio.Event()

        async def cancellation_probe() -> bool:
            return cancellation_event.is_set() or await execution_store.is_cancel_requested(run_id)

        class DurableSink:
            async def emit(self, *args: Any) -> None:
                if len(args) == 1 and isinstance(args[0], AgentRuntimeEvent):
                    event = args[0]
                else:
                    kind = str(args[0]) if args else "runtime.event"
                    event = AgentRuntimeEvent(
                        event_id=f"{run_id}:{uuid.uuid4().hex}",
                        run_id=run_id,
                        sequence=0,
                        kind=kind,
                        payload=dict(args[1] or {}) if len(args) > 1 and isinstance(args[1], Mapping) else {},
                        terminal=kind in {"run.completed", "run.failed", "run.cancelled", "run.terminal"},
                    )
                await execution_store.append(run_id, event.to_dict(), attempt=attempt)

        await execution_store.set_status(run_id, "running")
        context = _context(payload, request, cancellation_checker=cancellation_probe)
        try:
            execution_timeout = max(1.0, float(os.getenv("AGENT_RUNTIME_EXECUTION_TIMEOUT_SECONDS", "3600")))
            result = await asyncio.wait_for(
                getattr(get_adapter(), operation)(
                    request,
                    **({"interrupt": payload.get("interrupt") or {}} if operation == "resume" else {}),
                    context=context,
                    event_sink=DurableSink(),
                ),
                timeout=execution_timeout,
            )
            if result is None:
                result = AgentRuntimeResult(status="no_continuation", runtime_metadata={"continuation_available": False})
            result = result if isinstance(result, AgentRuntimeResult) else result_from_dict(result)
            terminal_kind = "run.continuation_empty" if result.status == "no_continuation" else "run.cancelled" if result.status == "cancelled" else "run.failed" if result.status == "failed" else "run.completed"
            terminal = AgentRuntimeEvent(
                event_id=f"{run_id}:terminal",
                run_id=run_id,
                sequence=0,
                kind=terminal_kind,
                terminal=True,
                payload={"status": result.status},
                continuation=result.continuation,
            )
            await execution_store.append(run_id, terminal.to_dict(), result=result.to_dict(), attempt=attempt)
            await execution_store.set_status(run_id, result.status, result=result.to_dict())
        except asyncio.TimeoutError:
            error = RuntimeError("runtime_execution_timeout", "Agent runtime execution timed out", retryable=True)
            result = AgentRuntimeResult(status="failed", error=error.to_dict())
            terminal = AgentRuntimeEvent(event_id=f"{run_id}:terminal", run_id=run_id, sequence=0, kind="run.failed", terminal=True, payload={"error": error.to_dict()})
            await execution_store.append(run_id, terminal.to_dict(), result=result.to_dict(), attempt=attempt)
            await execution_store.set_status(run_id, "failed", result=result.to_dict(), error=error.to_dict())
        except RuntimeError as exc:
            logger.exception("LangGraph runtime failed | run_id=%s", run_id)
            result = AgentRuntimeResult(status="failed", error=exc.to_dict())
            terminal = AgentRuntimeEvent(event_id=f"{run_id}:terminal", run_id=run_id, sequence=0, kind="run.failed", terminal=True, payload={"error": exc.to_dict()})
            await execution_store.append(run_id, terminal.to_dict(), result=result.to_dict(), attempt=attempt)
            await execution_store.set_status(run_id, "failed", result=result.to_dict(), error=exc.to_dict())
        except Exception as exc:
            logger.exception("LangGraph runtime execution failed | run_id=%s", run_id)
            error = RuntimeError.from_exception(exc, code="runtime_execution_failed", retryable=False, safe_message="Agent runtime execution failed")
            result = AgentRuntimeResult(status="failed", error=error.to_dict())
            terminal = AgentRuntimeEvent(event_id=f"{run_id}:terminal", run_id=run_id, sequence=0, kind="run.failed", terminal=True, payload={"error": error.to_dict()})
            await execution_store.append(run_id, terminal.to_dict(), result=result.to_dict(), attempt=attempt)
            await execution_store.set_status(run_id, "failed", result=result.to_dict(), error=error.to_dict())

    async def stream_operation(
        payload: Mapping[str, Any],
        operation: str,
        expected_run_id: str | None = None,
        after_sequence: int = 0,
        *,
        allow_start: bool = True,
    ) -> AsyncIterator[str]:
        request = request_from_dict(payload["request"])
        if expected_run_id and request.run_id != expected_run_id:
            raise HTTPException(status_code=400, detail="run_id does not match request path")
        if runtime_state["draining"]:
            raise HTTPException(status_code=503, detail="runtime is draining")
        async with execution_start_lock:
            # Event subscriptions are observers.  They must never create or
            # replace a durable execution record, especially after terminal
            # completion or after a process restart.
            if allow_start:
                record = await execution_store.create(request.run_id, operation, request.to_dict(), payload)
            else:
                record = await execution_store.get(request.run_id)
                if record is None:
                    raise HTTPException(status_code=404, detail="runtime run not found")

            task = runtime_state["active"].get(request.run_id)
            terminal_statuses = {"completed", "failed", "cancelled", "no_continuation"}
            explicitly_retryable = allow_start and operation in {"resume", "continue_run"} and record.status not in terminal_statuses
            should_start = allow_start and (
                record.status == "queued" or explicitly_retryable
            )
            if should_start and (task is None or task.done()):
                task = asyncio.create_task(
                    _execute_operation(payload, operation, request, attempt=record.attempt),
                    name=f"agent-runtime-{request.run_id}",
                )
                runtime_state["active"][request.run_id] = task

                def _remove_finished(done_task: asyncio.Task[Any]) -> None:
                    if runtime_state["active"].get(request.run_id) is done_task:
                        runtime_state["active"].pop(request.run_id, None)

                task.add_done_callback(_remove_finished)
            attempt = record.attempt
        last_sequence = after_sequence
        while True:
            events = await execution_store.events_after(request.run_id, last_sequence, attempt=attempt)
            for item in events:
                event = event_from_dict(item)
                result = result_from_dict(item["result"]) if item.get("result") else None
                yield sse_encode(event, result=result)
                last_sequence = max(last_sequence, event.sequence)
                if event.terminal:
                    return
            record = await execution_store.get(request.run_id)
            if record and record.status in {"completed", "failed", "cancelled", "no_continuation"}:
                return
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.1)

    @app.post("/v1/runs/start")
    async def start(payload: Mapping[str, Any]) -> StreamingResponse:
        return StreamingResponse(stream_operation(payload, "start"), media_type="text/event-stream")

    @app.get("/v1/runs/{run_id}/events")
    async def events(run_id: str, after_sequence: int = 0) -> StreamingResponse:
        record = await execution_store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="runtime run not found")
        payload = {"request": record.request, "context": record.payload.get("context") or {}}
        return StreamingResponse(
            stream_operation(
                payload,
                record.operation,
                run_id,
                after_sequence,
                allow_start=False,
            ),
            media_type="text/event-stream",
        )

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
        await execution_store.request_cancel(request.run_id)
        cancellation_events.setdefault(request.run_id, asyncio.Event()).set()
        return json_envelope(status="ok", request_id=request.headers.get("x-request-id"), result={"run_id": request.run_id, "status": "cancellation_requested"})

    @app.post("/v1/runs/{run_id}/inspect")
    async def inspect(run_id: str, payload: Mapping[str, Any], request_context: Request) -> dict[str, Any]:
        request = request_from_dict(payload["request"])
        if request.run_id != run_id:
            raise HTTPException(status_code=400, detail="run_id does not match request path")
        durable = await execution_store.get(run_id)
        runtime_inspection = dict(await get_adapter().inspect(request))
        if durable is not None:
            runtime_inspection.update({
                "run_id": run_id,
                "status": durable.status,
                "cancel_requested": durable.cancel_requested,
                "last_sequence": durable.next_sequence - 1,
                "durable": execution_store.durable,
            })
        return json_envelope(status="ok", request_id=request_context.headers.get("x-request-id"), result=runtime_inspection)

    @app.delete("/v1/continuations/{binding_id}")
    async def delete_continuation(binding_id: str, payload: Mapping[str, Any], request: Request) -> dict[str, Any]:
        from app.runtime.transport import _binding

        continuation = _binding(payload.get("continuation"))
        result = await get_adapter().delete_continuation(continuation)
        return json_envelope(status="ok", request_id=request.headers.get("x-request-id"), result=result if isinstance(result, Mapping) else {"value": result})

    return app
