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
from runtime_service.execution_store import ExecutionStore, LeaseLostError, ExecutionConflictError, request_fingerprint
from runtime_service.dependencies import (
    DependencyMonitor,
    langgraph_dependency_requirements,
)


logger = logging.getLogger(__name__)


class DependencyUnavailable(Exception):
    def __init__(self, details: Mapping[str, Any]) -> None:
        self.details = dict(details)
        super().__init__("A dependency required by this agent is unavailable")


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
        "started": False,
        "active": {},
        "readiness": {},
    }
    execution_store = execution_store or ExecutionStore()
    dependency_monitor = DependencyMonitor()
    # Serialize the durable-record check and task registration.  Without this
    # small critical section, two subscribers can both observe no active task
    # and start the same execution.
    execution_start_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime_state["draining"] = False
        await execution_store.initialize()
        from app.agent_workflows.checkpointing import open_agent_checkpointer

        async with open_agent_checkpointer():
            pass
        runtime_state["started"] = True
        dependency_stop = asyncio.Event()
        await dependency_monitor.refresh()
        dependency_task = asyncio.create_task(dependency_monitor.run(dependency_stop), name="agent-runtime-dependency-monitor")
        recovery_enabled = os.getenv("AGENT_RUNTIME_RECOVERY_LOOP_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        recovery_interval = max(
            1.0,
            float(os.getenv("AGENT_RUNTIME_RECOVERY_INTERVAL_SECONDS", str(max(1.0, execution_store.lease_seconds / 3)))),
        )
        recovery_batch_size = max(1, min(1000, int(os.getenv("AGENT_RUNTIME_RECOVERY_BATCH_SIZE", "100"))))

        async def recover_once() -> None:
            for record in await execution_store.list_recovery_candidates(recovery_batch_size):
                if runtime_state["active"].get(record.run_id) is not None:
                    continue
                try:
                    admission_failure = dependency_monitor.unavailable(langgraph_dependency_requirements(record.payload))
                    if admission_failure:
                        logger.info(
                            "Runtime recovery deferred | run_id=%s dependency=%s reason=%s",
                            record.run_id,
                            admission_failure["dependency"],
                            admission_failure["reason"],
                        )
                        continue
                    recovered_request = request_from_dict(record.request)
                    fencing_token = await execution_store.claim(record.run_id)
                    if fencing_token is None:
                        # Another worker still owns a valid lease. The next
                        # scan must reconsider this record after expiry.
                        continue
                    task = asyncio.create_task(
                        _execute_operation(record.payload, record.operation, recovered_request, attempt=record.attempt, fencing_token=fencing_token),
                        name=f"agent-runtime-recovery-{record.run_id}",
                    )
                    runtime_state["active"][record.run_id] = task
                    task.add_done_callback(
                        lambda done_task, run_id=record.run_id: runtime_state["active"].pop(run_id, None)
                        if runtime_state["active"].get(run_id) is done_task else None
                    )
                except Exception:
                    logger.exception("Unable to recover runtime execution | run_id=%s", record.run_id)

        recovery_task: asyncio.Task[Any] | None = None
        if recovery_enabled:
            await recover_once()

            async def recovery_loop() -> None:
                while not runtime_state["draining"]:
                    try:
                        await asyncio.sleep(recovery_interval)
                        if not runtime_state["draining"]:
                            await recover_once()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception("Runtime recovery scan failed")

            recovery_task = asyncio.create_task(recovery_loop(), name="agent-runtime-recovery-loop")
        yield
        runtime_state["draining"] = True
        dependency_stop.set()
        dependency_task.cancel()
        try:
            await dependency_task
        except asyncio.CancelledError:
            pass
        if recovery_task is not None:
            recovery_task.cancel()
            try:
                await recovery_task
            except asyncio.CancelledError:
                pass
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

    @app.exception_handler(DependencyUnavailable)
    async def dependency_unavailable_handler(request: Request, exc: DependencyUnavailable) -> JSONResponse:
        return JSONResponse(
            json_envelope(
                status="failed",
                request_id=request.headers.get("x-request-id"),
                error={
                    "code": "runtime_dependency_unavailable",
                    "safe_message": str(exc),
                    "retryable": True,
                    "details": exc.details,
                },
                runtime_metadata={"framework": "langgraph", "builder_id": "langgraph_graph"},
            ),
            status_code=503,
        )

    def get_adapter() -> Any:
        """Load legacy execution code only when an execution operation is called."""
        nonlocal adapter
        if adapter is None:
            from app.runtime.langgraph_adapter import LangGraphRuntimeAdapter

            adapter = LangGraphRuntimeAdapter()
        return adapter

    async def _preflight_operation(run_id: str, payload: Mapping[str, Any], operation: str) -> None:
        """Return an HTTP conflict before opening an SSE response."""
        record = await execution_store.get(run_id)
        if record is None or record.status not in {"completed", "failed", "cancelled", "no_continuation"}:
            return
        request = request_from_dict(payload["request"])
        fingerprint = request_fingerprint(operation, request.to_dict())
        existing = record.request_fingerprint or request_fingerprint(record.operation, record.request)
        if fingerprint != existing:
            raise HTTPException(status_code=409, detail={"code": "runtime_operation_conflict", "safe_message": "terminal execution is immutable; use retry", "retryable": False})

    def _admit_dependencies(payload: Mapping[str, Any]) -> None:
        failure = dependency_monitor.unavailable(langgraph_dependency_requirements(payload))
        if failure:
            raise DependencyUnavailable(failure)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "langgraph-runtime"}

    @app.get("/startupz")
    async def startupz() -> JSONResponse:
        started = bool(runtime_state["started"])
        return JSONResponse({"status": "ok" if started else "starting"}, status_code=200 if started else 503)

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        checks: dict[str, Any] = {
            "startup": {"status": "ok" if runtime_state["started"] else "failed"},
            "draining": {"status": "failed" if runtime_state["draining"] else "ok"},
        }
        try:
            checks["execution_store"] = {"status": "ok" if await execution_store.health() else "failed"}
        except Exception as exc:
            checks["execution_store"] = {"status": "failed", "error": type(exc).__name__}
        try:
            from app.agent_workflows.checkpointing import open_agent_checkpointer

            async with open_agent_checkpointer(setup=False) as checkpointer:
                await checkpointer.aget_tuple({"configurable": {"thread_id": "__runtime_readiness__", "checkpoint_ns": ""}})
            checks["checkpoint_store"] = {
                "status": "ok",
                "backend": os.getenv("ASKPDF_AGENT_CHECKPOINTER", "memory").strip().lower(),
            }
        except Exception as exc:
            checks["checkpoint_store"] = {"status": "failed", "error": type(exc).__name__}
        if os.getenv("AGENT_RUNTIME_LEGACY_STRICT_READINESS", "false").strip().lower() in {"1", "true", "yes", "on"}:
            for name, value in dependency_monitor.snapshot().items():
                checks[name] = {"status": "ok" if value["state"] == "available" else "failed", "state": value["state"]}
        healthy = all(value.get("status") == "ok" for value in checks.values())
        runtime_state["readiness"] = checks
        return JSONResponse({"status": "ok" if healthy else "not_ready", "checks": checks}, status_code=200 if healthy else 503)

    @app.get("/v1/dependencies")
    async def dependencies(request: Request) -> dict[str, Any]:
        return json_envelope(
            status="ok",
            request_id=request.headers.get("x-request-id"),
            result={"dependencies": dependency_monitor.snapshot(), "counters": dict(dependency_monitor.counters)},
            runtime_metadata={"framework": "langgraph", "builder_id": "langgraph_graph"},
        )

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

    async def _execute_operation(payload: Mapping[str, Any], operation: str, request: Any, *, attempt: int, fencing_token: int) -> None:
        """Run independently of any HTTP subscriber and journal every event."""
        run_id = request.run_id
        cancellation_event = asyncio.Event()
        owner_id = execution_store.owner_id

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
                await execution_store.append(run_id, event.to_dict(), attempt=attempt, owner_id=owner_id, fencing_token=fencing_token)

        await execution_store.set_status(run_id, "running", owner_id=owner_id, fencing_token=fencing_token)
        heartbeat_stop = asyncio.Event()

        async def heartbeat() -> None:
            interval = max(1.0, execution_store.lease_seconds / 3)
            while not heartbeat_stop.is_set():
                try:
                    await asyncio.wait_for(heartbeat_stop.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    if not await execution_store.heartbeat(run_id, owner_id=owner_id, fencing_token=fencing_token):
                        cancellation_event.set()
                        return

        heartbeat_task = asyncio.create_task(heartbeat(), name=f"agent-runtime-heartbeat-{run_id}")
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
            await execution_store.append(run_id, terminal.to_dict(), result=result.to_dict(), attempt=attempt, owner_id=owner_id, fencing_token=fencing_token)
            await execution_store.set_status(run_id, result.status, result=result.to_dict(), owner_id=owner_id, fencing_token=fencing_token)
        except LeaseLostError:
            logger.warning("Runtime worker lost its lease; abandoning execution | run_id=%s", run_id)
            return
        except asyncio.TimeoutError:
            error = RuntimeError("runtime_execution_timeout", "Agent runtime execution timed out", retryable=True)
            result = AgentRuntimeResult(status="failed", error=error.to_dict())
            terminal = AgentRuntimeEvent(event_id=f"{run_id}:terminal", run_id=run_id, sequence=0, kind="run.failed", terminal=True, payload={"error": error.to_dict()})
            await execution_store.append(run_id, terminal.to_dict(), result=result.to_dict(), attempt=attempt, owner_id=owner_id, fencing_token=fencing_token)
            await execution_store.set_status(run_id, "failed", result=result.to_dict(), error=error.to_dict(), owner_id=owner_id, fencing_token=fencing_token)
        except RuntimeError as exc:
            logger.exception("LangGraph runtime failed | run_id=%s", run_id)
            result = AgentRuntimeResult(status="failed", error=exc.to_dict())
            terminal = AgentRuntimeEvent(event_id=f"{run_id}:terminal", run_id=run_id, sequence=0, kind="run.failed", terminal=True, payload={"error": exc.to_dict()})
            await execution_store.append(run_id, terminal.to_dict(), result=result.to_dict(), attempt=attempt, owner_id=owner_id, fencing_token=fencing_token)
            await execution_store.set_status(run_id, "failed", result=result.to_dict(), error=exc.to_dict(), owner_id=owner_id, fencing_token=fencing_token)
        except Exception as exc:
            logger.exception("LangGraph runtime execution failed | run_id=%s", run_id)
            error = RuntimeError.from_exception(exc, code="runtime_execution_failed", retryable=False, safe_message="Agent runtime execution failed")
            result = AgentRuntimeResult(status="failed", error=error.to_dict())
            terminal = AgentRuntimeEvent(event_id=f"{run_id}:terminal", run_id=run_id, sequence=0, kind="run.failed", terminal=True, payload={"error": error.to_dict()})
            await execution_store.append(run_id, terminal.to_dict(), result=result.to_dict(), attempt=attempt, owner_id=owner_id, fencing_token=fencing_token)
            await execution_store.set_status(run_id, "failed", result=result.to_dict(), error=error.to_dict(), owner_id=owner_id, fencing_token=fencing_token)
        finally:
            heartbeat_stop.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    async def stream_operation(
        payload: Mapping[str, Any],
        operation: str,
        expected_run_id: str | None = None,
        after_sequence: int = 0,
        *,
        allow_start: bool = True,
        operation_id: str | None = None,
        source_attempt: int | None = None,
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
                try:
                    record = await execution_store.create(
                        request.run_id,
                        operation,
                        request.to_dict(),
                        payload,
                        operation_id=operation_id,
                        source_attempt=source_attempt,
                    )
                except ExecutionConflictError as exc:
                    raise HTTPException(status_code=409, detail={"code": "runtime_operation_conflict", "safe_message": str(exc), "retryable": False}) from exc
            else:
                record = await execution_store.get(request.run_id)
                if record is None:
                    raise HTTPException(status_code=404, detail="runtime run not found")

            task = runtime_state["active"].get(request.run_id)
            terminal_statuses = {"completed", "failed", "cancelled", "no_continuation"}
            should_start = allow_start and (
                record.status == "queued"
            )
            if should_start and (task is None or task.done()):
                fencing_token = await execution_store.claim(request.run_id)
                if fencing_token is None:
                    raise HTTPException(status_code=409, detail="runtime execution is owned by another worker")
                task = asyncio.create_task(
                    _execute_operation(payload, record.operation, request_from_dict(record.request), attempt=record.attempt, fencing_token=fencing_token),
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
        request = request_from_dict(payload["request"])
        await _preflight_operation(request.run_id, payload, "start")
        _admit_dependencies(payload)
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
        await _preflight_operation(run_id, payload, "resume")
        _admit_dependencies(payload)
        return StreamingResponse(stream_operation(payload, "resume", run_id), media_type="text/event-stream")

    @app.post("/v1/runs/{run_id}/continue")
    async def continue_run(run_id: str, payload: Mapping[str, Any]) -> StreamingResponse:
        await _preflight_operation(run_id, payload, "continue_run")
        _admit_dependencies(payload)
        return StreamingResponse(stream_operation(payload, "continue_run", run_id), media_type="text/event-stream")

    @app.post("/v1/runs/{run_id}/retry")
    async def retry(run_id: str, payload: Mapping[str, Any]) -> StreamingResponse:
        request_payload = dict(payload.get("request") or {})
        if str(request_payload.get("run_id") or run_id) != run_id:
            raise HTTPException(status_code=400, detail="run_id does not match request path")
        attempt_id = str(payload.get("attempt_id") or "")
        source_attempt = payload.get("source_attempt")
        if not attempt_id or source_attempt is None:
            raise HTTPException(status_code=400, detail={"code": "retry_metadata_required", "safe_message": "attempt_id and source_attempt are required", "retryable": False})
        retry_payload = dict(payload)
        retry_payload["request"] = {
            **request_payload,
            "retry_operation": str(payload.get("operation") or "start"),
            "retry_request": request_payload,
        }
        _admit_dependencies(retry_payload)
        return StreamingResponse(
            stream_operation(retry_payload, "retry", run_id, operation_id=attempt_id, source_attempt=int(source_attempt)),
            media_type="text/event-stream",
        )

    @app.post("/v1/runs/{run_id}/cancel")
    async def cancel(run_id: str, payload: Mapping[str, Any], request_context: Request) -> dict[str, Any]:
        runtime_request = request_from_dict(payload["request"])
        if runtime_request.run_id != run_id:
            raise HTTPException(status_code=400, detail="run_id does not match request path")
        outcome = await execution_store.request_cancel(runtime_request.run_id)
        if outcome.is_unknown:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "runtime_run_not_found",
                    "safe_message": "Runtime run not found",
                    "retryable": False,
                },
            )
        if outcome.is_terminal:
            result = {
                "run_id": runtime_request.run_id,
                "status": outcome.run_status,
                "cancellation_requested": False,
                "no_op": True,
            }
        else:
            result = {
                "run_id": runtime_request.run_id,
                "status": "cancellation_requested",
                "run_status": outcome.run_status,
                "cancellation_requested": True,
            }
        return json_envelope(status="ok", request_id=request_context.headers.get("x-request-id"), result=result)

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
