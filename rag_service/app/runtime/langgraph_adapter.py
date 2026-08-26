"""In-process LangGraph implementation of the neutral runtime adapter."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping, Optional

from app.runtime.adapter import AgentRuntimeAdapter, RuntimeExecutionContext
from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeCapabilities,
    RuntimeValidationIssue,
    RuntimeValidationResult,
)
from app.runtime.events import create_runtime_event
from app.runtime.observability import normalize_runtime_event


def _result_from_graph(result: Mapping[str, Any]) -> AgentRuntimeResult:
    status = str(result.get("status") or ("clarification" if result.get("clarification_options") else "completed"))
    return AgentRuntimeResult(
        status=status,
        output=result.get("answer") if "answer" in result else result.get("final_output"),
        clarification={"options": list(result["clarification_options"])} if result.get("clarification_options") else None,
        interruption=result.get("pending_interrupt") if isinstance(result.get("pending_interrupt"), Mapping) else None,
        usage=dict(result.get("usage") or result.get("metrics") or {}),
        runtime_metadata={key: result[key] for key in ("agent_run_id", "checkpoint_thread_id", "agent_workflow_id") if key in result},
        error=result.get("agent_error") if isinstance(result.get("agent_error"), Mapping) else None,
    )


def _event_from_graph(event: Mapping[str, Any], *, run_id: str, sequence: int) -> AgentRuntimeEvent:
    data = dict(event.get("data") or {})
    source_kind = str(event.get("event") or event.get("kind") or "runtime.event")
    kind, data = normalize_runtime_event(source_kind, data)
    return create_runtime_event(
        event_id=str(data.get("event_id") or f"{run_id}:{sequence}"),
        run_id=run_id,
        sequence=sequence,
        kind=kind,
        payload=data,
        occurred_at=data.get("occurred_at") or data.get("timestamp"),
        trace_id=data.get("trace_id"),
        source_metadata={"framework": "langgraph", "source_event": source_kind},
    )


class _LangGraphEventBridge:
    """Translate LangGraph's event callback shape into the canonical runtime sink."""

    def __init__(self, run_id: str, sink: Any) -> None:
        self.run_id = run_id
        self.sink = sink
        self.sequence = 0
        self._pending: set[asyncio.Task[None]] = set()

    def _runtime_event(self, kind: str, payload: Mapping[str, Any] | None) -> AgentRuntimeEvent:
        self.sequence += 1
        return _event_from_graph(
            {"event": kind, "data": dict(payload or {})},
            run_id=self.run_id,
            sequence=self.sequence,
        )

    async def emit(self, kind: str, payload: Mapping[str, Any] | None = None) -> None:
        await self.sink.emit_runtime_event(self._runtime_event(str(kind), payload))

    def emit_nowait(self, kind: str, payload: Mapping[str, Any] | None = None) -> None:
        task = asyncio.create_task(self.emit(kind, payload))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def parallel_events(self) -> list[Mapping[str, Any]]:
        getter = getattr(self.sink, "parallel_events", None)
        return list(getter()) if getter is not None else []

    async def drain(self) -> None:
        if self._pending:
            await asyncio.gather(*tuple(self._pending))


def _event_bridge(run_id: str, sink: Any) -> _LangGraphEventBridge | None:
    return _LangGraphEventBridge(run_id, sink) if sink is not None else None
from app.runtime.errors import RuntimeError
from app.runtime.langgraph_capabilities import langgraph_capabilities, langgraph_deployment_capabilities
from app.runtime.langgraph import checkpointing


class LangGraphRuntimeAdapter(AgentRuntimeAdapter):
    framework = "langgraph"
    builder_id = "langgraph_graph"

    async def capabilities(self, definition: AgentDefinition) -> RuntimeCapabilities:
        return langgraph_capabilities(definition)

    async def deployment_capabilities(self) -> RuntimeCapabilities:
        return langgraph_deployment_capabilities()

    async def validate(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> RuntimeValidationResult:
        from app.runtime.langgraph.compiler import WorkflowCompiler
        from app.runtime.langgraph.validator import WorkflowValidator

        validator = WorkflowValidator()
        report = validator.report(dict(spec))
        issues = tuple(
            RuntimeValidationIssue(
                code=str(issue.get("code") or "invalid_workflow"),
                message=str(issue.get("message") or "Invalid workflow"),
                path=issue.get("path"),
                severity=str(issue.get("severity") or "error"),
                details=dict(issue),
            )
            for issue in report.get("issues") or []
            if isinstance(issue, Mapping)
        )
        normalized_spec = None
        if not issues:
            normalized_spec = WorkflowCompiler().materialize_spec(dict(spec))
        return RuntimeValidationResult(
            valid=not issues,
            issues=issues,
            normalized_spec=normalized_spec,
            runtime_metadata={
                "framework": self.framework,
                "builder_id": self.builder_id,
                "definition_id": definition.definition_id,
            },
        )

    def _execution_context(self, request: AgentRuntimeRequest, context: RuntimeExecutionContext) -> dict[str, Any]:
        return {
            **dict(context.agent_run_context or {}),
            "agent_run_id": request.run_id,
            "agent_workflow_id": request.definition_id,
        }

    async def start(
        self,
        request: AgentRuntimeRequest,
        *,
        context: RuntimeExecutionContext,
        event_sink: Any = None,
    ) -> AgentRuntimeResult:
        from app.runtime.langgraph import checkpointing, router_runtime

        bridge = _event_bridge(request.run_id, event_sink)
        async with checkpointing.open_agent_checkpointer() as checkpointer:
            try:
                result = await router_runtime.execute_compiled_rag_chat(
                    request.thread_id,
                    context.request,
                    context.embedding_model,
                    resolved_spec=dict(context.resolved_spec),
                    agent_run_context=self._execution_context(request, context),
                    trace_recorder=context.trace_recorder,
                    checkpointer=checkpointer,
                    execution_event_sink=bridge,
                    cancellation_checker=context.cancellation_checker,
                    persist_product_records=False,
                )
            finally:
                if bridge is not None:
                    await bridge.drain()
        return _result_from_graph(result)

    async def resume(
        self,
        request: AgentRuntimeRequest,
        *,
        interrupt: Mapping[str, Any],
        context: RuntimeExecutionContext,
        event_sink: Any = None,
    ) -> AgentRuntimeResult:
        from app.runtime.langgraph import checkpointing, router_runtime

        run = context.agent_run_context.get("run")
        if run is None:
            raise ValueError("LangGraph resume requires the persisted AgentRun in execution context")
        if context.resolved_spec:
            run.resolved_spec_json = dict(context.resolved_spec)
        kwargs = {
            "trace_recorder": context.trace_recorder,
            "cancellation_checker": context.cancellation_checker,
            "persist_product_records": False,
        }
        bridge = _event_bridge(request.run_id, event_sink)
        if bridge is not None:
            kwargs["execution_event_sink"] = bridge
        async with checkpointing.open_agent_checkpointer() as checkpointer:
            try:
                result = await router_runtime.resume_compiled_rag_chat(
                    run, interrupt=dict(interrupt), checkpointer=checkpointer, **kwargs
                )
            finally:
                if bridge is not None:
                    await bridge.drain()
        return _result_from_graph(result)

    async def continue_run(
        self,
        request: AgentRuntimeRequest,
        *,
        context: RuntimeExecutionContext,
        event_sink: Any = None,
    ) -> Optional[AgentRuntimeResult]:
        from app.runtime.langgraph import checkpointing, router_runtime

        run = context.agent_run_context.get("run")
        if run is None:
            raise ValueError("LangGraph continuation requires the persisted AgentRun in execution context")
        # The HTTP transport carries the authoritative execution snapshot in
        # context.resolved_spec. Keep the runtime execution snapshot complete
        # before invoking the graph continuation.
        if context.resolved_spec:
            run.resolved_spec_json = dict(context.resolved_spec)
        bridge = _event_bridge(request.run_id, event_sink)
        async with checkpointing.open_agent_checkpointer() as checkpointer:
            try:
                result = await router_runtime.continue_compiled_rag_chat(
                    run,
                    checkpointer=checkpointer,
                    trace_recorder=context.trace_recorder,
                    execution_event_sink=bridge,
                    cancellation_checker=context.cancellation_checker,
                    persist_product_records=False,
                )
            finally:
                if bridge is not None:
                    await bridge.drain()
        return _result_from_graph(result) if result is not None else None

    async def cancel(self, request: AgentRuntimeRequest) -> Any:
        from app.agent_workflows import chat_cancellation

        return await chat_cancellation.request_chat_run_cancel(request.run_id, thread_id=request.thread_id)

    async def update_state(
        self,
        request: AgentRuntimeRequest,
        update: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from app.runtime.langgraph.compiler import WorkflowCompiler

        checkpoint_thread_id = str(
            (request.continuation.payload.get("checkpoint_thread_id") if request.continuation else None)
            or ""
        ).strip()
        if not checkpoint_thread_id:
            raise RuntimeError("runtime_binding_missing", "LangGraph state updates require a checkpoint binding")
        resolved_spec = request.options.get("resolved_spec")
        if not isinstance(resolved_spec, Mapping) or not resolved_spec:
            raise RuntimeError("runtime_state_unavailable", "LangGraph state updates require the resolved workflow state")

        async with checkpointing.open_agent_checkpointer() as checkpointer:
            app = WorkflowCompiler().compile(dict(resolved_spec), checkpointer=checkpointer)
            config = {"configurable": {"thread_id": checkpoint_thread_id}}
            snapshot = await app.aget_state(config)
            if not getattr(snapshot, "values", None):
                raise RuntimeError("runtime_state_unavailable", "LangGraph checkpoint state is unavailable")
            updated_config = await app.aupdate_state(config, dict(update))
            updated = await app.aget_state(updated_config)
            return {
                "status": "updated",
                "checkpoint_thread_id": checkpoint_thread_id,
                "state": dict(getattr(updated, "values", None) or {}),
            }

    async def inspect_state(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        continuation = request.continuation
        checkpoint_id = None
        if continuation is not None:
            checkpoint_id = continuation.payload.get("checkpoint_thread_id")
        return {
            "framework": self.framework,
            "builder_id": self.builder_id,
            "checkpoint_thread_id": checkpoint_id,
            "continuation_available": bool(checkpoint_id),
        }

    async def delete_continuation(self, continuation: ContinuationBinding) -> Any:
        from app.runtime.langgraph import checkpointing

        if continuation is None:
            return []
        checkpoint_id = continuation.payload.get("checkpoint_thread_id")
        return await checkpointing.delete_agent_checkpoints([str(checkpoint_id)]) if checkpoint_id else []

    async def project_trace(
        self,
        events: list[Mapping[str, Any]],
        *,
        run_id: str,
        context: RuntimeExecutionContext | None = None,
    ) -> list[Any]:
        return [
            _event_from_graph(event, run_id=run_id, sequence=index)
            for index, event in enumerate(events, start=1)
        ]
