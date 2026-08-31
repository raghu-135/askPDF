"""In-process LangGraph implementation of the neutral runtime adapter."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, Optional

from langgraph_runtime.runtime_adapter import AgentRuntimeAdapter, RuntimeExecutionContext
from runtime_protocol.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeCapabilities,
    RuntimeValidationIssue,
    RuntimeValidationResult,
    RuntimeOperationId,
    TaskOrchestrationDelta,
)
from runtime_protocol.events import create_runtime_event
from langgraph_runtime.runtime_support.observability import normalize_runtime_event
from langgraph_runtime.runtime_support.task_results import normalize_runtime_task_result
from langgraph_runtime.bindings import issue_binding, resolve_binding


def _public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _public_value(item)
            for key, item in value.items()
            if str(key) != "checkpoint_thread_id"
        }
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    return value


def _result_from_graph(result: Mapping[str, Any]) -> AgentRuntimeResult:
    status = str(result.get("status") or ("clarification" if result.get("clarification_options") else "completed"))
    interruption = result.get("pending_interrupt") if isinstance(result.get("pending_interrupt"), Mapping) else None
    checkpoint_thread_id = interruption.get("checkpoint_thread_id") if interruption else result.get("checkpoint_thread_id")
    continuation = (
        ContinuationBinding(
            binding_type="langgraph.checkpoint",
            payload={"binding_id": issue_binding(
                checkpoint_thread_id=str(checkpoint_thread_id),
                run_id=str(result.get("agent_run_id") or ""),
            )},
        )
        if checkpoint_thread_id and status in {"awaiting_human", "paused"}
        else None
    )
    public_result = _public_value(result)
    public_interruption = _public_value(interruption) if interruption is not None else None
    task_id = str(result.get("agent_task_id") or "")
    orchestration_delta = None
    if task_id:
        run_id = str(result.get("agent_run_id") or "")
        orchestration_delta = TaskOrchestrationDelta(
            event_id=f"{run_id}:task-delta:{int(result.get('task_plan_revision') or 0)}",
            attempt_id=run_id,
            idempotency_key=f"task-delta:{run_id}",
            observed_task_version=int(result.get("agent_task_version") or 0),
            observed_plan_revision=int(result.get("task_plan_revision") or 0),
            plan=dict(result["task_plan"]) if isinstance(result.get("task_plan"), Mapping) else None,
            todo_changes=tuple(dict(value) for value in result.get("task_todos") or [] if isinstance(value, Mapping)),
            subagent_changes=tuple(dict(value) for value in result.get("task_result_packets") or [] if isinstance(value, Mapping)),
            budget_usage=dict(result.get("task_budget_usage") or {}),
            web_access={"status": str(result.get("task_web_access"))} if result.get("task_web_access") else None,
            artifacts=tuple(dict(value) for value in result.get("runtime_artifacts") or [] if isinstance(value, Mapping)),
            pending_interrupt=dict(public_interruption) if isinstance(public_interruption, Mapping) else None,
            result={"status": status, "incomplete_reasons": list(result.get("task_incomplete_reasons") or [])},
        )
    final_text = result.get("final_answer") or result.get("answer")
    task_result = None
    if status == "completed" and (final_text or result.get("runtime_artifacts")):
        task_result = normalize_runtime_task_result({
            "status": "completed_with_warnings" if result.get("warnings") or result.get("task_incomplete_reasons") or result.get("task_result_gaps") else "completed",
            "text": final_text,
            "structured_output": result.get("structured_output"),
            "warnings": result.get("warnings") or [],
            "gaps": list(dict.fromkeys([*(result.get("task_incomplete_reasons") or []), *(result.get("task_result_gaps") or [])])),
            "usage": result.get("usage") or result.get("metrics") or {},
            "framework_details": {"framework": "langgraph"},
        })
    return AgentRuntimeResult(
        status=status,
        output=dict(public_result),
        task_result=task_result,
        clarification={"options": list(result["clarification_options"])} if result.get("clarification_options") else None,
        interruption=public_interruption,
        usage=dict(result.get("usage") or result.get("metrics") or {}),
        runtime_metadata={key: result[key] for key in ("agent_run_id", "agent_workflow_id") if key in result},
        continuation=continuation,
        error=result.get("agent_error") if isinstance(result.get("agent_error"), Mapping) else None,
        checkpoint_boundary_available=(
            bool(result["checkpoint_boundary_available"])
            if "checkpoint_boundary_available" in result
            else True if continuation is not None else None
        ),
        orchestration_delta=orchestration_delta,
    )


def _event_from_graph(event: Mapping[str, Any], *, run_id: str, sequence: int) -> AgentRuntimeEvent:
    data = dict(event.get("data") or {})
    source_kind = str(event.get("event") or event.get("kind") or "runtime.event")
    kind, data = normalize_runtime_event(source_kind, data)
    checkpoint_thread_id = data.get("checkpoint_thread_id")
    continuation = (
        ContinuationBinding(
            binding_type="langgraph.checkpoint",
            payload={"binding_id": issue_binding(
                checkpoint_thread_id=str(checkpoint_thread_id), run_id=run_id,
            )},
        )
        if checkpoint_thread_id and kind == "interrupt.requested"
        else None
    )
    return create_runtime_event(
        event_id=str(data.get("event_id") or f"{run_id}:{sequence}"),
        run_id=run_id,
        sequence=sequence,
        kind=kind,
        payload=dict(_public_value(data)),
        occurred_at=data.get("occurred_at") or data.get("timestamp"),
        trace_id=data.get("trace_id"),
        source_metadata={
            "framework": "langgraph",
            "source_event": source_kind,
            "visualization_id": "langgraph.graph",
        },
        continuation=continuation,
        checkpoint_boundary_available=(
            bool(data["checkpoint_boundary_available"])
            if "checkpoint_boundary_available" in data
            else True if continuation is not None else None
        ),
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
from runtime_protocol.errors import RuntimeError
from langgraph_runtime.capabilities import langgraph_capabilities, langgraph_deployment_capabilities
from langgraph_runtime import checkpointing


class LangGraphRuntimeAdapter(AgentRuntimeAdapter):
    framework = "langgraph"
    builder_id = "langgraph_graph"
    supports_task_pause = True
    implemented_operations = frozenset({
        RuntimeOperationId.RUN_START,
        RuntimeOperationId.RUN_CANCEL,
        RuntimeOperationId.RUN_RESUME,
        RuntimeOperationId.RUN_INSPECT_STATE,
        RuntimeOperationId.RUN_CONTINUATION_CLEANUP,
        RuntimeOperationId.TRACE_PROJECT,
    })

    async def prepare_execution_context(
        self,
        context: RuntimeExecutionContext,
    ) -> RuntimeExecutionContext:
        task = context.task_context
        if task is None:
            return context
        metadata = dict(task.metadata or {})
        permissions = dict(task.permissions or {})
        request = SimpleNamespace(
            question=task.objective,
            llm_model=metadata.get("llm_model"),
            context_window=metadata.get("context_window"),
            use_web_search=bool(permissions.get("use_web_search")),
            web_search_mode=str(permissions.get("web_search_mode") or "off"),
            task_web_access=permissions.get("web_access"),
            use_reranker=bool(metadata.get("use_reranker", True)),
            bypass_clarification=True,
            system_role_override="",
            tool_instructions_override={},
            custom_instructions_override="",
            client_timezone=None,
            client_locale=None,
            client_now_iso=None,
            agent_task_id=task.task_id,
            agent_task_version=metadata.get("task_version"),
            task_enabled_profiles=list(metadata.get("enabled_profiles") or []),
            task_limits=dict(task.limits or {}),
            task_plan_revision=int(metadata.get("plan_revision") or 0),
            task_run_plan_count=0,
            task_todos=[dict(todo) for todo in task.todos],
            task_budget_usage=dict(metadata.get("budget_usage") or {}),
            task_orchestration=dict(metadata.get("orchestration") or {}),
            runtime_execution_mode=True,
            runtime_artifact_manifest=[dict(value) for value in task.artifact_manifests],
            runtime_artifact_contents=dict(task.artifact_contents),
        )
        return replace(context, request=request)

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
        from langgraph_runtime.compiler import WorkflowCompiler
        from langgraph_runtime.validator import WorkflowValidator

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
            "checkpoint_thread_id": request.run_id,
        }

    @staticmethod
    def _run_with_continuation(request: AgentRuntimeRequest, run: Any) -> Any:
        binding = request.continuation
        checkpoint_thread_id = (
            resolve_binding(str(binding.payload.get("binding_id") or ""), run_id=request.run_id)
            if binding is not None and binding.binding_type == "langgraph.checkpoint"
            else None
        )
        if not checkpoint_thread_id:
            raise RuntimeError("runtime_binding_missing", "LangGraph continuation requires a checkpoint binding")
        runtime_run = copy.copy(run)
        runtime_run.checkpoint_thread_id = str(checkpoint_thread_id)
        return runtime_run

    async def start(
        self,
        request: AgentRuntimeRequest,
        *,
        context: RuntimeExecutionContext,
        event_sink: Any = None,
    ) -> AgentRuntimeResult:
        from langgraph_runtime import checkpointing, router_runtime

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
                    pause_checker=context.pause_checker,
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
        from langgraph_runtime import checkpointing, router_runtime

        run = context.agent_run_context.get("run") or SimpleNamespace(
            id=request.run_id,
            thread_id=request.thread_id,
            workflow_id=request.definition_id,
            resolved_spec_json=dict(context.resolved_spec),
            checkpoint_thread_id=None,
        )
        run = self._run_with_continuation(request, run)
        if context.resolved_spec:
            run.resolved_spec_json = dict(context.resolved_spec)
        kwargs = {
            "trace_recorder": context.trace_recorder,
            "cancellation_checker": context.cancellation_checker,
            "pause_checker": context.pause_checker,
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
        from langgraph_runtime import checkpointing, router_runtime

        run = context.agent_run_context.get("run") or SimpleNamespace(
            id=request.run_id,
            thread_id=request.thread_id,
            workflow_id=request.definition_id,
            resolved_spec_json=dict(context.resolved_spec),
            checkpoint_thread_id=None,
        )
        run = self._run_with_continuation(request, run)
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
                    pause_checker=context.pause_checker,
                )
            finally:
                if bridge is not None:
                    await bridge.drain()
        return _result_from_graph(result) if result is not None else None

    async def cancel(self, request: AgentRuntimeRequest) -> Any:
        from langgraph_runtime.workflows import cancellation

        return await cancellation.request_chat_run_cancel(request.run_id, thread_id=request.thread_id)

    async def inspect_state(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        from langgraph_runtime.compiler import WorkflowCompiler

        checkpoint_thread_id = resolve_binding(
            str(request.continuation.payload.get("binding_id") or "") if request.continuation else "",
            run_id=request.run_id,
        )
        if not checkpoint_thread_id:
            raise RuntimeError("runtime_binding_missing", "LangGraph state inspection requires a checkpoint binding")
        resolved_spec = request.options.get("resolved_spec")
        if not isinstance(resolved_spec, Mapping) or not resolved_spec:
            raise RuntimeError("runtime_state_unavailable", "LangGraph state inspection requires the resolved workflow state")

        async with checkpointing.open_agent_checkpointer() as checkpointer:
            app = WorkflowCompiler().compile(dict(resolved_spec), checkpointer=checkpointer)
            config = {"configurable": {"thread_id": checkpoint_thread_id}}
            snapshot = await app.aget_state(config)
        values = getattr(snapshot, "values", None)
        if values is None:
            raise RuntimeError("runtime_state_unavailable", "LangGraph checkpoint state is unavailable")
        return {
            "framework": self.framework,
            "builder_id": self.builder_id,
            "continuation_available": True,
            "state": dict(values),
            "next": list(getattr(snapshot, "next", ()) or ()),
            "metadata": dict(getattr(snapshot, "metadata", {}) or {}),
        }

    async def delete_continuation(self, continuation: ContinuationBinding) -> Any:
        from langgraph_runtime import checkpointing

        if continuation is None:
            return []
        checkpoint_id = resolve_binding(str(continuation.payload.get("binding_id") or ""))
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
