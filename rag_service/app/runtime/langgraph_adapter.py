"""In-process LangGraph implementation of the neutral runtime adapter."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from app.runtime.adapter import AgentRuntimeAdapter, RuntimeExecutionContext
from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeCapabilities,
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeSupportLevel,
    RuntimeValidationIssue,
    RuntimeValidationResult,
)
from app.runtime.langgraph_compat import event_from_legacy, result_from_legacy
from app.runtime.errors import RuntimeError
# Module-level compatibility aliases keep the Phase 5 monkeypatch seam stable
# while the provider owns builder selection in Phase 6.
from app.runtime.langgraph import checkpointing, router_runtime


class LangGraphRuntimeAdapter(AgentRuntimeAdapter):
    framework = "langgraph"
    builder_id = "langgraph_graph"

    async def capabilities(self, definition: AgentDefinition) -> RuntimeCapabilities:
        task_runtime = bool(definition.capabilities.get("supports_long_running_tasks"))
        return RuntimeCapabilities(
            operations={
                RuntimeOperationId.RUN_START.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.NATIVE, True,
                ),
                RuntimeOperationId.RUN_RESUME.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.NATIVE, True,
                    semantics="resume_from_interrupt",
                ),
                RuntimeOperationId.RUN_CANCEL.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.NATIVE, True,
                    modes=("interrupt",),
                    confirmation="asynchronous",
                    terminal_states=("cancelled", "interrupted"),
                ),
                RuntimeOperationId.RUN_PAUSE.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.CONDITIONAL, task_runtime,
                    semantics="product_task_pause",
                    disabled_reason=None if task_runtime else "definition_not_task_runtime",
                ),
                RuntimeOperationId.RUN_RETRY.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.CONDITIONAL, task_runtime,
                    semantics="product_task_retry",
                    disabled_reason=None if task_runtime else "definition_not_task_runtime",
                ),
                RuntimeOperationId.RUN_EVENTS.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.NATIVE, True,
                ),
                RuntimeOperationId.RUN_INSPECT_STATE.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.NATIVE, True,
                ),
                RuntimeOperationId.RUN_APPROVAL_RESPOND.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.UNSUPPORTED, False,
                    disabled_reason="runtime_capability_unsupported",
                ),
                RuntimeOperationId.RUN_STEER_LIVE.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.UNSUPPORTED, False,
                    disabled_reason="runtime_capability_unsupported",
                ),
                RuntimeOperationId.RUN_SEND_FOLLOWUP.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.UNSUPPORTED, False,
                    disabled_reason="runtime_capability_unsupported",
                ),
                RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.UNSUPPORTED, False,
                    disabled_reason="runtime_capability_unsupported",
                ),
                RuntimeOperationId.RUN_UPDATE_STATE.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.NATIVE, True,
                    semantics="checkpoint_boundary_update",
                ),
                RuntimeOperationId.RUN_CONTINUATION_CLEANUP.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.NATIVE, True,
                ),
            },
        )

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

    def _legacy_context(self, request: AgentRuntimeRequest, context: RuntimeExecutionContext) -> dict[str, Any]:
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

        async with checkpointing.open_agent_checkpointer() as checkpointer:
            result = await router_runtime.execute_compiled_rag_chat(
                request.thread_id,
                context.request,
                context.embedding_model,
                resolved_spec=dict(context.resolved_spec),
                agent_run_context=self._legacy_context(request, context),
                trace_recorder=context.trace_recorder,
                checkpointer=checkpointer,
                execution_event_sink=event_sink,
                cancellation_checker=context.cancellation_checker,
                persist_product_records=False,
            )
        return result_from_legacy(result)

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
        if event_sink is not None:
            kwargs["execution_event_sink"] = event_sink
        async with checkpointing.open_agent_checkpointer() as checkpointer:
            try:
                result = await router_runtime.resume_compiled_rag_chat(
                    run, interrupt=dict(interrupt), checkpointer=checkpointer, **kwargs
                )
            except TypeError as exc:
                # Preserve the staged-migration monkeypatch seam for callers
                # that still provide the pre-Phase-4 function signature.
                if "persist_product_records" not in str(exc):
                    raise
                kwargs.pop("persist_product_records", None)
                result = await router_runtime.resume_compiled_rag_chat(
                    run, interrupt=dict(interrupt), checkpointer=checkpointer, **kwargs
                )
        return result_from_legacy(result)

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
        # context.resolved_spec. Inject it into the legacy run-shaped object
        # consumed by continue_compiled_rag_chat; otherwise a serialized
        # SQLModel/namespace may contain only identity fields and compile {}.
        if context.resolved_spec:
            run.resolved_spec_json = dict(context.resolved_spec)
        async with checkpointing.open_agent_checkpointer() as checkpointer:
            result = await router_runtime.continue_compiled_rag_chat(
                run,
                checkpointer=checkpointer,
                trace_recorder=context.trace_recorder,
                execution_event_sink=event_sink,
                cancellation_checker=context.cancellation_checker,
                persist_product_records=False,
            )
        return result_from_legacy(result) if result is not None else None

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
            event_from_legacy(event, run_id=run_id, sequence=index)
            for index, event in enumerate(events, start=1)
        ]
