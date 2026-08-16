"""In-process LangGraph implementation of the neutral runtime adapter."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeCapabilities,
    RuntimeValidationIssue,
    RuntimeValidationResult,
)
from app.runtime.langgraph_compat import event_from_legacy, result_from_legacy


class LangGraphRuntimeAdapter:
    framework = "langgraph"
    builder_id = "langgraph_graph"

    async def project_task_result(self, **kwargs: Any) -> dict[str, Any]:
        from app.runtime.langgraph.router_runtime import project_agent_task_result

        return await project_agent_task_result(**kwargs)

    async def capabilities(self, definition: AgentDefinition) -> RuntimeCapabilities:
        features = dict(definition.capabilities or {})
        return RuntimeCapabilities(
            streaming=True,
            resume=True,
            cancellation=True,
            inspection=True,
            continuation_cleanup=True,
            task_execution=bool(features.get("supports_long_running_tasks")),
            native_checkpoints=True,
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

    async def inspect(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
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
