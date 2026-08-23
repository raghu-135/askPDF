from __future__ import annotations

import time
import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional

from app.agent_workflows.chat_cancellation import chat_run_cancel_requested
from app.agent_workflows.debug_trace import AgentTraceRecorder, merge_debug_payloads
from app.agent_workflows.enums import AgentRunResumeAction, InterruptStatus
from app.agent_workflows.execution_stream import AgentExecutionEventSink
from app.agent_workflows.metrics import build_run_metrics
from app.agent_workflows.parallel_observability import project_parallel_events
from app.agent_workflows.repository import AgentWorkflowRepository, InterruptResolutionResult
from app.agent_workflows.builtin_workflows import builtin_workflow_keys
from app.agent_workflows.workflow_runtime import (
    default_agent_workflow_key,
    workflow_is_chat_eligible,
)
from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.catalog import (
    continuation_from_run,
    definition_from_run,
    definition_from_workflow,
    result_to_product_payload,
)
from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest, RuntimeApprovalResponse, RuntimeOperationId, RuntimeSteeringInput
from app.runtime.capability_resolver import require_capability
from app.runtime.registry import adapter_for_definition, get_runtime_registry
from app.runtime.builder_registry import builder_for_definition
from app.services.agent_runtime_projection import AgentRuntimeProjection
from app.db import AgentRunStatus, ChatTurnStatus, get_thread_settings


logger = logging.getLogger(__name__)
CLARIFICATION_REQUIRED_STATUS = "clarification_required"


def _is_web_approval_interrupt(interrupt: Dict[str, Any]) -> bool:
    proposed_tool = interrupt.get("proposed_tool")
    return (
        interrupt.get("type") in {"external_research_approval", "tool_approval"}
        and isinstance(proposed_tool, dict)
        and proposed_tool.get("name") == "search_web"
    )


def _attach_parallel_projection(result: Dict[str, Any], execution_event_sink: Any) -> None:
    if execution_event_sink is None or not hasattr(execution_event_sink, "parallel_events"):
        return
    projection = project_parallel_events(execution_event_sink.parallel_events())
    result["parallel_attempts"] = projection["journal"]
    if result.get("parallel_summary") or projection["summary"].get("dispatch_id"):
        result["parallel_summary"] = {
            **projection["summary"],
            **(result.get("parallel_summary") or {}),
        }


def _workflow_version_info(workflow: Any) -> SimpleNamespace:
    metadata = workflow.metadata_json if isinstance(getattr(workflow, "metadata_json", None), dict) else {}
    try:
        version = int(metadata.get("version") or getattr(workflow, "schema_version", 1) or 1)
    except (TypeError, ValueError):
        version = 1
    return SimpleNamespace(
        id=str(metadata.get("version_id") or f"{workflow.id}:v{version}"),
        version=version,
    )


class AgentRunService:
    """Runs the selected agent workflow, defaulting to the compiled Router graph."""

    def __init__(
        self,
        repository: Optional[AgentWorkflowRepository] = None,
    ):
        self.repository = repository or AgentWorkflowRepository()
        self.projection = AgentRuntimeProjection()

    async def _delete_continuation(self, adapter: Any, binding: Any) -> Any:
        try:
            return await adapter.delete_continuation(binding)
        except Exception as exc:
            if getattr(exc, "code", None) == "runtime_capability_unsupported":
                return {"status": "unsupported", "code": exc.code}
            raise

    async def cancel_agent_run(self, run_id: str, *, thread_id: str) -> Any:
        run = await self.repository.get_run(run_id)
        if run is None or run.thread_id != thread_id:
            return None
        definition = definition_from_run(run)
        await require_capability(definition, RuntimeOperationId.RUN_CANCEL, registry=get_runtime_registry(), run=run)
        adapter = adapter_for_definition(definition)
        request = AgentRuntimeRequest(
            run_id=run.id,
            thread_id=run.thread_id,
            definition_id=definition.definition_id,
            framework=definition.framework,
            builder_id=definition.builder_id,
            continuation=continuation_from_run(run),
        )
        return await adapter.cancel(request)

    async def inspect_agent_run(self, run: Any) -> Dict[str, Any]:
        definition = definition_from_run(run)
        await require_capability(definition, RuntimeOperationId.RUN_INSPECT_STATE, registry=get_runtime_registry(), run=run)
        adapter = adapter_for_definition(definition)
        request = AgentRuntimeRequest(
            run_id=run.id,
            thread_id=run.thread_id,
            definition_id=definition.definition_id,
            framework=definition.framework,
            builder_id=definition.builder_id,
            continuation=continuation_from_run(run),
        )
        return dict(await adapter.inspect_state(request))

    async def operate_agent_run(
        self,
        run: Any,
        operation: RuntimeOperationId,
        *,
        input: Optional[Dict[str, Any]] = None,
        update: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        definition = definition_from_run(run)
        adapter = adapter_for_definition(definition)
        await require_capability(definition, operation, registry=get_runtime_registry(), run=run)
        request = AgentRuntimeRequest(
            run_id=run.id, thread_id=run.thread_id, definition_id=definition.definition_id,
            framework=definition.framework, builder_id=definition.builder_id,
            input=dict(input or {}),
            options={
                "resolved_spec": dict(run.resolved_spec_json or {})
                if isinstance(getattr(run, "resolved_spec_json", None), dict) else {},
                "idempotency_key": idempotency_key,
            },
            task_id=getattr(run, "task_id", None), continuation=continuation_from_run(run),
        )
        if operation is RuntimeOperationId.RUN_SEND_FOLLOWUP:
            return dict(await adapter.send_followup(request, dict(input or {})))
        if operation is RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT:
            return dict(await adapter.interrupt_with_input(request, dict(input or {})))
        if operation is RuntimeOperationId.RUN_STEER_LIVE:
            text = str((input or {}).get("text") or "").strip()
            return dict(await adapter.steer_live(request, RuntimeSteeringInput(text)))
        if operation is RuntimeOperationId.RUN_UPDATE_STATE:
            return dict(await adapter.update_state(request, dict(update or {})))
        raise ValueError(f"Unsupported runtime operation: {operation}")

    async def run_thread_chat(
        self,
        thread_id: str,
        req: Any,
        embedding_model: str,
        *,
        execution_event_sink: Any = None,
    ) -> Dict[str, Any]:
        thread_settings = await get_thread_settings(thread_id)
        hitl_web_approval_override = getattr(req, "hitl_web_approval", None)
        if hitl_web_approval_override is not None:
            thread_settings = {
                **(thread_settings if isinstance(thread_settings, dict) else {}),
                "hitl_web_approval": bool(hitl_web_approval_override),
            }
        agent_settings = thread_settings.get("agent_workflow") if isinstance(thread_settings, dict) else None
        agent_settings = agent_settings if isinstance(agent_settings, dict) else {}
        default_workflow_key = default_agent_workflow_key()
        workflow_id = agent_settings.get("workflow_id") or default_workflow_key
        include_custom_for_lookup = True
        logger.info("Resolving agent workflow for thread %s | requested_workflow=%s", thread_id, workflow_id)

        # Built-in manifests are source-controlled service contracts. Refresh the
        # persisted copy before resolving one so upgrades do not leave an older
        # database spec incompatible with the running service.
        if workflow_id in builtin_workflow_keys() and hasattr(self.repository, "seed_builtin_workflows"):
            await self.repository.seed_builtin_workflows()
        workflow = await self.repository.get_workflow(workflow_id, include_custom=include_custom_for_lookup)
        if workflow is None:
            await self.repository.seed_builtin_workflows()
            workflow = await self.repository.get_workflow(workflow_id, include_custom=include_custom_for_lookup)
        if workflow is None:
            logger.error(
                "Selected agent workflow unavailable; aborting run | thread_id=%s requested_workflow=%s",
                thread_id,
                workflow_id,
            )
            raise RuntimeError(f"Selected agent workflow is unavailable: {workflow_id}")
        if not workflow_is_chat_eligible(workflow.spec_json):
            logger.warning(
                "Ignoring a task-only workflow stored in chat settings | thread_id=%s workflow=%s fallback=%s",
                thread_id,
                workflow.id,
                default_workflow_key,
            )
            workflow = await self.repository.get_workflow(default_workflow_key, include_custom=False)
            if workflow is None:
                raise RuntimeError(f"Default agent workflow is unavailable: {default_workflow_key}")
        logger.info(
            "Selected agent workflow for thread %s | workflow=%s",
            thread_id,
            workflow.id,
        )

        request_overrides = {
            "llm_model": getattr(req, "llm_model", None),
            "use_web_search": getattr(req, "use_web_search", None),
            "use_reranker": getattr(req, "use_reranker", None),
            "replans": getattr(req, "replans", None),
            "system_role": getattr(req, "system_role_override", None),
            "tool_instructions": getattr(req, "tool_instructions_override", None),
            "custom_instructions": getattr(req, "custom_instructions_override", None),
        }
        definition = definition_from_workflow(workflow)
        try:
            provider = builder_for_definition(definition)
            provider_request_overrides = provider.filter_request_overrides(
                definition,
                request_overrides,
                reject_unsupported=False,
            )
            resolved_spec = await provider.resolve(
                definition,
                workflow.spec_json,
                thread_settings=thread_settings,
                request_overrides=provider_request_overrides,
            )
            stored_resolved_spec = dict(await provider.normalize(definition, resolved_spec))
        except ValueError as exc:
            logger.exception(
                "Selected agent workflow failed validation; aborting run | thread_id=%s requested_workflow=%s error=%s",
                thread_id,
                workflow.id,
                exc,
            )
            raise RuntimeError(
                f"Selected agent workflow is incompatible with this service version: {workflow.id}"
            ) from exc
        workflow_version = _workflow_version_info(workflow)

        run = await self.repository.create_run(
            thread_id=thread_id,
            workflow_id=workflow.id,
            workflow_version_id=workflow_version.id if workflow_version is not None else None,
            workflow_version=workflow_version.version if workflow_version is not None else None,
            framework=definition.framework,
            builder_id=definition.builder_id,
            definition_category=getattr(workflow, "category", None),
            resolved_spec_json=stored_resolved_spec,
            run_metadata_json={
                "executed_workflow_id": workflow.id,
                "framework": definition.framework,
                "builder_id": definition.builder_id,
            },
        )

        started = time.perf_counter()
        trace_recorder = AgentTraceRecorder(run)
        if execution_event_sink is None and hasattr(self.repository, "append_run_event"):
            # Non-streaming callers still need the same durable runtime event
            # journal and retained trace projection as SSE callers.
            execution_event_sink = AgentExecutionEventSink(include_details=False)
        if execution_event_sink is not None and hasattr(execution_event_sink, "bind_trace_recorder"):
            execution_event_sink.bind_trace_recorder(trace_recorder)
        if execution_event_sink is not None and hasattr(execution_event_sink, "bind_runtime_binding_persister"):
            execution_event_sink.bind_runtime_binding_persister(self.repository.update_runtime_binding)
        if (
            execution_event_sink is not None
            and hasattr(execution_event_sink, "bind_runtime_event_persister")
            and hasattr(self.repository, "append_run_event")
        ):
            execution_event_sink.bind_runtime_event_persister(run.id, self.repository.append_run_event)
        context = {
            "agent_run_id": run.id,
            "agent_workflow_id": workflow.id,
            "agent_workflow_version": workflow_version.version if workflow_version is not None else None,
            "checkpoint_thread_id": run.checkpoint_thread_id,
        }
        if execution_event_sink is not None:
            await execution_event_sink.emit(
                "run.started",
                {"run_id": run.id, "workflow_id": workflow.id, "status": run.status},
            )

        try:
            logger.info("Invoking compiled agent workflow for thread %s | workflow=%s", thread_id, workflow.id)
            definition = definition_from_workflow(workflow)
            adapter = adapter_for_definition(definition)
            runtime_request = AgentRuntimeRequest(
                run_id=run.id,
                thread_id=thread_id,
                definition_id=definition.definition_id,
                framework=definition.framework,
                builder_id=definition.builder_id,
                input={"question": getattr(req, "question", "")},
                options={
                    "embedding_model": embedding_model,
                    "llm_model": getattr(req, "llm_model", None),
                    "use_web_search": getattr(req, "use_web_search", None),
                    "use_reranker": getattr(req, "use_reranker", None),
                    "replans": getattr(req, "replans", None),
                    "system_role_override": getattr(req, "system_role_override", None),
                    "tool_instructions_override": getattr(req, "tool_instructions_override", None),
                    "custom_instructions_override": getattr(req, "custom_instructions_override", None),
                    "bypass_clarification": bool(getattr(req, "bypass_clarification", False)),
                    "hitl_web_approval": getattr(req, "hitl_web_approval", None),
                },
            )
            runtime_result = await adapter.start(
                runtime_request,
                context=RuntimeExecutionContext(
                    request=req,
                    embedding_model=embedding_model,
                    resolved_spec=stored_resolved_spec,
                    agent_run_context=context,
                    trace_recorder=trace_recorder,
                    cancellation_checker=lambda: chat_run_cancel_requested(run.id),
                ),
                event_sink=execution_event_sink,
            )
            if runtime_result.continuation is not None:
                await self.repository.update_runtime_binding(run.id, runtime_result.continuation)
            result = result_to_product_payload(runtime_result)
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            _attach_parallel_projection(result, execution_event_sink)
            error_json = result.get("agent_error") if isinstance(result, dict) else None
            status = result.get("status") if isinstance(result.get("status"), str) else None
            if status is None:
                status = (
                    AgentRunStatus.FAILED.value
                    if error_json
                    else ChatTurnStatus.CLARIFICATION.value
                    if result.get("clarification_options")
                    else AgentRunStatus.COMPLETED.value
                )
            if status in {AgentRunStatus.COMPLETED.value, AgentRunStatus.FAILED.value}:
                result = await self.projection.project_chat_result(
                    thread_id=thread_id,
                    question=getattr(req, "question", ""),
                    result=result,
                    run_context=context,
                    duration_ms=duration_ms,
                )
            metrics = build_run_metrics(result, duration_ms=duration_ms)
            result.pop("_parallel_attempt_records", None)
            result.pop("_corrective_wave_records", None)
            result.pop("_corrective_metrics_state", None)
            if status == AgentRunStatus.CANCELLED.value:
                try:
                    await self.repository.complete_run(
                        run.id,
                        status=AgentRunStatus.CANCELLED.value,
                        metrics_json=metrics,
                        error_json=error_json,
                    )
                except Exception:
                    logger.exception(
                        "Could not mark temporary canceled chat run terminal before cleanup | "
                        "thread_id=%s run_id=%s",
                        thread_id,
                        run.id,
                    )
                try:
                    await self._delete_continuation(adapter, continuation_from_run(run))
                    deleted = await self.repository.delete_run(run.id)
                    if not deleted:
                        raise RuntimeError(f"Canceled chat Agent Run {run.id} was not found during cleanup")
                except Exception:
                    logger.exception(
                        "Canceled chat cleanup failed; terminal run remains eligible for pruning | "
                        "thread_id=%s run_id=%s checkpoint_thread_id=%s",
                        thread_id,
                        run.id,
                        run.checkpoint_thread_id,
                    )
                result.update(
                    {
                        "agent_run_id": None,
                        "checkpoint_thread_id": None,
                        "agent_trace_refs": None,
                        "agent_workflow_id": workflow.id,
                        "agent_workflow_version": workflow_version.version if workflow_version is not None else None,
                        "node_events": [],
                        "tool_events": [],
                    }
                )
                return result
            if status == CLARIFICATION_REQUIRED_STATUS:
                # Keep a terminal record only as a cleanup fallback. The normal path removes
                # both checkpoint state and the exact run before returning clarification.
                try:
                    await self.repository.complete_run(
                        run.id,
                        status=AgentRunStatus.CLARIFICATION.value,
                        metrics_json=metrics,
                        error_json=error_json,
                    )
                except Exception:
                    logger.exception(
                        "Could not mark temporary clarification run terminal before cleanup | "
                        "thread_id=%s run_id=%s",
                        thread_id,
                        run.id,
                    )
                try:
                    await self._delete_continuation(adapter, continuation_from_run(run))
                    deleted = await self.repository.delete_run(run.id)
                    if not deleted:
                        raise RuntimeError(f"Clarification agent run {run.id} was not found during cleanup")
                except Exception:
                    logger.exception(
                        "Clarification cleanup failed; terminal run remains eligible for pruning | "
                        "thread_id=%s run_id=%s checkpoint_thread_id=%s",
                        thread_id,
                        run.id,
                        run.checkpoint_thread_id,
                    )
                result.update(
                    {
                        "agent_run_id": None,
                        "checkpoint_thread_id": None,
                        "agent_trace_refs": None,
                        "agent_workflow_id": workflow.id,
                        "agent_workflow_version": workflow_version.version if workflow_version is not None else None,
                        "node_events": [],
                        "tool_events": [],
                    }
                )
                return result
            if status == AgentRunStatus.AWAITING_HUMAN.value:
                if hasattr(trace_recorder, "record_interrupted_snapshot"):
                    trace_recorder.record_interrupted_snapshot(
                        interrupt=result.get("pending_interrupt") or {},
                        state=result,
                    )
                if hasattr(trace_recorder, "record_runtime_event"):
                    trace_recorder.record_runtime_event(
                        "checkpoint.created",
                        attributes={
                            "askpdf.run.id": run.id,
                            "askpdf.thread.id": thread_id,
                            "askpdf.checkpoint.thread_id": run.checkpoint_thread_id,
                            "askpdf.status": AgentRunStatus.AWAITING_HUMAN.value,
                        },
                        output_data={
                            "interrupt_id": (result.get("pending_interrupt") or {}).get("interrupt_id"),
                            "route": result.get("route"),
                        },
                    )
                debug_payload = trace_recorder.finalize(
                    run=run,
                    chat_turn_id=None,
                    metrics=metrics,
                    route=result.get("route"),
                    route_reason=result.get("route_reason"),
                    error=error_json,
                    result=result,
                )
                paused_run = await self.repository.mark_run_awaiting_human(
                    run.id,
                    result.get("pending_interrupt") or {},
                    metrics_json=metrics,
                    debug_trace_json=debug_payload,
                )
                if paused_run is not None:
                    result["pending_interrupt"] = paused_run.pending_interrupt_json
                result.update(context)
                return result
            completed_run = await self.repository.complete_run(
                run.id,
                status=status,
                metrics_json=metrics,
                error_json=error_json,
            )
            if completed_run is not None:
                debug_payload = trace_recorder.finalize(
                    run=completed_run,
                    chat_turn_id=result.get("chat_turn_id"),
                    metrics=metrics,
                    route=result.get("route"),
                    route_reason=result.get("route_reason"),
                    error=error_json,
                    result=result,
                )
                await self.repository.set_run_debug_trace(run.id, debug_payload)
            result.update(context)
            return result
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            error_json = {
                "code": "agent_run_failed",
                "raw_message": str(exc),
                "retryable": True,
            }
            metrics = build_run_metrics({"agent_error": error_json}, duration_ms=duration_ms)
            completed_run = await self.repository.complete_run(
                run.id,
                status=AgentRunStatus.FAILED.value,
                metrics_json=metrics,
                error_json=error_json,
            )
            if completed_run is not None:
                debug_payload = trace_recorder.finalize(
                    run=completed_run,
                    chat_turn_id=None,
                    metrics=metrics,
                    error=error_json,
                    result={"agent_error": error_json},
                )
                await self.repository.set_run_debug_trace(run.id, debug_payload)
            raise

    async def resume_agent_run(
        self,
        run_id: str,
        *,
        interrupt_id: str,
        action: str,
        edited_payload: Optional[Dict[str, Any]] = None,
        client_metadata: Optional[Dict[str, Any]] = None,
        selected_option_ids: Optional[list[str]] = None,
        resume_token: Optional[str] = None,
        resume_version: Optional[int] = None,
        expected_thread_id: Optional[str] = None,
        execution_event_sink: Any = None,
        approval_scope: Optional[str] = None,
        approval_feedback: Optional[str] = None,
        approval_modifications: Optional[Dict[str, Any]] = None,
    ) -> Optional[InterruptResolutionResult]:
        current_run = await self.repository.get_run(run_id)
        if current_run is None or (
            expected_thread_id is not None
            and str(getattr(current_run, "thread_id", "")) != str(expected_thread_id)
        ):
            return None
        pending_interrupt = getattr(current_run, "pending_interrupt_json", None)
        if (
            str(getattr(current_run, "status", "")) not in {"completed", "failed", "rejected", "expired", "cancelled"}
            and isinstance(pending_interrupt, dict)
            and pending_interrupt.get("checkpoint_resume") is True
        ):
            operation = RuntimeOperationId(str(pending_interrupt.get("response_operation") or RuntimeOperationId.RUN_RESUME.value))
            await require_capability(
                definition_from_run(current_run),
                operation,
                registry=get_runtime_registry(),
                run=current_run,
            )
        resolution = await self.repository.resolve_pending_interrupt(
            run_id,
            interrupt_id=interrupt_id,
            action=action,
            edited_payload=edited_payload,
            client_metadata=client_metadata,
            selected_option_ids=selected_option_ids,
            resume_token=resume_token,
            resume_version=resume_version,
            expected_thread_id=expected_thread_id,
        )
        if resolution is None:
            return None
        if execution_event_sink is None and hasattr(self.repository, "append_run_event"):
            execution_event_sink = AgentExecutionEventSink(include_details=False)
        if (
            hasattr(execution_event_sink, "bind_runtime_event_persister")
            and hasattr(self.repository, "append_run_event")
        ):
            execution_event_sink.bind_runtime_event_persister(
                resolution.run.id,
                self.repository.append_run_event,
            )
        if execution_event_sink is not None:
            await execution_event_sink.emit(
                "run.started",
                {
                    "run_id": resolution.run.id,
                    "workflow_id": resolution.run.workflow_id,
                    "status": resolution.run.status,
                    "resumed": True,
                },
            )
        if (
            resolution.duplicate
            or resolution.outcome != InterruptStatus.RESUMED.value
            or not isinstance(resolution.interrupt, dict)
            or resolution.interrupt.get("checkpoint_resume") is not True
        ):
            return resolution

        # Long-running tasks are resumed by the leased task runner. The
        # canonical interrupt resolver above still owns validation, audit,
        # duplicate decisions, and resume guards; only execution is deferred.
        if resolution.run.task_id:
            from app.services.agent_task_repository import (
                WEB_ACCESS_ALLOWED,
                WEB_ACCESS_DENIED,
                queue_task_after_interrupt,
                set_task_web_access,
            )

            response_operation = RuntimeOperationId(str(resolution.interrupt.get("response_operation") or RuntimeOperationId.RUN_RESUME.value))
            if response_operation is RuntimeOperationId.RUN_APPROVAL_RESPOND:
                definition = definition_from_run(resolution.run)
                await require_capability(
                    definition,
                    RuntimeOperationId.RUN_APPROVAL_RESPOND,
                    registry=get_runtime_registry(),
                    run=resolution.run,
                )
                request = AgentRuntimeRequest(
                    run_id=resolution.run.id, thread_id=resolution.run.thread_id,
                    definition_id=definition.definition_id, framework=definition.framework,
                    builder_id=definition.builder_id, task_id=resolution.run.task_id,
                    continuation=continuation_from_run(resolution.run),
                )
                await adapter_for_definition(definition).respond_to_approval(
                    request,
                    RuntimeApprovalResponse(
                        decision="reject" if action == AgentRunResumeAction.REJECT.value else "approve",
                        modifications=approval_modifications,
                        feedback=approval_feedback,
                        scope=approval_scope or "once",
                    ),
                )
            if _is_web_approval_interrupt(resolution.interrupt):
                if action == AgentRunResumeAction.APPROVE_FOR_SCOPE.value:
                    await set_task_web_access(
                        resolution.run.task_id,
                        WEB_ACCESS_ALLOWED,
                        agent_run_id=resolution.run.id,
                        interrupt_id=interrupt_id,
                    )
                elif action in {
                    AgentRunResumeAction.CONTINUE_WITHOUT.value,
                    AgentRunResumeAction.REJECT.value,
                }:
                    await set_task_web_access(
                        resolution.run.task_id,
                        WEB_ACCESS_DENIED,
                        agent_run_id=resolution.run.id,
                        interrupt_id=interrupt_id,
                    )

            await queue_task_after_interrupt(
                resolution.run.task_id,
                reason=f"interrupt:{interrupt_id}:{action}",
                interrupt_id=interrupt_id,
                action=action,
            )
            return resolution

        try:
            resume_trace_recorder = AgentTraceRecorder(resolution.run)
            if execution_event_sink is not None and hasattr(execution_event_sink, "bind_trace_recorder"):
                execution_event_sink.bind_trace_recorder(resume_trace_recorder)
            if (
                execution_event_sink is not None
                and hasattr(execution_event_sink, "bind_runtime_event_persister")
                and hasattr(self.repository, "append_run_event")
            ):
                execution_event_sink.bind_runtime_event_persister(resolution.run.id, self.repository.append_run_event)
            if execution_event_sink is not None and hasattr(execution_event_sink, "bind_runtime_binding_persister"):
                execution_event_sink.bind_runtime_binding_persister(self.repository.update_runtime_binding)
            definition = definition_from_run(resolution.run)
            await require_capability(
                definition,
                RuntimeOperationId.RUN_RESUME,
                registry=get_runtime_registry(),
                run=resolution.run,
            )
            adapter = adapter_for_definition(definition)
            runtime_request = AgentRuntimeRequest(
                run_id=resolution.run.id,
                thread_id=resolution.run.thread_id,
                definition_id=definition.definition_id,
                framework=definition.framework,
                builder_id=definition.builder_id,
                continuation=continuation_from_run(resolution.run),
            )
            runtime_result = await adapter.resume(
                runtime_request,
                interrupt=resolution.interrupt,
                context=RuntimeExecutionContext(
                    agent_run_context={"run": resolution.run},
                    trace_recorder=resume_trace_recorder,
                    cancellation_checker=lambda: chat_run_cancel_requested(resolution.run.id),
                ),
                event_sink=execution_event_sink,
            )
            result = result_to_product_payload(runtime_result)
            _attach_parallel_projection(result, execution_event_sink)
            prior_metrics = dict(resolution.run.metrics_json or {})
            metrics = {
                **prior_metrics,
                **build_run_metrics(result, duration_ms=float(result.get("duration_ms") or 0)),
            }
            result.pop("_parallel_attempt_records", None)
            result.pop("_corrective_wave_records", None)
            result.pop("_corrective_metrics_state", None)
            error_json = result.get("agent_error") if isinstance(result, dict) else None
            status = result.get("status") if isinstance(result.get("status"), str) else AgentRunStatus.COMPLETED.value
            if status in {AgentRunStatus.COMPLETED.value, AgentRunStatus.FAILED.value}:
                result = await self.projection.project_chat_result(
                    thread_id=resolution.run.thread_id,
                    question=str(result.get("question") or ""),
                    result=result,
                    run_context={
                        "agent_run_id": resolution.run.id,
                        "agent_workflow_id": resolution.run.workflow_id,
                    },
                    duration_ms=float(result.get("duration_ms") or 0),
                )
            if status == AgentRunStatus.AWAITING_HUMAN.value:
                pending_interrupt = result.get("pending_interrupt") or {}
                if hasattr(resume_trace_recorder, "record_interrupted_snapshot"):
                    resume_trace_recorder.record_interrupted_snapshot(
                        interrupt=pending_interrupt,
                        state=result,
                    )
                if hasattr(resume_trace_recorder, "record_runtime_event"):
                    resume_trace_recorder.record_runtime_event(
                        "checkpoint.created",
                        attributes={
                            "askpdf.run.id": resolution.run.id,
                            "askpdf.thread.id": resolution.run.thread_id,
                            "askpdf.checkpoint.thread_id": resolution.run.checkpoint_thread_id,
                            "askpdf.status": AgentRunStatus.AWAITING_HUMAN.value,
                        },
                        output_data={
                            "interrupt_id": pending_interrupt.get("interrupt_id"),
                            "route": result.get("route"),
                        },
                    )
                resume_debug_payload = resume_trace_recorder.finalize(
                    run=resolution.run,
                    chat_turn_id=None,
                    metrics=metrics,
                    route=result.get("route"),
                    route_reason=result.get("route_reason"),
                    error=error_json,
                    result=result,
                )
                debug_payload = resume_debug_payload
                if isinstance(resolution.run.debug_trace_json, dict):
                    debug_payload = merge_debug_payloads(
                        resolution.run.debug_trace_json,
                        resume_debug_payload,
                        resolved_spec=resolution.run.resolved_spec_json if isinstance(resolution.run.resolved_spec_json, dict) else {},
                        run_status=status,
                        completed_at=None,
                        chat_turn_id=None,
                        metrics=metrics,
                    )
                paused_run = await self.repository.mark_run_awaiting_human(
                    resolution.run.id,
                    pending_interrupt,
                    metrics_json=metrics,
                    debug_trace_json=debug_payload,
                )
                if paused_run is not None:
                    return InterruptResolutionResult(
                        run=paused_run,
                        outcome=resolution.outcome,
                        interrupt=paused_run.pending_interrupt_json or resolution.interrupt,
                        duplicate=False,
                    )
                return resolution

            completed_run = await self.repository.complete_run(
                resolution.run.id,
                status=status,
                metrics_json=metrics,
                error_json=error_json,
            )
            if completed_run is None:
                return resolution
            resume_debug_payload = resume_trace_recorder.finalize(
                run=completed_run,
                chat_turn_id=result.get("chat_turn_id"),
                metrics=metrics,
                route=result.get("route"),
                route_reason=result.get("route_reason"),
                error=error_json,
                result=result,
            )
            if isinstance(completed_run.debug_trace_json, dict):
                debug_payload = merge_debug_payloads(
                    completed_run.debug_trace_json,
                    resume_debug_payload,
                    resolved_spec=completed_run.resolved_spec_json if isinstance(completed_run.resolved_spec_json, dict) else {},
                    run_status=status,
                    completed_at=completed_run.completed_at,
                    chat_turn_id=result.get("chat_turn_id"),
                    metrics=metrics,
                )
                completed_run = await self.repository.set_run_debug_trace(completed_run.id, debug_payload) or completed_run
            else:
                completed_run = await self.repository.set_run_debug_trace(completed_run.id, resume_debug_payload) or completed_run
            return InterruptResolutionResult(
                run=completed_run,
                outcome=resolution.outcome,
                interrupt=resolution.interrupt,
                duplicate=False,
            )
        except Exception as exc:
            prior_metrics = dict(resolution.run.metrics_json or {})
            prior_metrics["error_count"] = max(int(prior_metrics.get("error_count") or 0), 1)
            await self.repository.complete_run(
                resolution.run.id,
                status=AgentRunStatus.FAILED.value,
                metrics_json=prior_metrics,
                error_json={
                    "code": "agent_run_resume_failed",
                    "raw_message": str(exc),
                    "retryable": True,
                },
            )
            raise
