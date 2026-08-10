from __future__ import annotations

import time
import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional

from app.agent_workflows.checkpointing import delete_agent_checkpoints, open_agent_checkpointer
from app.agent_workflows.chat_cancellation import chat_run_cancel_requested
from app.agent_workflows.debug_trace import AgentTraceRecorder, merge_debug_payloads
from app.agent_workflows.enums import AgentRunResumeAction, InterruptStatus
from app.agent_workflows.metrics import build_run_metrics
from app.agent_workflows.parallel_observability import project_parallel_events
from app.agent_workflows.repository import AgentWorkflowRepository, InterruptResolutionResult
from app.agent_workflows.validator import WorkflowResolver, WorkflowValidationError
from app.agent_workflows.workflow_runtime import default_agent_workflow_key
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
        resolver: Optional[WorkflowResolver] = None,
    ):
        self.repository = repository or AgentWorkflowRepository()
        self.resolver = resolver or WorkflowResolver()

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
        runtime_features = ((workflow.spec_json.get("runtime") or {}).get("features") or {})
        if runtime_features.get("supports_long_running_tasks"):
            raise RuntimeError("This workflow must be started through the Deep research task workspace")
        logger.info(
            "Selected agent workflow for thread %s | workflow=%s",
            thread_id,
            workflow.id,
        )

        request_overrides = {
            "use_web_search": getattr(req, "use_web_search", None),
            "use_reranker": getattr(req, "use_reranker", None),
            "replans": getattr(req, "replans", None),
            "system_role": getattr(req, "system_role_override", None),
            "tool_instructions": getattr(req, "tool_instructions_override", None),
            "custom_instructions": getattr(req, "custom_instructions_override", None),
        }
        try:
            resolved_spec = self.resolver.resolve(
                workflow.spec_json,
                thread_settings=thread_settings,
                request_overrides=request_overrides,
            )
        except WorkflowValidationError as exc:
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
        from app.agent_workflows.compiler import WorkflowCompiler
        from app.agent_workflows.graph import normalize_hitl_policy_for_thread_settings

        resolved_config = resolved_spec.get("config") if isinstance(resolved_spec.get("config"), dict) else {}
        resolved_config["hitl_policy"] = normalize_hitl_policy_for_thread_settings(
            resolved_config.get("hitl_policy"),
            thread_settings,
        )
        resolved_spec["config"] = resolved_config
        stored_resolved_spec = WorkflowCompiler().materialize_spec(
            resolved_spec,
        )

        run = await self.repository.create_run(
            thread_id=thread_id,
            workflow_id=workflow.id,
            workflow_version_id=workflow_version.id if workflow_version is not None else None,
            workflow_version=workflow_version.version if workflow_version is not None else None,
            resolved_spec_json=stored_resolved_spec,
            run_metadata_json={
                "executed_workflow_id": workflow.id,
            },
        )

        started = time.perf_counter()
        trace_recorder = AgentTraceRecorder(run)
        if execution_event_sink is not None and hasattr(execution_event_sink, "bind_trace_recorder"):
            execution_event_sink.bind_trace_recorder(trace_recorder)
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
            from app.agent_workflows import router_runtime

            async with open_agent_checkpointer() as checkpointer:
                stream_kwargs = {
                    "cancellation_checker": lambda: chat_run_cancel_requested(run.id),
                }
                if execution_event_sink is not None:
                    stream_kwargs["execution_event_sink"] = execution_event_sink
                result = await router_runtime.execute_compiled_rag_chat(
                    thread_id,
                    req,
                    embedding_model,
                    resolved_spec=stored_resolved_spec,
                    agent_run_context=context,
                    trace_recorder=trace_recorder,
                    checkpointer=checkpointer,
                    **stream_kwargs,
                )
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
                    await delete_agent_checkpoints([str(run.checkpoint_thread_id or run.id)])
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
                    await delete_agent_checkpoints([str(run.checkpoint_thread_id or run.id)])
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
    ) -> Optional[InterruptResolutionResult]:
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
            from app.agent_workflows.router_runtime import resume_compiled_rag_chat

            resume_trace_recorder = AgentTraceRecorder(resolution.run)
            if execution_event_sink is not None and hasattr(execution_event_sink, "bind_trace_recorder"):
                execution_event_sink.bind_trace_recorder(resume_trace_recorder)
            async with open_agent_checkpointer() as checkpointer:
                stream_kwargs = {
                    "cancellation_checker": lambda: chat_run_cancel_requested(resolution.run.id),
                }
                if execution_event_sink is not None:
                    stream_kwargs["execution_event_sink"] = execution_event_sink
                result = await resume_compiled_rag_chat(
                    resolution.run,
                    interrupt=resolution.interrupt,
                    checkpointer=checkpointer,
                    trace_recorder=resume_trace_recorder,
                    **stream_kwargs,
                )
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
