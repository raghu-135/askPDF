from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from app.agent_workflows.debug_trace import AgentTraceRecorder, finalize_and_merge_debug_payload
from app.agent_workflows.execution_stream import AgentExecutionEventSink
from app.agent_workflows.repository import AgentWorkflowRepository
from app.db import AgentRunStatus, get_thread, get_thread_settings
from app.models.deep_research import AgentTaskStatus
from app.services import agent_task_repository as tasks
from app.services.agent_runtime_reconciliation import record_terminal_result
from app.services.agent_run_cancellation import request_task_cancellation
from app.services.task_artifact_service import persist_task_artifact
from app.services.agent_grounding_evaluator import AgentGroundingEvaluator
from app.services.agent_task_maintenance import MAINTENANCE_INTERVAL_SECONDS, run_task_maintenance
from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import AgentRuntimeRequest, AgentRuntimeResult, RuntimeOperationId, RuntimeTaskContext
from app.runtime.capability_resolver import require_capability
from app.runtime.errors import RuntimeError as AgentRuntimeError
from app.runtime.catalog import (
    continuation_from_run,
    definition_from_run,
    definition_from_workflow,
    result_to_product_payload,
)
from app.runtime.registry import RuntimeRegistry, adapter_for_definition, get_runtime_registry
from app.runtime.builder_registry import builder_for_definition
from app.runtime.operational_limits import positive_float_value
from app.runtime.task_results import normalize_runtime_task_result


logger = logging.getLogger(__name__)
grounding_evaluator = AgentGroundingEvaluator()
LEASE_SECONDS = 60
HEARTBEAT_SECONDS = 15


async def _invoke_task_runtime(
    *,
    adapter: Any,
    definition: Any,
    run: Any,
    runtime_request: AgentRuntimeRequest,
    runtime_context: RuntimeExecutionContext,
    runtime_event_sink: Any,
    repository: AgentWorkflowRepository,
    registry: RuntimeRegistry,
) -> AgentRuntimeResult | None:
    """Dispatch one task attempt using its explicit lifecycle contract."""

    projection = dict((getattr(run, "run_metadata_json", None) or {}).get("projection") or {})
    persisted_result = projection.get("runtime_result")
    if isinstance(persisted_result, dict):
        persisted_task_result = persisted_result.get("runtime_task_result")
        return AgentRuntimeResult(
            status=str(persisted_result.get("status") or AgentRunStatus.FAILED.value),
            output=(dict(persisted_result) if isinstance(persisted_task_result, Mapping) else persisted_result.get("answer")),
            task_result=(
                normalize_runtime_task_result(persisted_task_result)
                if isinstance(persisted_task_result, Mapping)
                else None
            ),
            interruption=persisted_result.get("pending_interrupt"),
            runtime_metadata=dict(persisted_result.get("runtime_metadata") or {}),
            error=dict(persisted_result.get("agent_error") or {}),
        )

    pending = dict(run.pending_interrupt_json or {})
    if getattr(run, "_fresh_runtime_run", False):
        await require_capability(
            definition,
            RuntimeOperationId.RUN_START,
            registry=registry,
            run=run,
        )
        # Submission may commit upstream before the streaming response is
        # established. Persist ownership first so cancellation and recovery
        # never mistake an active external execution for an unsubmitted run.
        await repository.mark_runtime_started(run.id)
        result = await adapter.start(
            runtime_request,
            context=runtime_context,
            event_sink=runtime_event_sink,
        )
        return result

    if pending.get("status") in {"resumed", "resolved"} and isinstance(pending.get("decision"), dict):
        response_operation = pending.get("response_operation")
        if response_operation == RuntimeOperationId.RUN_RESUME.value:
            await require_capability(
                definition,
                RuntimeOperationId.RUN_RESUME,
                registry=registry,
                run=run,
                include_resolved_response=True,
            )
            return await adapter.resume(
                runtime_request,
                interrupt=pending,
                context=runtime_context,
                event_sink=runtime_event_sink,
            )
        if response_operation == RuntimeOperationId.RUN_APPROVAL_RESPOND.value:
            await require_capability(
                definition,
                RuntimeOperationId.RUN_APPROVAL_RESPOND,
                registry=registry,
                run=run,
                include_resolved_response=True,
            )
            return await adapter.continue_run(
                runtime_request,
                context=runtime_context,
                event_sink=runtime_event_sink,
            )
        raise AgentRuntimeError(
            code="interrupt_response_operation_invalid",
            safe_message="The pending interrupt does not declare a supported response operation",
            retryable=False,
            details={"response_operation": response_operation},
        )

    return await adapter.continue_run(
        runtime_request,
        context=runtime_context,
        event_sink=runtime_event_sink,
    )


async def _task_context_snapshot(task: Any, thread: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Create a bounded, deterministic context seed; retrieval remains MCP-backed."""

    from app.db import get_recent_messages
    messages = await get_recent_messages(task.thread_id, limit=20)
    conversation: list[dict[str, str]] = []
    context_window = int(config.get("context_window") or 32_768)
    remaining = min(24_000, max(4_000, context_window))
    for message in reversed(messages):
        content = str(getattr(message, "context_compact", None) or getattr(message, "content", "")).strip()
        if not content or remaining <= 0:
            continue
        content = content[-remaining:]
        conversation.append({"role": str(getattr(message, "role", "user")), "content": content})
        remaining -= len(content)
    conversation.reverse()
    documents = []
    for file_hash, metadata in sorted(dict(getattr(thread, "documents_meta", None) or {}).items()):
        item = dict(metadata or {}) if isinstance(metadata, dict) else {}
        documents.append({
            "file_hash": str(file_hash),
            "name": str(item.get("file_name") or item.get("filename") or file_hash),
        })
    return {
        "objective": task.objective,
        "thread_id": task.thread_id,
        "project_id": task.project_id,
        "model": config.get("llm_model"),
        "embedding_model": thread.embedding_model,
        "context_window": context_window,
        "limits": dict(config.get("limits") or {}),
        "recent_conversation": conversation,
        "documents": documents,
    }


async def _complete_run_with_trace(
    repository: AgentWorkflowRepository,
    *,
    run: Any,
    recorder: AgentTraceRecorder,
    status: str,
    metrics: dict[str, Any],
    result: dict[str, Any],
    error: Optional[dict[str, Any]] = None,
) -> Any:
    """Atomically persist one terminal AgentRun and its merged trace payload."""

    completed_at = datetime.now(timezone.utc)
    debug_payload = finalize_and_merge_debug_payload(
        recorder=recorder,
        run=run,
        metrics=metrics,
        result=result,
        route=result.get("route"),
        route_reason=result.get("route_reason"),
        error=error,
        run_status=status,
        completed_at=completed_at,
    )
    return await repository.complete_run(
        run.id,
        status=status,
        metrics_json=metrics,
        error_json=error,
        debug_trace_json=debug_payload,
        completed_at=completed_at,
    )


async def _finalize_task_run(
    *,
    task: Any,
    run: Any,
    recorder: AgentTraceRecorder,
    sink: AgentExecutionEventSink,
    run_status: str,
    task_status: str,
    metrics: dict[str, Any],
    result: dict[str, Any],
    error: Optional[dict[str, Any]] = None,
    reason: Optional[str] = None,
    final_artifact_id: Optional[str] = None,
) -> None:
    completed_at = datetime.now(timezone.utc)
    terminal_kind = (
        "run.cancelled" if run_status == AgentRunStatus.CANCELLED.value
        else "run.failed" if run_status == AgentRunStatus.FAILED.value
        else "run.completed"
    )

    async def commit(terminal_event: Any) -> None:
        debug_payload = finalize_and_merge_debug_payload(
            recorder=recorder,
            run=run,
            metrics=metrics,
            result=result,
            route=result.get("route"),
            route_reason=result.get("route_reason"),
            error=error,
            run_status=run_status,
            completed_at=completed_at,
        )
        await tasks.finalize_task_run(
            task.id,
            run.id,
            run_status=run_status,
            task_status=task_status,
            metrics=metrics,
            error=error,
            debug_trace=debug_payload,
            terminal_reason=reason,
            terminal_event=terminal_event,
            final_artifact_id=final_artifact_id,
            completed_at=completed_at,
        )

    await sink.finish(
        terminal_kind,
        {
            "run_id": run.id,
            "task_id": task.id,
            "status": run_status,
            "response": result,
            "error": error,
            "terminal_reason": reason,
        },
        terminal_committer=commit,
    )


async def ensure_task_run(task_id: str):
    task = await tasks.get_task(task_id)
    if task is None:
        raise ValueError("task_not_found")
    active = await tasks.get_task_run(task_id)
    if active is not None and active.status in {AgentRunStatus.RUNNING.value, AgentRunStatus.AWAITING_HUMAN.value}:
        metadata = dict(active.run_metadata_json or {})
        binding = dict(active.runtime_binding_json or {})
        binding_payload = dict(binding.get("payload") or {})
        # A runtime can commit its continuation before the caller receives the
        # first event. If that caller is interrupted, runtime_started is
        # still false even though the product run owns an upstream execution.
        # Retire the partial attempt after admitted cancellation and let the
        # normal path allocate a new immutable run identity.
        if metadata.get("runtime_started") is False and binding_payload:
            definition = definition_from_run(active)
            adapter = adapter_for_definition(definition)
            await require_capability(
                definition,
                RuntimeOperationId.RUN_CANCEL,
                registry=get_runtime_registry(),
                run=active,
            )
            cancel_request = AgentRuntimeRequest(
                run_id=active.id,
                thread_id=active.thread_id,
                definition_id=definition.definition_id,
                framework=definition.framework,
                builder_id=definition.builder_id,
                task_id=task.id,
                continuation=continuation_from_run(active),
            )
            await adapter.cancel(cancel_request)
            await AgentWorkflowRepository().complete_run(
                active.id,
                status=AgentRunStatus.CANCELLED.value,
                error_json={
                    "code": "runtime_start_interrupted",
                    "retryable": True,
                    "details": {"replaced_by_new_attempt": True},
                },
            )
            active = None
        else:
            # Mark an unsubmitted active run explicitly so it is not mistaken
            # for a continuation after a worker restart.
            setattr(active, "_fresh_runtime_run", metadata.get("runtime_started") is False)
            return active

    repository = AgentWorkflowRepository()
    workflow = await repository.get_workflow(task.workflow_id, include_custom=False)
    if workflow is None:
        await repository.seed_builtin_workflows()
        workflow = await repository.get_workflow(task.workflow_id, include_custom=False)
    if workflow is None:
        raise RuntimeError("deep_research_workflow_unavailable")

    thread_settings = await get_thread_settings(task.thread_id)
    definition = definition_from_workflow(workflow)
    provider = builder_for_definition(definition)
    resolved = await provider.resolve(
        definition,
        workflow.spec_json,
        thread_settings=thread_settings,
        request_overrides={
            "llm_model": (task.config_json or {}).get("llm_model"),
            "context_window": (task.config_json or {}).get("context_window"),
            "use_web_search": bool((task.config_json or {}).get("use_web_search")),
        },
    )
    config = dict(resolved.get("config") or {})
    task_policy = dict(config.get("task_policy") or {})
    task_policy["limits"] = dict((task.config_json or {}).get("limits") or {})
    task_policy["profiles"] = list((task.config_json or {}).get("enabled_profiles") or [])
    config["task_policy"] = task_policy
    config["use_web_search"] = bool((task.config_json or {}).get("use_web_search"))
    resolved["config"] = config
    frozen_spec = dict(await provider.normalize(definition, resolved))
    metadata = dict(getattr(workflow, "metadata_json", None) or {})
    version = int(metadata.get("version") or workflow.schema_version or 1)
    run = await repository.create_run(
        thread_id=task.thread_id,
        workflow_id=workflow.id,
        workflow_version_id=str(metadata.get("version_id") or f"{workflow.id}:v{version}"),
        workflow_version=version,
        framework=definition.framework,
        builder_id=definition.builder_id,
        definition_category=getattr(workflow, "category", None),
        resolved_spec_json=frozen_spec,
        user_id=task.user_id,
        run_metadata_json={
            "executed_workflow_id": workflow.id,
            "run_kind": "agent_task",
            "agent_task_id": task.id,
            "runtime_started": False,
        },
    )
    # attach_run reloads the winning row in its own session, so apply the
    # process-local fresh-run marker to that returned instance rather than the
    # detached create_run instance. This also handles a concurrent creator
    # winning the task attachment while preserving persisted runtime state.
    attached = await tasks.attach_run(
        task.id,
        run,
        parent_run_id=active.id if active is not None else None,
    )
    attached_metadata = dict(attached.run_metadata_json or {})
    setattr(attached, "_fresh_runtime_run", attached_metadata.get("runtime_started") is False)
    return attached


async def _heartbeat(task_id: str, worker_id: str) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        if not await tasks.heartbeat_task(task_id, worker_id, lease_seconds=LEASE_SECONDS):
            return


async def execute_claimed_task(task_id: str, worker_id: str) -> None:
    task = await tasks.get_task(task_id)
    if task is None:
        return
    if task.status == AgentTaskStatus.CANCELLING.value:
        active_run = await tasks.get_task_run(task_id)
        if active_run is not None and active_run.status in {
            AgentRunStatus.RUNNING.value,
            AgentRunStatus.AWAITING_HUMAN.value,
        }:
            try:
                await request_task_cancellation(task, active_run)
            except AgentRuntimeError as exc:
                logger.warning(
                    "Runtime cancellation remains pending after worker recovery | task_id=%s run_id=%s code=%s",
                    task_id,
                    active_run.id,
                    exc.code,
                )
        elif active_run is None:
            await tasks.complete_task(
                task_id,
                status=AgentTaskStatus.CANCELLED.value,
                reason="cancelled_by_user",
            )
        await tasks.release_task_lease(task_id, worker_id, lease_seconds=LEASE_SECONDS)
        return
    if await tasks.active_runtime_budget_exhausted(task_id):
        await tasks.complete_task(
            task_id,
            status=AgentTaskStatus.FAILED.value,
            reason="active_runtime_budget_exhausted",
        )
        await tasks.release_task_lease(task_id, worker_id, lease_seconds=LEASE_SECONDS)
        return
    run = await ensure_task_run(task_id)
    task = await tasks.get_task(task_id)
    thread = await get_thread(task.thread_id) if task else None
    if task is None or thread is None:
        await tasks.complete_task(task_id, status=AgentTaskStatus.FAILED.value, reason="task_thread_missing")
        await tasks.release_task_lease(task_id, worker_id, lease_seconds=LEASE_SECONDS)
        return

    config = dict(task.config_json or {})
    review_context = [
        dict(value) for value in config.get("result_review_context") or []
        if isinstance(value, Mapping)
    ]
    followup_input = str((review_context[-1] if review_context else {}).get("followup_input") or "").strip()
    runtime_question = task.objective
    if followup_input:
        runtime_question = f"{task.objective}\n\nResult review follow-up: {followup_input}"
    existing_artifacts = await tasks.list_artifacts(task.id)
    artifact_manifest: list[dict[str, Any]] = []
    artifact_contents: dict[str, str] = {}
    from app.services.content_store import get_content_store
    content_store = get_content_store()
    for artifact in existing_artifacts[: int(config.get("max_context_artifacts", 200))]:
        manifest = {
            "id": artifact.id, "kind": artifact.kind, "sha256": artifact.sha256,
            "byte_size": artifact.byte_size, "summary": artifact.summary_json,
            "todo_id": artifact.todo_id, "subagent_run_id": artifact.subagent_run_id,
            "provenance": dict(artifact.provenance_json or {}),
        }
        artifact_manifest.append(manifest)
        if artifact.kind in {"intermediate_report", "context_summary", "tool_output"} and artifact.byte_size <= 20_000:
            try:
                artifact_contents[artifact.id] = (await content_store.read(artifact.object_key)).decode("utf-8", errors="replace")
            except (FileNotFoundError, OSError):
                continue
    todos = await tasks.list_todos(task.id)
    task_web_access = await tasks.get_task_web_access(task.id)
    repository = AgentWorkflowRepository()
    trace = AgentTraceRecorder(run)
    context = {
        "agent_run_id": run.id,
        "agent_workflow_id": run.workflow_id,
        "agent_workflow_version": run.workflow_version,
        "checkpoint_thread_id": run.checkpoint_thread_id,
    }
    started = time.perf_counter()
    runtime_event_sink = AgentExecutionEventSink(include_details=False)
    runtime_event_sink.detach_delivery()
    runtime_event_sink.bind_trace_recorder(trace)
    runtime_event_sink.bind_runtime_binding_persister(repository.update_runtime_binding)
    runtime_event_sink.bind_runtime_fact_persister(repository.update_run_metadata_fields)
    existing_run_events = await repository.list_run_events(run.id)
    runtime_event_sink.bind_runtime_event_persister(
        run.id,
        repository.append_run_event,
        initial_sequence=max(
            (int(getattr(event, "sequence", 0) or 0) for event in existing_run_events),
            default=0,
        ),
    )
    heartbeat = asyncio.create_task(_heartbeat(task.id, worker_id))
    async def cancellation_requested() -> bool:
        return await tasks.task_cancel_requested(task.id) or await tasks.active_runtime_budget_exhausted(task.id)

    try:
        definition = definition_from_run(run)
        adapter = adapter_for_definition(definition)
        resolved_spec = dict(run.resolved_spec_json or {})
        if not resolved_spec:
            raise AgentRuntimeError(
                "runtime_definition_invalid",
                "The task run has no materialized runtime definition",
                retryable=False,
            )
        task_context = RuntimeTaskContext(
            task_id=task.id,
            objective=runtime_question,
            todos=tuple({
                "id": todo.id,
                "title": todo.title,
                "description": todo.description,
                "completion_criteria": todo.completion_criteria,
                "status": todo.status,
                "priority": todo.priority,
                "required": todo.required,
                "dependency_ids": list(todo.dependency_ids_json or []),
                "profile_id": todo.profile_id,
                "attempt": todo.attempt,
                "max_attempts": todo.max_attempts,
                "progress": todo.progress,
                "result_summary": todo.result_summary,
                "artifact_ids": list(todo.artifact_ids_json or []),
                "version": todo.version,
            } for todo in todos),
            artifact_manifests=tuple(artifact_manifest),
            artifact_contents=dict(artifact_contents),
            limits=dict(config.get("limits") or {}),
            permissions={
                "use_web_search": bool(config.get("use_web_search")),
                "web_search_mode": str(config.get("web_search_mode") or "off"),
                "web_access": task_web_access,
            },
            metadata={
                "llm_model": config.get("llm_model"),
                "context_window": config.get("context_window"),
                "use_reranker": True,
                "task_version": task.version,
                "enabled_profiles": list(config.get("enabled_profiles") or []),
                "plan_revision": max((todo.updated_revision for todo in todos), default=0),
                "budget_usage": dict(task.budgets_json or {}),
                "orchestration": dict(
                    (dict((resolved_spec.get("config") or {}).get("task_policy") or {})).get("orchestration")
                    or {}
                ),
            },
            context_data=await _task_context_snapshot(task, thread, config),
        )
        runtime_request = AgentRuntimeRequest(
            run_id=run.id,
            thread_id=run.thread_id,
            definition_id=definition.definition_id,
            framework=definition.framework,
            builder_id=definition.builder_id,
            input={"question": runtime_question},
            task_id=task.id,
            continuation=continuation_from_run(run),
        )
        runtime_context = RuntimeExecutionContext(
            embedding_model=thread.embedding_model,
            resolved_spec=resolved_spec,
            agent_run_context={**context, "run": run},
            trace_recorder=trace,
            cancellation_checker=cancellation_requested,
            task_id=task.id,
            task_worker_id=worker_id,
            task_context=task_context,
        )
        runtime_context = await adapter.prepare_execution_context(runtime_context)
        runtime_request = await adapter.prepare_request(runtime_request, context=runtime_context)
        runtime_result = await _invoke_task_runtime(
            adapter=adapter,
            definition=definition,
            run=run,
            runtime_request=runtime_request,
            runtime_context=runtime_context,
            runtime_event_sink=runtime_event_sink,
            repository=repository,
            registry=get_runtime_registry(),
        )
        if runtime_result is None:
            # A continuation is optional at the runtime boundary. A missing
            # checkpoint is a terminal runtime outcome.
            runtime_result = AgentRuntimeResult(
                status="failed",
                error={
                    "code": "runtime_continuation_missing",
                    "message": "The runtime did not return a durable continuation for this run",
                    "retryable": False,
                },
            )
        if runtime_result.continuation is not None:
            await repository.update_runtime_binding(run.id, runtime_result.continuation)
        if runtime_result.checkpoint_boundary_available is not None:
            await repository.update_run_metadata_fields(run.id, {
                "checkpoint_boundary_available": runtime_result.checkpoint_boundary_available,
            })
        result = result_to_product_payload(runtime_result)
        if str(result.get("status") or "") in {
            AgentRunStatus.COMPLETED.value,
            AgentRunStatus.FAILED.value,
            AgentRunStatus.CANCELLED.value,
        } and not dict((getattr(run, "run_metadata_json", None) or {}).get("projection") or {}).get("runtime_result"):
            await record_terminal_result(run, result)
        await runtime_event_sink.flush()
        # Runtime artifacts are data, not product records. Project them in
        # rag-service after the stream completes and translate deterministic
        # runtime IDs to the persisted artifact IDs used by task APIs.
        runtime_artifacts = [value for value in result.get("runtime_artifacts") or [] if isinstance(value, dict) and value.get("content") is not None]
        artifact_id_map: dict[str, str] = {}
        for artifact in runtime_artifacts:
            projected = await persist_task_artifact(
                task_id=task.id,
                agent_run_id=run.id,
                kind=str(artifact.get("kind") or "tool_output"),
                content=str(artifact.get("content") or ""),
                media_type=str(artifact.get("media_type") or "text/plain"),
                todo_id=artifact.get("todo_id"),
                # Runtime subagent IDs are intentionally opaque and are not
                # product FK values; task ownership is retained in provenance.
                provenance={**dict(artifact.get("provenance") or {}), "runtime_artifact_id": artifact.get("artifact_id"), "runtime_subagent_run_id": artifact.get("subagent_run_id")},
                source_refs=dict(artifact.get("source_refs") or {}),
            )
            if artifact.get("artifact_id"):
                artifact_id_map[str(artifact["artifact_id"])] = projected.id
        if artifact_id_map:
            def replace_ids(value: Any) -> Any:
                if isinstance(value, str):
                    return artifact_id_map.get(value, value)
                if isinstance(value, list):
                    return [replace_ids(item) for item in value]
                if isinstance(value, dict):
                    return {key: replace_ids(item) for key, item in value.items()}
                return value
            for key in ("task_todos", "task_artifact_manifest", "task_evidence_manifest", "task_result_packets"):
                if key in result:
                    result[key] = replace_ids(result[key])
        status = str(result.get("status") or AgentRunStatus.COMPLETED.value)
        metrics = dict(run.metrics_json or {})
        metrics.update({"duration_ms": round((time.perf_counter() - started) * 1000, 2)})
        if status == AgentRunStatus.AWAITING_HUMAN.value:
            pending = dict(result.get("pending_interrupt") or {})
            trace.record_interrupted_snapshot(interrupt=pending, state=result)
            trace.record_runtime_event(
                "checkpoint.created",
                attributes={
                    "askpdf.run.id": run.id,
                    "askpdf.thread.id": task.thread_id,
                    "askpdf.checkpoint.thread_id": run.checkpoint_thread_id,
                    "askpdf.status": AgentRunStatus.AWAITING_HUMAN.value,
                },
                output_data={
                    "interrupt_id": pending.get("interrupt_id"),
                    "route": result.get("route"),
                },
            )
            debug_payload = finalize_and_merge_debug_payload(
                recorder=trace,
                run=run,
                metrics=metrics,
                result=result,
                route=result.get("route"),
                route_reason=result.get("route_reason"),
                run_status=AgentRunStatus.AWAITING_HUMAN.value,
            )
            await repository.mark_run_awaiting_human(
                run.id,
                pending,
                metrics_json=metrics,
                debug_trace_json=debug_payload,
            )
            task_status = AgentTaskStatus.PAUSED.value if pending.get("type") == "task_pause" else AgentTaskStatus.AWAITING_APPROVAL.value
            await tasks.set_task_runtime_status(task.id, task_status, phase="checkpointed_interrupt")
            if task_status == AgentTaskStatus.AWAITING_APPROVAL.value:
                await tasks.append_event(
                    task.id,
                    "task.approval_requested",
                    agent_run_id=run.id,
                    payload={
                        "interrupt_id": pending.get("interrupt_id"),
                        "title": pending.get("title"),
                        "type": pending.get("type"),
                        "approval_scope_kind": pending.get("approval_scope_kind"),
                    },
                )
            await runtime_event_sink.finish_boundary()
            return
        if status == AgentRunStatus.CANCELLED.value:
            latest_task = await tasks.get_task(task.id)
            budget_exhausted = bool(latest_task and latest_task.terminal_reason == "active_runtime_budget_exhausted")
            await _finalize_task_run(
                task=task, run=run, recorder=trace, sink=runtime_event_sink,
                run_status=status,
                task_status=AgentTaskStatus.FAILED.value if budget_exhausted else AgentTaskStatus.CANCELLED.value,
                metrics=metrics, result=result,
                reason="active_runtime_budget_exhausted" if budget_exhausted else "cancelled_by_user",
            )
            return
        error = result.get("agent_error") if isinstance(result.get("agent_error"), dict) else None
        if status == AgentRunStatus.COMPLETED.value:
            terminal_todos = await tasks.list_todos(task.id)
            incomplete = any(todo.required and todo.status != "completed" for todo in terminal_todos)
            final_answer = str(result.get("final_answer") or result.get("answer") or "").strip()
            if not final_answer:
                terminal_error = {"code": "final_report_missing", "retryable": True}
                await _finalize_task_run(
                    task=task, run=run, recorder=trace, sink=runtime_event_sink,
                    run_status=AgentRunStatus.FAILED.value, task_status=AgentTaskStatus.FAILED.value,
                    metrics=metrics, result=result, error=terminal_error, reason="final_report_missing",
                )
                return
            evidence_policy = dict((resolved_spec.get("config") or {}).get("task_policy") or {}).get("evidence")
            if evidence_policy == "document_when_available":
                grounding = grounding_evaluator.evaluate(
                    result,
                    await repository.list_run_events(run.id),
                    documents_present=bool(dict(getattr(thread, "documents_meta", None) or {})),
                    artifacts=await tasks.list_artifacts(task.id, agent_run_id=run.id),
                )
                metrics["grounding"] = grounding
                result["grounding"] = grounding
            evidence_manifest = [
                value for value in result.get("task_evidence_manifest") or []
                if isinstance(value, dict) and value.get("id")
            ]
            incomplete_reasons = [str(value) for value in result.get("task_incomplete_reasons") or []]
            result_warnings = [
                dict(value) for value in result.get("warnings") or [] if isinstance(value, Mapping)
            ]
            grounding = result.get("grounding") if isinstance(result.get("grounding"), Mapping) else None
            if grounding is not None and grounding.get("grounded") is False:
                incomplete_reasons.append(
                    f"Required {grounding.get('requirement') or 'research'} evidence was not established."
                )
                result_warnings.append({
                    "code": "grounding_requirement_unsatisfied",
                    "details": dict(grounding),
                })
            incomplete_reasons = list(dict.fromkeys(incomplete_reasons))
            if incomplete and not incomplete_reasons:
                incomplete_reasons = [
                    f"Required work item {todo.id} ended with status {todo.status}."
                    for todo in terminal_todos
                    if todo.required and todo.status != "completed"
                ]
            final_artifact = await persist_task_artifact(
                task_id=task.id,
                agent_run_id=run.id,
                kind="final_report",
                content=final_answer,
                provenance={
                    "incomplete": incomplete,
                    "draft_model": result.get("task_draft_metadata") or {},
                    "quality_review": result.get("task_critic_report") or {},
                    "plan_revision": int(
                        result.get("task_plan_revision")
                        or task_context.metadata.get("plan_revision")
                        or 0
                    ),
                    "evidence_manifest": evidence_manifest,
                    "evidence_gaps": incomplete_reasons,
                    "warnings": result_warnings,
                    "outcome": "completed_with_warnings" if incomplete or result_warnings else "completed",
                },
                source_refs={"artifact_ids": [str(item["id"]) for item in evidence_manifest]},
            )
            orchestration = dict(
                (dict((resolved_spec.get("config") or {}).get("task_policy") or {})).get("orchestration")
                or {}
            )
            incomplete_policy = str(orchestration.get("incomplete_result_policy") or "review")
            max_review_rounds = min(5, max(0, int(orchestration.get("max_incomplete_review_rounds") or 3)))
            needs_review = bool(incomplete or incomplete_reasons or result_warnings)
            if needs_review and incomplete_policy == "fail":
                await _finalize_task_run(
                    task=task, run=run, recorder=trace, sink=runtime_event_sink,
                    run_status=AgentRunStatus.FAILED.value,
                    task_status=AgentTaskStatus.FAILED.value,
                    metrics=metrics, result=result,
                    error={"code": "incomplete_result_rejected", "retryable": False},
                    reason="incomplete_result_rejected",
                    final_artifact_id=final_artifact.id,
                )
                return
            review_round = max(1, int(getattr(run, "task_attempt", 1) or 1))
            if needs_review and incomplete_policy == "review" and review_round <= max_review_rounds:
                pending = {
                    "interrupt_id": f"result-review:{run.id}:{review_round}",
                    "type": "incomplete_result_review",
                    "kind": "approval",
                    "title": "Review incomplete result",
                    "body": "The agent returned usable output with warnings or unresolved gaps.",
                    "response_operation": RuntimeOperationId.TASK_RESULT_REVIEW_RESPOND.value,
                    "allowed_actions": ["accept", "retry_with_input"],
                    "response_schema": {
                        "type": "object",
                        "properties": {"followup_input": {"type": "string", "maxLength": 20000}},
                    },
                    "review_round": review_round,
                    "max_review_rounds": max_review_rounds,
                    "provisional_artifact_id": final_artifact.id,
                    "provisional_answer": final_answer,
                    "warnings": result_warnings,
                    "gaps": incomplete_reasons,
                }
                debug_payload = finalize_and_merge_debug_payload(
                    recorder=trace, run=run, metrics=metrics, result={
                        **result,
                        "result_outcome": "completed_with_warnings",
                        "warnings": result_warnings,
                        "gaps": incomplete_reasons,
                        "provisional_artifact_id": final_artifact.id,
                    },
                    route=result.get("route"), route_reason="awaiting_review",
                    run_status=AgentRunStatus.AWAITING_HUMAN.value,
                )
                await repository.mark_run_awaiting_human(
                    run.id, pending, metrics_json=metrics, debug_trace_json=debug_payload,
                )
                await tasks.set_task_runtime_status(
                    task.id, AgentTaskStatus.AWAITING_APPROVAL.value,
                    phase="awaiting_result_review", reason="incomplete_result",
                )
                await tasks.append_event(
                    task.id, "task.result_review_requested", agent_run_id=run.id,
                    artifact_id=final_artifact.id,
                    payload={
                        "interrupt_id": pending["interrupt_id"],
                        "result_outcome": "completed_with_warnings",
                        "warnings": result_warnings,
                        "gaps": incomplete_reasons,
                        "review_round": review_round,
                    },
                )
                await runtime_event_sink.finish_boundary()
                return
            await _finalize_task_run(
                task=task, run=run, recorder=trace, sink=runtime_event_sink,
                run_status=status, task_status=AgentTaskStatus.COMPLETED.value,
                metrics=metrics, result=result, error=error,
                reason="completed_with_warnings" if needs_review else "completed",
                final_artifact_id=final_artifact.id,
            )
        else:
            await _finalize_task_run(
                task=task, run=run, recorder=trace, sink=runtime_event_sink,
                run_status=status, task_status=AgentTaskStatus.FAILED.value,
                metrics=metrics, result=result, error=error,
                reason=str((error or {}).get("code") or status),
            )
    except Exception as exc:
        logger.exception("Deep research task execution failed | task_id=%s run_id=%s", task.id, run.id)
        terminal_error = exc.to_dict() if isinstance(exc, AgentRuntimeError) else {
            "code": str(getattr(exc, "code", "deep_research_execution_failed")),
            "type": type(exc).__name__,
            "raw_message": str(exc)[:1000],
            "retryable": bool(getattr(exc, "retryable", True)),
            **({"field_path": str(exc.field_path)} if getattr(exc, "field_path", None) else {}),
            **({"correlation_id": str(exc.correlation_id)} if getattr(exc, "correlation_id", None) else {}),
        }
        failure_metrics = {"duration_ms": round((time.perf_counter() - started) * 1000, 2), "error_count": 1}
        await _finalize_task_run(
            task=task, run=run, recorder=trace, sink=runtime_event_sink,
            run_status=AgentRunStatus.FAILED.value, task_status=AgentTaskStatus.FAILED.value,
            metrics=failure_metrics, result={"agent_error": terminal_error}, error=terminal_error,
            reason=str(terminal_error.get("code") or "deep_research_execution_failed"),
        )
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        await tasks.release_task_lease(task.id, worker_id, lease_seconds=LEASE_SECONDS)


async def run_task_worker(
    *,
    once: bool = False,
    poll_seconds: float = 1.0,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Claim and execute durable tasks until a cooperative shutdown is requested."""
    shutdown = stop_event or asyncio.Event()
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    await run_task_maintenance()
    next_maintenance = time.monotonic() + MAINTENANCE_INTERVAL_SECONDS
    while True:
        if shutdown.is_set():
            return
        task = await tasks.claim_next_task(worker_id, lease_seconds=LEASE_SECONDS)
        if task is not None:
            try:
                run = await tasks.get_task_run(task.id)
                framework = str(getattr(run, "framework", "") or "").strip() if run is not None else ""
                builder_id = str(getattr(run, "builder_id", "") or "").strip() if run is not None else ""
                if (
                    run is None
                    or str(getattr(task, "active_run_id", "") or "") != str(run.id)
                    or str(getattr(run, "task_id", "") or "") != str(task.id)
                    or not framework
                    or not builder_id
                ):
                    logger.error(
                        "Claimed task has invalid persisted runtime identity | task_id=%s active_run_id=%s",
                        task.id,
                        getattr(task, "active_run_id", None),
                    )
                    await tasks.complete_task(
                        task.id,
                        status=AgentTaskStatus.FAILED.value,
                        reason="task_runtime_identity_invalid",
                    )
                    await tasks.release_task_lease(task.id, worker_id)
                    continue
                wake_limit = positive_float_value(
                    ((task.config_json or {}).get("limits") or {}).get("wake_limit_seconds"),
                    name="wake_limit_seconds",
                )
                await asyncio.wait_for(execute_claimed_task(task.id, worker_id), timeout=wake_limit)
            except asyncio.TimeoutError:
                if await tasks.active_runtime_budget_exhausted(task.id):
                    await tasks.complete_task(
                        task.id,
                        status=AgentTaskStatus.FAILED.value,
                        reason="active_runtime_budget_exhausted",
                    )
                else:
                    await tasks.requeue_after_wake(task.id, reason="active_runtime_wake_limit")
            except Exception:
                logger.exception("Task runner failed before task execution could be contained | task_id=%s", task.id)
                with suppress(Exception):
                    await tasks.complete_task(
                        task.id,
                        status=AgentTaskStatus.FAILED.value,
                        reason="deep_research_runner_failed",
                    )
                with suppress(Exception):
                    await tasks.release_task_lease(task.id, worker_id)
        elif once:
            return
        else:
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=max(0.2, poll_seconds))
                return
            except asyncio.TimeoutError:
                pass
        if time.monotonic() >= next_maintenance:
            with suppress(Exception):
                await run_task_maintenance()
            next_maintenance = time.monotonic() + MAINTENANCE_INTERVAL_SECONDS
