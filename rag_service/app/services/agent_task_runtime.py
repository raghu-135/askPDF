from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Optional

from app.agent_workflows.checkpointing import open_agent_checkpointer
from app.agent_workflows.compiler import WorkflowCompiler
from app.agent_workflows.debug_trace import AgentTraceRecorder, finalize_and_merge_debug_payload
from app.agent_workflows.repository import AgentWorkflowRepository
from app.agent_workflows.router_runtime import (
    continue_compiled_rag_chat,
    execute_compiled_rag_chat,
    project_agent_task_result,
    resume_compiled_rag_chat,
)
from app.agent_workflows.validator import WorkflowResolver
from app.db import AgentRunStatus, get_thread, get_thread_settings
from app.models.deep_research import AgentTaskStatus, DEEP_RESEARCH_WORKFLOW_ID
from app.services import agent_task_repository as tasks
from app.services.task_artifact_service import persist_task_artifact
from app.services.agent_task_maintenance import MAINTENANCE_INTERVAL_SECONDS, run_task_maintenance


logger = logging.getLogger(__name__)
LEASE_SECONDS = 60
HEARTBEAT_SECONDS = 15
WAKE_RUNTIME_LIMIT_SECONDS = 15 * 60


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
    """Complete one AgentRun, then durably persist its merged trace payload."""

    completed_run = await repository.complete_run(
        run.id,
        status=status,
        metrics_json=metrics,
        error_json=error,
    )
    if completed_run is None:
        return None
    debug_payload = finalize_and_merge_debug_payload(
        recorder=recorder,
        run=completed_run,
        metrics=metrics,
        result=result,
        route=result.get("route"),
        route_reason=result.get("route_reason"),
        error=error,
        run_status=status,
        completed_at=completed_run.completed_at,
    )
    await repository.set_run_debug_trace(run.id, debug_payload)
    return completed_run


async def ensure_task_run(task_id: str):
    task = await tasks.get_task(task_id)
    if task is None:
        raise ValueError("task_not_found")
    active = await tasks.get_task_run(task_id)
    if active is not None and active.status in {AgentRunStatus.RUNNING.value, AgentRunStatus.AWAITING_HUMAN.value}:
        return active

    repository = AgentWorkflowRepository()
    workflow = await repository.get_workflow(DEEP_RESEARCH_WORKFLOW_ID, include_custom=False)
    if workflow is None:
        await repository.seed_builtin_workflows()
        workflow = await repository.get_workflow(DEEP_RESEARCH_WORKFLOW_ID, include_custom=False)
    if workflow is None:
        raise RuntimeError("deep_research_workflow_unavailable")

    thread_settings = await get_thread_settings(task.thread_id)
    resolved = WorkflowResolver().resolve(
        workflow.spec_json,
        thread_settings=thread_settings,
        request_overrides={"use_web_search": bool((task.config_json or {}).get("use_web_search"))},
    )
    config = dict(resolved.get("config") or {})
    task_policy = dict(config.get("task_policy") or {})
    task_policy["limits"] = dict((task.config_json or {}).get("limits") or {})
    task_policy["profiles"] = list((task.config_json or {}).get("enabled_profiles") or [])
    config["task_policy"] = task_policy
    config["use_web_search"] = bool((task.config_json or {}).get("use_web_search"))
    resolved["config"] = config
    frozen_spec = WorkflowCompiler().materialize_spec(resolved)
    metadata = dict(getattr(workflow, "metadata_json", None) or {})
    version = int(metadata.get("version") or workflow.schema_version or 1)
    run = await repository.create_run(
        thread_id=task.thread_id,
        workflow_id=workflow.id,
        workflow_version_id=str(metadata.get("version_id") or f"{workflow.id}:v{version}"),
        workflow_version=version,
        resolved_spec_json=frozen_spec,
        user_id=task.user_id,
        run_metadata_json={"executed_workflow_id": workflow.id, "run_kind": "agent_task", "agent_task_id": task.id},
    )
    await tasks.attach_run(task.id, run, parent_run_id=active.id if active is not None else None)
    return run


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
            await AgentWorkflowRepository().complete_run(
                active_run.id,
                status=AgentRunStatus.CANCELLED.value,
                error_json={"code": "agent_task_cancelled", "retryable": False},
            )
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
    todos = await tasks.list_todos(task.id)
    task_web_access = await tasks.get_task_web_access(task.id)
    request = SimpleNamespace(
        question=task.objective,
        llm_model=config.get("llm_model"),
        context_window=config.get("context_window", 32_768),
        use_web_search=bool(config.get("use_web_search")),
        web_search_mode=str(config.get("web_search_mode") or "off"),
        task_web_access=task_web_access,
        use_reranker=True,
        bypass_clarification=True,
        system_role_override="",
        tool_instructions_override={},
        custom_instructions_override="",
        client_timezone=None,
        client_locale=None,
        client_now_iso=None,
        agent_task_id=task.id,
        agent_task_version=task.version,
        task_enabled_profiles=list(config.get("enabled_profiles") or []),
        task_limits=dict(config.get("limits") or {}),
        task_plan_revision=max((todo.updated_revision for todo in todos), default=0),
        task_run_plan_count=0,
        task_todos=[{
            "id": todo.id, "title": todo.title, "description": todo.description,
            "completion_criteria": todo.completion_criteria, "status": todo.status,
            "priority": todo.priority, "required": todo.required,
            "dependency_ids": list(todo.dependency_ids_json or []), "profile_id": todo.profile_id,
            "attempt": todo.attempt, "max_attempts": todo.max_attempts,
            "progress": todo.progress, "result_summary": todo.result_summary,
            "artifact_ids": list(todo.artifact_ids_json or []), "version": todo.version,
        } for todo in todos],
        task_budget_usage=dict(task.budgets_json or {}),
    )
    repository = AgentWorkflowRepository()
    trace = AgentTraceRecorder(run)
    context = {
        "agent_run_id": run.id,
        "agent_workflow_id": run.workflow_id,
        "agent_workflow_version": run.workflow_version,
        "checkpoint_thread_id": run.checkpoint_thread_id,
    }
    heartbeat = asyncio.create_task(_heartbeat(task.id, worker_id))
    started = time.perf_counter()
    async def cancellation_requested() -> bool:
        return await tasks.task_cancel_requested(task.id) or await tasks.active_runtime_budget_exhausted(task.id)

    try:
        async with open_agent_checkpointer() as checkpointer:
            pending = dict(run.pending_interrupt_json or {})
            if pending.get("status") in {"resumed", "resolved"} and isinstance(pending.get("decision"), dict):
                result = await resume_compiled_rag_chat(
                    run,
                    interrupt=pending,
                    checkpointer=checkpointer,
                    trace_recorder=trace,
                    cancellation_checker=cancellation_requested,
                    result_projector=project_agent_task_result,
                )
            else:
                result = await continue_compiled_rag_chat(
                    run,
                    checkpointer=checkpointer,
                    trace_recorder=trace,
                    cancellation_checker=cancellation_requested,
                    result_projector=project_agent_task_result,
                )
                if result is None:
                    result = await execute_compiled_rag_chat(
                        task.thread_id,
                        request,
                        thread.embedding_model,
                        resolved_spec=run.resolved_spec_json,
                        agent_run_context=context,
                        trace_recorder=trace,
                        checkpointer=checkpointer,
                        cancellation_checker=cancellation_requested,
                        result_projector=project_agent_task_result,
                    )
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
            return
        if status == AgentRunStatus.CANCELLED.value:
            latest_task = await tasks.get_task(task.id)
            budget_exhausted = bool(latest_task and latest_task.terminal_reason == "active_runtime_budget_exhausted")
            await _complete_run_with_trace(
                repository,
                run=run,
                recorder=trace,
                status=status,
                metrics=metrics,
                result=result,
            )
            await tasks.complete_task(
                task.id,
                status=AgentTaskStatus.FAILED.value if budget_exhausted else AgentTaskStatus.CANCELLED.value,
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
                await _complete_run_with_trace(
                    repository,
                    run=run,
                    recorder=trace,
                    status=AgentRunStatus.FAILED.value,
                    metrics=metrics,
                    result=result,
                    error=terminal_error,
                )
                await tasks.complete_task(task.id, status=AgentTaskStatus.FAILED.value, reason="final_report_missing")
                return
            evidence_manifest = [
                value for value in result.get("task_evidence_manifest") or []
                if isinstance(value, dict) and value.get("id")
            ]
            incomplete_reasons = [str(value) for value in result.get("task_incomplete_reasons") or []]
            final_artifact = await persist_task_artifact(
                task_id=task.id,
                agent_run_id=run.id,
                kind="final_report",
                content=final_answer,
                provenance={
                    "incomplete": incomplete,
                    "draft_model": result.get("task_draft_metadata") or {},
                    "quality_review": result.get("task_critic_report") or {},
                    "plan_revision": int(result.get("task_plan_revision") or request.task_plan_revision or 0),
                    "evidence_manifest": evidence_manifest,
                    "evidence_gaps": incomplete_reasons,
                },
                source_refs={"artifact_ids": [str(item["id"]) for item in evidence_manifest]},
            )
            await _complete_run_with_trace(
                repository,
                run=run,
                recorder=trace,
                status=status,
                metrics=metrics,
                result=result,
                error=error,
            )
            await tasks.complete_task(
                task.id,
                status=AgentTaskStatus.COMPLETED.value,
                reason="incomplete" if incomplete else "completed",
                final_artifact_id=final_artifact.id,
            )
        else:
            await _complete_run_with_trace(
                repository,
                run=run,
                recorder=trace,
                status=status,
                metrics=metrics,
                result=result,
                error=error,
            )
            await tasks.complete_task(task.id, status=AgentTaskStatus.FAILED.value, reason=str((error or {}).get("code") or status))
    except Exception as exc:
        logger.exception("Deep research task execution failed | task_id=%s run_id=%s", task.id, run.id)
        terminal_error = {
            "code": "deep_research_execution_failed",
            "type": type(exc).__name__,
            "raw_message": str(exc)[:1000],
            "retryable": True,
        }
        await _complete_run_with_trace(
            repository,
            run=run,
            recorder=trace,
            status=AgentRunStatus.FAILED.value,
            metrics={"duration_ms": round((time.perf_counter() - started) * 1000, 2), "error_count": 1},
            result={"agent_error": terminal_error},
            error=terminal_error,
        )
        await tasks.complete_task(task.id, status=AgentTaskStatus.FAILED.value, reason="deep_research_execution_failed")
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
                await asyncio.wait_for(execute_claimed_task(task.id, worker_id), timeout=WAKE_RUNTIME_LIMIT_SECONDS)
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
