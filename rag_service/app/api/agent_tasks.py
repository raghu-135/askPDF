from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from fastapi.responses import Response, StreamingResponse

from app.agent.tool_registry import TOOL_LIVE_WEB_RECON
from app.agent_workflows.repository import AgentWorkflowRepository
from app.agent_workflows.service import AgentRunService
from app.db import get_thread
from app.models.deep_research import (
    AgentTaskCommandRequest,
    AgentTaskCreateRequest,
    DEEP_RESEARCH_ENGINE_WORKFLOWS,
    DEEP_RESEARCH_WORKFLOW_ID,
)
from app.services import agent_task_repository as repository
from app.services.agent_task_runtime import (
    ensure_task_run,
)
from app.services.content_store import get_content_store
from app.services.agent_task_presentation import plan_diff, timeline_sources
from app.services.task_artifact_service import cleanup_deleted_task
from app.time_utils import maybe_iso_utc_z
from app.runtime.hermes_config import (
    HermesConfigurationError,
    hermes_model_context_length,
    hermes_runtime_enabled,
    validate_hermes_model_compatibility,
)


router = APIRouter(tags=["agent-tasks"])
logger = logging.getLogger(__name__)


def _hermes_available() -> bool:
    if not hermes_runtime_enabled() or len(os.getenv("HERMES_MCP_CONTEXT_SECRET", "")) < 32:
        return False
    try:
        validate_hermes_model_compatibility()
        return True
    except HermesConfigurationError:
        return False


async def _deep_research_contract() -> dict[str, Any]:
    workflow_repository = AgentWorkflowRepository()
    workflow = await workflow_repository.get_workflow(DEEP_RESEARCH_WORKFLOW_ID, include_custom=False)
    if workflow is None:
        await workflow_repository.seed_builtin_workflows()
        workflow = await workflow_repository.get_workflow(DEEP_RESEARCH_WORKFLOW_ID, include_custom=False)
    if workflow is None:
        raise HTTPException(status_code=503, detail={"code": "deep_research_workflow_unavailable"})
    spec = workflow.spec_json if isinstance(workflow.spec_json, dict) else {}
    config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    task_policy = config.get("task_policy") if isinstance(config.get("task_policy"), dict) else {}
    limits = task_policy.get("limits") if isinstance(task_policy.get("limits"), dict) else {}
    profiles = task_policy.get("profiles") if isinstance(task_policy.get("profiles"), list) else []
    allowed_tool_ids = config.get("allowed_tool_ids") if isinstance(config.get("allowed_tool_ids"), list) else []
    return {
        "limits": dict(limits),
        "profiles": list(profiles),
        # use_web_search is the workflow default. Capability comes from the
        # immutable profile and tool grants in the built-in specification.
        "web_enabled": "web_researcher" in profiles and TOOL_LIVE_WEB_RECON in allowed_tool_ids,
    }


@router.get("/deep-research/capabilities")
async def get_deep_research_capabilities():
    contract = await _deep_research_contract()
    limits = contract["limits"]
    hermes_enabled = _hermes_available()
    return {
        "enabled": True,
        "web_enabled": contract["web_enabled"],
        "limits": limits,
        "engines": {
            "langgraph": {"enabled": True, "workflow_id": DEEP_RESEARCH_ENGINE_WORKFLOWS["langgraph"]},
            "hermes": {
                "enabled": hermes_enabled,
                "workflow_id": DEEP_RESEARCH_ENGINE_WORKFLOWS["hermes"],
                "max_context_length": _hermes_context_length(required=False),
            },
        },
    }


def _hermes_context_length(*, required: bool) -> int | None:
    try:
        if required:
            return validate_hermes_model_compatibility()[0]
        value = hermes_model_context_length(required=False)
        if value is not None:
            validate_hermes_model_compatibility()
        return value
    except HermesConfigurationError as exc:
        if not required:
            return None
        raise HTTPException(status_code=503, detail={"code": exc.code}) from exc


def _task_payload(task: Any) -> dict[str, Any]:
    return {
        "id": task.id,
        "thread_id": task.thread_id,
        "project_id": task.project_id,
        "workflow_id": task.workflow_id,
        "objective": task.objective,
        "status": task.status,
        "version": task.version,
        "primary_run_id": task.primary_run_id,
        "active_run_id": task.active_run_id,
        "run_attempt": task.latest_run_attempt,
        "progress": task.progress,
        "completed_todos": task.completed_todos,
        "total_todos": task.total_todos,
        "current_phase": task.current_phase,
        "terminal_reason": task.terminal_reason,
        "budgets": dict(task.budgets_json or {}),
        "configuration": dict(task.config_json or {}),
        "created_at": maybe_iso_utc_z(task.created_at),
        "updated_at": maybe_iso_utc_z(task.updated_at),
        "started_at": maybe_iso_utc_z(task.started_at),
        "paused_at": maybe_iso_utc_z(task.paused_at),
        "completed_at": maybe_iso_utc_z(task.completed_at),
        "expires_at": maybe_iso_utc_z(task.expires_at),
    }


def _todo_payload(todo: Any) -> dict[str, Any]:
    return {
        "id": todo.id, "task_id": todo.task_id, "title": todo.title,
        "description": todo.description, "completion_criteria": todo.completion_criteria,
        "status": todo.status, "priority": todo.priority, "required": todo.required,
        "dependency_ids": list(todo.dependency_ids_json or []), "profile_id": todo.profile_id,
        "attempt": todo.attempt, "max_attempts": todo.max_attempts, "progress": todo.progress,
        "version": todo.version, "result_summary": todo.result_summary,
        "terminal_reason": todo.terminal_reason, "artifact_ids": list(todo.artifact_ids_json or []),
        "created_revision": todo.created_revision, "updated_revision": todo.updated_revision,
    }


def _artifact_payload(artifact: Any) -> dict[str, Any]:
    return {
        "id": artifact.id, "task_id": artifact.task_id, "run_id": artifact.agent_run_id, "todo_id": artifact.todo_id,
        "subagent_run_id": artifact.subagent_run_id, "kind": artifact.kind,
        "media_type": artifact.media_type, "byte_size": artifact.byte_size,
        "sha256": artifact.sha256, "version": artifact.version,
        "provenance": dict(artifact.provenance_json or {}),
        "source_refs": dict(artifact.source_refs_json or {}),
        "summary": dict(artifact.summary_json or {}), "validity": artifact.validity,
        "sensitivity": artifact.sensitivity, "created_at": maybe_iso_utc_z(artifact.created_at),
    }


def _run_payload(run: Any) -> dict[str, Any]:
    debug = (
        {key: value for key, value in run.debug_trace_json.items() if key != "details"}
        if isinstance(run.debug_trace_json, dict)
        else None
    )
    return {
        "id": run.id,
        "task_id": run.task_id,
        "attempt": run.task_attempt,
        "parent_run_id": run.parent_run_id,
        "status": run.status,
        "checkpoint_thread_id": run.checkpoint_thread_id,
        "pending_interrupt": dict(run.pending_interrupt_json or {}) or None,
        "metrics": dict(run.metrics_json or {}),
        "error": dict(run.error_json or {}) or None,
        "debug": debug,
        "started_at": maybe_iso_utc_z(run.started_at),
        "completed_at": maybe_iso_utc_z(run.completed_at),
    }


def _subagent_timeline_type(status: str) -> Optional[str]:
    if status == "completed":
        return "todo_result"
    if status in {"failed", "timed_out", "cancelled"}:
        return "todo_failure"
    return None


async def _owned_task(task_id: str, thread_id: str, *, include_deleted: bool = False):
    task = await repository.get_task(task_id, thread_id=thread_id, include_deleted=include_deleted)
    if task is None or await get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail={"code": "agent_task_not_found"})
    return task


def _conflict(exc: repository.AgentTaskConflict) -> HTTPException:
    return HTTPException(status_code=404 if exc.code == "task_not_found" else 409, detail={
        "code": exc.code, "message": str(exc), "current_version": exc.current_version,
    })


@router.post("/threads/{thread_id}/agent-tasks", status_code=201)
async def create_agent_task(
    thread_id: str,
    req: AgentTaskCreateRequest,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=200),
):
    thread = await get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail={"code": "thread_not_found"})
    contract = await _deep_research_contract()
    workflow_id = DEEP_RESEARCH_ENGINE_WORKFLOWS[req.engine]
    hermes_context_length: int | None = None
    if req.engine == "hermes":
        if not _hermes_available():
            raise HTTPException(status_code=409, detail={"code": "hermes_runtime_unavailable"})
        hermes_context_length = _hermes_context_length(required=True)
        if "context_window" in req.model_fields_set and req.context_window != hermes_context_length:
            raise HTTPException(status_code=409, detail={
                "code": "hermes_context_length_conflict",
                "configured_context_length": hermes_context_length,
            })
    if req.web_search_mode != "off" and not contract["web_enabled"]:
        raise HTTPException(status_code=409, detail={"code": "deep_research_web_unavailable"})
    config = req.model_dump(mode="json")
    if hermes_context_length is not None:
        # Hermes uses one deployment-owned context window. Never allow a
        # client or stale UI capability response to select a per-task value.
        config["context_window"] = hermes_context_length
    config["use_web_search"] = req.web_search_mode != "off"
    contract_limits = contract["limits"]
    config["limits"]["max_concurrency"] = min(
        config["limits"]["max_concurrency"], int(contract_limits["max_concurrency"]),
    )
    config["limits"]["max_fanout"] = min(
        config["limits"]["max_fanout"], int(contract_limits["max_fanout"]),
    )
    task, duplicate = await repository.create_task(
        thread_id=thread_id,
        project_id=thread.project_id,
        user_id=None,
        workflow_id=workflow_id,
        objective=req.objective,
        idempotency_key=idempotency_key,
        config=config,
    )
    return {"task": _task_payload(task), "duplicate": duplicate}


@router.get("/threads/{thread_id}/agent-tasks")
async def list_agent_tasks(thread_id: str, limit: int = Query(default=50, ge=1, le=100)):
    if await get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail={"code": "thread_not_found"})
    return {"tasks": [_task_payload(task) for task in await repository.list_tasks(thread_id, limit=limit)]}


@router.get("/agent-tasks/{task_id}")
async def get_agent_task(task_id: str, thread_id: str = Query(min_length=1)):
    task = await _owned_task(task_id, thread_id)
    run = await AgentWorkflowRepository().get_run(task.active_run_id) if task.active_run_id else None
    plan = await repository.get_latest_plan(task.id, agent_run_id=task.active_run_id)
    payload = _task_payload(task)
    payload["web_access"] = await repository.get_task_web_access(task.id)
    payload["active_run"] = None if run is None else {
        "id": run.id, "status": run.status, "checkpoint_thread_id": run.checkpoint_thread_id,
        "pending_interrupt": dict(run.pending_interrupt_json or {}) or None,
    }
    payload["plan"] = None if plan is None else {
        "revision": plan.revision, "reason": plan.reason, "objective": plan.objective,
        "completion_criteria": list(plan.completion_criteria_json or []),
        "ordered_todo_ids": list(plan.ordered_todo_ids_json or []), "content_hash": plan.content_hash,
    }
    return {"task": payload}


@router.post("/agent-tasks/{task_id}/{action}")
async def command_agent_task(
    task_id: str,
    action: str,
    req: AgentTaskCommandRequest,
    thread_id: str = Query(min_length=1),
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=200),
):
    if action not in {"start", "pause", "resume", "cancel", "retry"}:
        raise HTTPException(status_code=404, detail={"code": "task_command_unknown"})
    task = await _owned_task(task_id, thread_id)
    try:
        task, command, duplicate = await repository.apply_command(
            task.id, action=action, idempotency_key=idempotency_key,
            expected_version=req.expected_version,
        )
        if action in {"start", "retry"} and not duplicate:
            await ensure_task_run(task.id)
            task = await repository.get_task(task.id) or task
        if action == "resume" and not duplicate:
            run = await repository.get_task_run(task.id)
            pending = dict(run.pending_interrupt_json or {}) if run is not None else {}
            if pending.get("status") == "pending" and pending.get("type") == "task_pause":
                await AgentWorkflowRepository().resolve_pending_interrupt(
                    run.id,
                    interrupt_id=str(pending.get("interrupt_id")),
                    action="approve",
                    resume_token=pending.get("resume_token"),
                    resume_version=int(pending.get("resume_version") or 1),
                    expected_thread_id=thread_id,
                )
        if action == "cancel" and not duplicate:
            run = await repository.get_task_run(task.id)
            if run is not None:
                if getattr(run, "framework", None) == "hermes":
                    try:
                        await AgentRunService().cancel_agent_run(run.id, thread_id=thread_id)
                    except Exception as exc:
                        if getattr(exc, "code", None) != "runtime_binding_missing":
                            logger.warning("Hermes /stop failed for task %s: %s", task.id, exc)
                pending = dict(run.pending_interrupt_json or {})
                if pending.get("status") == "pending" and "reject" in (pending.get("allowed_actions") or []):
                    await AgentWorkflowRepository().resolve_pending_interrupt(
                        run.id,
                        interrupt_id=str(pending.get("interrupt_id")),
                        action="reject",
                        resume_token=pending.get("resume_token"),
                        resume_version=int(pending.get("resume_version") or 1),
                        expected_thread_id=thread_id,
                    )
                if task.status == "cancelled":
                    await AgentWorkflowRepository().complete_run(
                        run.id,
                        status="cancelled",
                        error_json={"code": "agent_task_cancelled", "retryable": False},
                    )
        return {"task": _task_payload(task), "command_id": command.id, "duplicate": duplicate}
    except repository.AgentTaskConflict as exc:
        raise _conflict(exc) from exc
    except Exception as exc:
        logger.exception(
            "Failed to initialize agent task run",
            extra={"agent_task_id": task.id, "task_action": action},
        )
        if action in {"start", "retry"}:
            await repository.complete_task(
                task.id,
                status="failed",
                reason="task_run_initialization_failed",
            )
        raise HTTPException(
            status_code=500,
            detail={"code": "task_run_initialization_failed"},
        ) from exc


@router.get("/agent-tasks/{task_id}/todos")
async def get_agent_task_todos(task_id: str, thread_id: str = Query(min_length=1)):
    await _owned_task(task_id, thread_id)
    return {"todos": [_todo_payload(todo) for todo in await repository.list_todos(task_id)]}


@router.get("/agent-tasks/{task_id}/runs")
async def get_agent_task_runs(task_id: str, thread_id: str = Query(min_length=1)):
    await _owned_task(task_id, thread_id)
    return {"runs": [_run_payload(run) for run in await repository.list_task_runs(task_id)]}


@router.get("/agent-tasks/{task_id}/artifacts")
async def get_agent_task_artifacts(
    task_id: str,
    thread_id: str = Query(min_length=1),
    run_id: Optional[str] = Query(default=None),
):
    await _owned_task(task_id, thread_id)
    return {"artifacts": [_artifact_payload(item) for item in await repository.list_artifacts(task_id, agent_run_id=run_id)]}


@router.get("/agent-tasks/{task_id}/artifacts/{artifact_id}")
async def get_agent_task_artifact(task_id: str, artifact_id: str, thread_id: str = Query(min_length=1)):
    await _owned_task(task_id, thread_id)
    artifact = await repository.get_artifact(task_id, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail={"code": "agent_task_artifact_not_found"})
    return {"artifact": _artifact_payload(artifact)}


@router.get("/agent-tasks/{task_id}/artifacts/{artifact_id}/download")
async def download_agent_task_artifact(task_id: str, artifact_id: str, thread_id: str = Query(min_length=1)):
    await _owned_task(task_id, thread_id)
    artifact = await repository.get_artifact(task_id, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail={"code": "agent_task_artifact_not_found"})
    body = await get_content_store().read(artifact.object_key)
    extension = ".md" if artifact.media_type in {"text/markdown", "text/plain"} else ""
    filename = f"{artifact.kind}-{artifact.id}{extension}"
    return Response(
        content=body,
        media_type=artifact.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/agent-tasks/{task_id}/subagents")
async def get_agent_task_subagents(
    task_id: str,
    thread_id: str = Query(min_length=1),
    run_id: Optional[str] = Query(default=None),
):
    await _owned_task(task_id, thread_id)
    rows = await repository.list_subagent_runs(task_id, agent_run_id=run_id)
    return {"subagents": [{
        "id": row.id, "todo_id": row.todo_id, "profile_id": row.profile_id,
        "attempt": row.attempt, "status": row.status, "timeout_ms": row.timeout_ms,
        "usage": dict(row.usage_json or {}), "error": dict(row.error_json or {}) or None,
        "output_artifact_ids": list(row.output_artifact_ids_json or []),
        "started_at": maybe_iso_utc_z(row.started_at), "completed_at": maybe_iso_utc_z(row.completed_at),
    } for row in rows]}


@router.get("/agent-tasks/{task_id}/runs/{run_id}/timeline")
async def get_agent_task_timeline(task_id: str, run_id: str, thread_id: str = Query(min_length=1)):
    task = await _owned_task(task_id, thread_id)
    runs = await repository.list_task_runs(task_id)
    run = next((value for value in runs if value.id == run_id), None)
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "agent_task_run_not_found"})
    plans = await repository.list_plans(task_id, agent_run_id=run_id)
    todos = {todo.id: todo for todo in await repository.list_todos(task_id)}
    subagents = await repository.list_subagent_runs(task_id, agent_run_id=run_id)
    all_artifacts = await repository.list_artifacts(task_id)
    artifacts = [artifact for artifact in all_artifacts if artifact.agent_run_id == run_id]
    attempts_by_run = {str(value.id): int(value.task_attempt or 0) for value in runs}
    events = await repository.list_events(task_id, agent_run_id=run_id, limit=1000)
    items: list[dict[str, Any]] = [{
        "id": f"objective:{run_id}", "type": "objective", "status": run.status,
        "primary_content": task.objective, "timestamp": maybe_iso_utc_z(run.started_at),
        "folds": {"technical_details": {"attempt": run.task_attempt}}, "trace_anchor": None,
    }]
    previous_plan_json: dict[str, Any] = {}
    for index, plan in enumerate(plans):
        plan_json = dict(plan.plan_json or {})
        plan_todos = plan_json.get("todos") if isinstance(plan_json.get("todos"), list) else []
        plan_lines = [str(plan.objective)]
        for position, todo_value in enumerate(plan_todos, start=1):
            if not isinstance(todo_value, dict):
                continue
            dependency_ids = [str(value) for value in todo_value.get("dependency_ids") or []]
            dependency_suffix = f" (after {', '.join(dependency_ids)})" if dependency_ids else ""
            plan_lines.append(f"{position}. {str(todo_value.get('title') or todo_value.get('id') or 'Research step')}{dependency_suffix}")
        revision_diff = plan_diff(previous_plan_json, plan_json, reason=str(plan.reason or "bounded_replan")) if index else None
        if revision_diff is not None:
            changes = [f"Replanned: {revision_diff['reason'].replace('_', ' ')}."]
            changes.extend(f"Added: {value.get('title') or value['id']}" for value in revision_diff["added"])
            changes.extend(f"Removed: {value.get('title') or value['id']}" for value in revision_diff["removed"])
            changes.extend(f"Updated: {value.get('title') or value['id']} ({', '.join(value['fields'])})" for value in revision_diff["changed"])
            if revision_diff["reordered"]:
                changes.append("Reordered the research steps.")
            primary_content = "\n".join(changes) if len(changes) > 1 else f"{changes[0]} No material plan changes were required."
        else:
            primary_content = "\n".join(plan_lines)
        items.append({
            "id": f"plan:{plan.id}", "type": "plan" if index == 0 else "replan", "status": "completed",
            "primary_content": primary_content,
            "timestamp": maybe_iso_utc_z(plan.created_at),
            "folds": ({
                "research_plan": {"objective": plan.objective, "steps": plan_todos},
                "definition_of_done": list(plan.completion_criteria_json or []),
            } if index == 0 else {
                "replan_changes": revision_diff,
                "revised_plan": {"objective": plan.objective, "steps": plan_todos},
                "definition_of_done": list(plan.completion_criteria_json or []),
            }),
            "trace_anchor": {"node_type": "deep_task_planner", "revision": plan.revision},
        })
        previous_plan_json = plan_json
    artifact_by_id = {artifact.id: artifact for artifact in artifacts}
    for attempt in subagents:
        timeline_type = _subagent_timeline_type(attempt.status)
        if timeline_type is None:
            continue
        todo = todos.get(attempt.todo_id)
        attempt_artifacts = [artifact_by_id[value] for value in attempt.output_artifact_ids_json or [] if value in artifact_by_id]
        report_artifact = next((value for value in attempt_artifacts if value.kind == "intermediate_report"), None)
        report_content = None
        if report_artifact is not None:
            try:
                report_content = (await get_content_store().read(report_artifact.object_key)).decode("utf-8", errors="replace")[:200_000]
            except FileNotFoundError:
                # Preserve the timeline when an artifact predates shared
                # content-volume mounting or was cleaned up independently.
                report_content = None
        items.append({
            "id": f"subagent:{attempt.id}",
            "type": timeline_type,
            "status": attempt.status,
            "primary_content": report_content or (todo.result_summary if todo else None) or (todo.title if todo else attempt.todo_id),
            "timestamp": maybe_iso_utc_z(attempt.completed_at or attempt.started_at or attempt.created_at),
            "folds": {
                "attempts": {"attempt": attempt.attempt, "profile_id": attempt.profile_id, "error": attempt.error_json},
            },
            "artifacts": [_artifact_payload(value) for value in attempt_artifacts],
            "sources": timeline_sources(attempt_artifacts, attempts_by_run=attempts_by_run, selected_run_id=run_id),
            "artifact_ids": list(attempt.output_artifact_ids_json or []),
            "trace_anchor": {"node_type": "deep_research_subagent", "subagent_run_id": attempt.id},
        })
    approval_items: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.event_type not in {"task.approval_requested", "task.approval_resolved"}:
            continue
        payload = dict(event.payload_json or {})
        interrupt_id = str(payload.get("interrupt_id") or event.id)
        if event.event_type == "task.approval_requested":
            approval_items[interrupt_id] = {
                "id": f"approval:{interrupt_id}", "type": "approval", "status": "pending",
                "primary_content": str(payload.get("title") or "Human approval"),
                "timestamp": maybe_iso_utc_z(event.created_at), "folds": {"technical_details": payload},
                "trace_anchor": {"interrupt_id": interrupt_id},
            }
        elif interrupt_id in approval_items:
            approval_items[interrupt_id]["status"] = str(payload.get("action") or "resolved")
            approval_items[interrupt_id]["folds"]["decision"] = payload
    items.extend(approval_items.values())
    final_report = next((artifact for artifact in artifacts if artifact.kind == "final_report" and artifact.validity == "valid"), None)
    if final_report is not None:
        final_provenance = dict(final_report.provenance_json or {})
        evidence_manifest = [value for value in final_provenance.get("evidence_manifest") or [] if isinstance(value, dict)]
        evidence_ids = {str(value.get("id") or "") for value in evidence_manifest}
        evidence_artifacts = [artifact for artifact in all_artifacts if artifact.id in evidence_ids and artifact.validity == "valid"]
        try:
            final_report_content = (await get_content_store().read(final_report.object_key)).decode("utf-8", errors="replace")[:500_000]
        except FileNotFoundError:
            final_report_content = None
        items.append({
            "id": f"final:{final_report.id}", "type": "final_report", "status": "incomplete" if final_provenance.get("incomplete") else run.status,
            "primary_content": final_report_content,
            "timestamp": maybe_iso_utc_z(final_report.created_at),
            "folds": {
                "evidence_gaps": list(final_provenance.get("evidence_gaps") or []),
                "quality_review": dict(final_provenance.get("quality_review") or {}),
            },
            "artifacts": [_artifact_payload(final_report)],
            "sources": timeline_sources(evidence_artifacts, attempts_by_run=attempts_by_run, selected_run_id=run_id),
            "evidence_manifest": evidence_manifest,
            "artifact_ids": [final_report.id], "trace_anchor": {"node_type": "finalizer"},
        })
    if run.status == "failed":
        run_error = dict(run.error_json or {})
        safe_message = str(run_error.get("safe_message") or "Deep Research failed before producing a report.")
        items.append({
            "id": f"failure:{run.id}",
            "type": "run_failure",
            "status": "failed",
            "primary_content": safe_message,
            "timestamp": maybe_iso_utc_z(run.completed_at or run.started_at),
            "folds": {
                "technical_details": {
                    "code": run_error.get("code"),
                    "retryable": bool(run_error.get("retryable")),
                    "details": dict(run_error.get("details") or {}),
                },
            },
            "trace_anchor": {"run_id": run.id},
        })
    items.sort(key=lambda item: (item.get("timestamp") or "", item["id"]))
    task_payload = _task_payload(task)
    task_payload["web_access"] = await repository.get_task_web_access(task.id)
    return {"task": task_payload, "run": _run_payload(run), "items": items}


@router.get("/agent-tasks/{task_id}/runs/{run_id}/evidence")
async def get_agent_task_run_evidence(task_id: str, run_id: str, thread_id: str = Query(min_length=1)):
    await _owned_task(task_id, thread_id)
    runs = await repository.list_task_runs(task_id)
    run = next((value for value in runs if value.id == run_id), None)
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "agent_task_run_not_found"})
    artifacts = await repository.list_artifacts(task_id)
    final_report = next((
        artifact for artifact in artifacts
        if artifact.agent_run_id == run_id and artifact.kind == "final_report" and artifact.validity == "valid"
    ), None)
    provenance = dict(final_report.provenance_json or {}) if final_report is not None else {}
    manifest = [value for value in provenance.get("evidence_manifest") or [] if isinstance(value, dict)]
    return {
        "task_id": task_id,
        "run_id": run_id,
        "attempt": run.task_attempt,
        "manifest": manifest,
        "evidence_gaps": [str(value) for value in provenance.get("evidence_gaps") or []],
    }


@router.delete("/agent-tasks/{task_id}")
async def delete_agent_task(
    task_id: str,
    req: AgentTaskCommandRequest,
    background_tasks: BackgroundTasks,
    thread_id: str = Query(min_length=1),
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=200),
):
    task = await _owned_task(task_id, thread_id, include_deleted=True)
    try:
        task, command, duplicate = await repository.request_task_deletion(
            task.id,
            idempotency_key=idempotency_key,
            expected_version=req.expected_version,
        )
    except repository.AgentTaskConflict as exc:
        raise _conflict(exc) from exc
    if not duplicate:
        background_tasks.add_task(cleanup_deleted_task, task.id)
    return {"task_id": task.id, "hidden": True, "command_id": command.id, "duplicate": duplicate}


@router.get("/agent-tasks/{task_id}/events")
async def stream_agent_task_events(
    task_id: str,
    thread_id: str = Query(min_length=1),
    after_sequence: int = Query(default=0, ge=0),
    run_id: Optional[str] = Query(default=None),
    scope: str = Query(default="run", pattern="^(run|task)$"),
):
    task = await _owned_task(task_id, thread_id)
    if scope == "run" and not run_id:
        raise HTTPException(status_code=422, detail={"code": "agent_task_run_id_required"})
    if run_id is not None and not any(run.id == run_id for run in await repository.list_task_runs(task_id)):
        raise HTTPException(status_code=404, detail={"code": "agent_task_run_not_found"})
    active_run_id = str(task.active_run_id or "")

    async def events():
        sequence = after_sequence
        idle = 0
        while True:
            rows = await repository.list_events(task_id, after_sequence=sequence)
            if rows:
                idle = 0
                for row in rows:
                    sequence = row.sequence
                    if scope == "run" and row.agent_run_id != run_id:
                        continue
                    if scope == "task" and row.agent_run_id not in {None, active_run_id}:
                        continue
                    event_payload = row.payload_json if isinstance(row.payload_json, dict) else {}
                    payload = {
                        "id": row.id, "sequence": row.sequence, "type": row.event_type,
                        "task_id": row.task_id, "run_id": row.agent_run_id, "todo_id": row.todo_id,
                        "subagent_run_id": row.subagent_run_id, "artifact_id": row.artifact_id,
                        "payload": event_payload, "created_at": maybe_iso_utc_z(row.created_at),
                    }
                    yield f"id: {sequence}\nevent: task_event\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
            else:
                idle += 1
                if idle >= 12:
                    yield f": heartbeat {sequence}\n\n"
                    idle = 0
            await asyncio.sleep(1)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
