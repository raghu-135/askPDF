from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Mapping, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from app.agent_workflows.repository import AgentWorkflowRepository
from app.agent_workflows.trace_payloads import is_current_debug_payload
from app.agent_workflows.service import AgentRunService
from app.db import get_thread, get_thread_settings
from app.models.deep_research import (
    AgentTaskCommandRequest,
    AgentTaskCreateRequest,
)
from app.services import agent_task_repository as repository
from app.services.agent_task_runtime import (
    ensure_task_run,
)
from app.services.agent_run_cancellation import require_task_cancellation, request_task_cancellation
from app.services.content_store import get_content_store
from app.services.agent_task_presentation import plan_diff, timeline_sources
from app.services.task_artifact_service import cleanup_deleted_task
from app.time_utils import maybe_iso_utc_z
from app.runtime.capability_resolver import resolve_definition_capability_resolution, require_capability
from app.runtime.catalog import definition_from_run, definition_from_workflow
from app.runtime.contracts import RuntimeOperationId
from app.runtime.errors import RuntimeError as AgentRuntimeError
from app.runtime.registry import get_runtime_registry
from app.runtime.builder_registry import builder_for_definition
from app.runtime.operational_limits import required_positive_float


router = APIRouter(tags=["agent-tasks"])
logger = logging.getLogger(__name__)


@router.get("/agent-definitions")
async def list_agent_definitions():
    workflows = await AgentWorkflowRepository().list_workflows(include_custom=False)
    registry = get_runtime_registry()
    entries = []
    for workflow in workflows:
        definition = definition_from_workflow(workflow)
        spec = workflow.spec_json if isinstance(workflow.spec_json, dict) else {}
        config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
        policy = config.get("task_policy") if isinstance(config.get("task_policy"), dict) else {}
        error: dict[str, Any] | None = None
        capabilities = None
        builder = None
        web_enabled = False
        runtime_deployment_id = f"{definition.framework}:{definition.builder_id}"
        try:
            builder = builder_for_definition(definition)
            web_enabled = builder.supports_task_web_search(definition)
            adapter = registry.get(definition)
            runtime_deployment_id = f"{adapter.framework}:{adapter.builder_id}"
            resolution = await resolve_definition_capability_resolution(definition, registry=registry)
            capabilities = resolution.capabilities
            error = resolution.error
        except Exception as exc:
            error = exc.to_dict() if isinstance(exc, AgentRuntimeError) else {
                "code": "runtime_selection_failed",
                "message": str(exc),
                "retryable": False,
            }
        fields: list[Mapping[str, Any]] = []
        if builder is not None:
            try:
                fields = list(builder.task_configuration_fields(definition, spec))
            except Exception as exc:
                logger.warning(
                    "Definition configuration discovery failed | definition_id=%s",
                    definition.definition_id,
                    exc_info=True,
                )
                if error is None:
                    error = exc.to_dict() if isinstance(exc, AgentRuntimeError) else {
                        "code": "definition_configuration_unavailable",
                        "message": str(exc),
                        "retryable": False,
                    }
        entries.append({
            "definition_id": definition.definition_id,
            "runtime_deployment_id": runtime_deployment_id,
            "display_name": definition.display_name,
            "category": definition.category,
            "available": error is None and capabilities is not None,
            "task_eligible": bool(policy),
            "task_start_available": bool(
                capabilities
                and (task_start := capabilities.operations.get(RuntimeOperationId.TASK_START))
                and task_start.enabled
                and task_start.support.value != "unsupported"
            ),
            "configuration": {"fields": fields},
            "operations": {
                operation.value: descriptor.to_dict()
                for operation, descriptor in (capabilities.operations.items() if capabilities else ())
            },
            "features": {
                feature: descriptor.to_dict()
                for feature, descriptor in (capabilities.features.items() if capabilities else ())
            },
            "metadata": definition.definition_metadata,
            "error": error,
        })
    return {"definitions": entries}


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
    retained_debug = run.debug_trace_json if isinstance(run.debug_trace_json, dict) else None
    if retained_debug is not None and not is_current_debug_payload(retained_debug):
        logger.error(
            "Invalid retained debug trace contract | correlation_id=trace:%s version=%r",
            run.id,
            retained_debug.get("version"),
        )
    debug = dict(retained_debug) if retained_debug else None
    return {
        "id": run.id,
        "task_id": run.task_id,
        "attempt": run.task_attempt,
        "parent_run_id": run.parent_run_id,
        "status": run.status,
        "checkpoint_thread_id": run.checkpoint_thread_id,
        "runtime_binding_status": run.runtime_binding_status,
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


async def _require_task_capability(task: Any, action: str) -> None:
    workflow = await AgentWorkflowRepository().get_workflow(task.workflow_id, include_custom=True)
    if workflow is None:
        raise HTTPException(status_code=409, detail={"code": "agent_workflow_unavailable"})
    operation = {
        "start": RuntimeOperationId.TASK_START,
        "pause": RuntimeOperationId.TASK_PAUSE,
        "resume": RuntimeOperationId.TASK_RESUME,
        "cancel": RuntimeOperationId.TASK_CANCEL,
        "retry": RuntimeOperationId.TASK_RETRY,
    }.get(action)
    if operation is None:
        raise HTTPException(status_code=404, detail={"code": "task_command_unknown"})
    run = await repository.get_task_run(task.id)
    definition = definition_from_run(run) if run is not None else definition_from_workflow(workflow)
    try:
        await require_capability(
            definition,
            operation,
            registry=get_runtime_registry(),
            run=run,
            task=task,
        )
    except AgentRuntimeError as exc:
        logger.warning(
            "Agent task start admission rejected | task_id=%s workflow=%s code=%s details=%s",
            task.id,
            task.workflow_id,
            exc.code,
            dict(exc.details or {}),
        )
        raise HTTPException(status_code=409, detail=exc.to_dict()) from exc


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
    workflow = await AgentWorkflowRepository().get_workflow(req.definition_id, include_custom=False)
    if workflow is None:
        raise HTTPException(status_code=404, detail={"code": "agent_definition_not_found"})
    definition = definition_from_workflow(workflow)
    spec = workflow.spec_json if isinstance(workflow.spec_json, dict) else {}
    config_spec = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    task_policy = config_spec.get("task_policy") if isinstance(config_spec.get("task_policy"), dict) else {}
    if not task_policy:
        raise HTTPException(status_code=409, detail={"code": "agent_definition_not_task_eligible"})
    try:
        builder = builder_for_definition(definition)
        if req.web_search_mode != "off" and not builder.supports_task_web_search(definition):
            raise AgentRuntimeError(
                "agent_definition_web_unavailable",
                "The selected definition does not support web search.",
            )
        config = req.model_dump(mode="json")
        config.pop("definition_id", None)
        config["use_web_search"] = req.web_search_mode != "off"
        requested_limits = (
            req.limits.model_dump(mode="json")
            if req.limits is not None
            else dict(task_policy.get("limits") or {})
        )
        config["limits"] = dict(builder.normalize_task_limits(requested_limits))
        resolved = await builder.resolve(
            definition,
            spec,
            thread_settings=await get_thread_settings(thread_id),
            request_overrides={
                "llm_model": req.llm_model,
                "context_window": req.context_window,
                "use_web_search": req.web_search_mode != "off",
            },
        )
    except (AgentRuntimeError, ValueError) as exc:
        detail = exc.to_dict() if isinstance(exc, AgentRuntimeError) else {
            "code": "agent_definition_configuration_invalid",
            "message": str(exc),
            "retryable": False,
        }
        raise HTTPException(status_code=409, detail=detail) from exc
    resolved_config = resolved.get("config") if isinstance(resolved.get("config"), dict) else {}
    config["context_window"] = int(resolved_config.get("context_window") or req.context_window)
    try:
        await require_capability(definition, RuntimeOperationId.TASK_START, registry=get_runtime_registry())
    except AgentRuntimeError as exc:
        raise HTTPException(status_code=409, detail=exc.to_dict()) from exc
    task, duplicate = await repository.create_task(
        thread_id=thread_id,
        project_id=thread.project_id,
        user_id=None,
        workflow_id=definition.definition_id,
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
        "runtime_binding_status": run.runtime_binding_status,
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
    if bool((getattr(task, "config_json", None) or {}).get("workflow_contract_invalidated")):
        raise HTTPException(
            status_code=409,
            detail={"code": "workflow_contract_invalidated", "retryable": False},
        )
    await _require_task_capability(task, action)
    existing_run = await repository.get_task_run(task.id) if action == "cancel" else None
    if (
        action == "cancel"
        and existing_run is not None
        and (existing_run.run_metadata_json or {}).get("runtime_started") is True
    ):
        try:
            await require_task_cancellation(task, existing_run)
        except AgentRuntimeError as exc:
            raise HTTPException(status_code=409, detail=exc.to_dict()) from exc
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
            run = existing_run or await repository.get_task_run(task.id)
            if run is not None and task.status == "cancelling":
                try:
                    result = await request_task_cancellation(task, run)
                except AgentRuntimeError as exc:
                    await repository.complete_control_command(
                        command.id,
                        result={"error": exc.to_dict(), "runtime_confirmation": "pending"},
                        rejected=True,
                    )
                    raise
                await repository.complete_control_command(command.id, result=dict(result))
                task = await repository.get_task(task.id) or task
        return {"task": _task_payload(task), "command_id": command.id, "duplicate": duplicate}
    except repository.AgentTaskConflict as exc:
        raise _conflict(exc) from exc
    except AgentRuntimeError as exc:
        logger.warning("Runtime cancellation submission failed | task_id=%s code=%s", task.id, exc.code)
        raise HTTPException(status_code=503 if exc.retryable else 409, detail=exc.to_dict()) from exc
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
    request: Request,
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
        poll_interval = required_positive_float("AGENT_EVENT_POLL_INTERVAL_SECONDS")
        heartbeat_interval = required_positive_float("AGENT_SSE_HEARTBEAT_INTERVAL_SECONDS")
        idle_seconds = 0.0
        while True:
            if await request.is_disconnected():
                return
            rows = await repository.list_events(task_id, after_sequence=sequence)
            if rows:
                idle_seconds = 0.0
                for row in rows:
                    if await request.is_disconnected():
                        return
                    sequence = row.sequence
                    if scope == "run" and row.agent_run_id != run_id:
                        continue
                    if scope == "task" and row.agent_run_id not in {None, active_run_id}:
                        continue
                    event_payload = row.payload_json if isinstance(row.payload_json, dict) else {}
                    event_id = getattr(row, "event_id", None)
                    occurred_at = getattr(row, "occurred_at", None)
                    terminal = bool(getattr(row, "terminal", False))
                    source_metadata = getattr(row, "source_metadata_json", {})
                    payload = {
                        "id": row.id, "event_id": event_id, "sequence": row.sequence, "type": row.event_type,
                        "task_id": row.task_id, "run_id": row.agent_run_id, "todo_id": row.todo_id,
                        "subagent_run_id": row.subagent_run_id, "artifact_id": row.artifact_id,
                        "payload": event_payload, "created_at": maybe_iso_utc_z(row.created_at),
                        "occurred_at": maybe_iso_utc_z(occurred_at), "terminal": bool(terminal),
                        "source_metadata": dict(source_metadata or {}),
                    }
                    yield f"id: {sequence}\nevent: task_event\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
                    if terminal:
                        return
            else:
                idle_seconds += poll_interval
                if idle_seconds >= heartbeat_interval:
                    if await request.is_disconnected():
                        return
                    yield f": heartbeat {sequence}\n\n"
                    idle_seconds = 0.0
            await asyncio.sleep(poll_interval)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
