from __future__ import annotations

import hashlib
import json
import uuid
from datetime import timedelta
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import AgentRunStatus
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import (
    AgentRun,
    AgentTask,
    AgentTaskArtifact,
    AgentTaskCommand,
    AgentTaskEvent,
    AgentTaskPlanRevision,
    AgentTaskSubagentRun,
    AgentTaskTodo,
)
from app.models.deep_research import AgentTaskStatus, DeepResearchPlanProposal
from app.time_utils import utc_now
from app.agent_workflows.trace_details import sanitize_trace_detail
from app.runtime.contracts import TERMINAL_RUNTIME_EVENT_KINDS
from app.runtime.events import normalize_product_event_kind


ACTIVE_TASK_STATUSES = {
    AgentTaskStatus.QUEUED.value,
    AgentTaskStatus.RUNNING.value,
    AgentTaskStatus.PAUSING.value,
    AgentTaskStatus.PAUSED.value,
    AgentTaskStatus.AWAITING_APPROVAL.value,
    AgentTaskStatus.CANCELLING.value,
}
TERMINAL_TASK_STATUSES = {
    AgentTaskStatus.CANCELLED.value,
    AgentTaskStatus.COMPLETED.value,
    AgentTaskStatus.FAILED.value,
    AgentTaskStatus.EXPIRED.value,
}
WEB_ACCESS_EVENT_PREFIX = "web_access."
WEB_ACCESS_ALLOWED = "allowed_for_task"
WEB_ACCESS_DENIED = "denied_for_task"


class AgentTaskConflict(ValueError):
    def __init__(self, code: str, message: str, *, current_version: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.current_version = current_version


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def create_task(
    *,
    thread_id: str,
    project_id: Optional[str],
    user_id: Optional[str],
    workflow_id: str,
    objective: str,
    idempotency_key: str,
    config: Dict[str, Any],
) -> tuple[AgentTask, bool]:
    objective = " ".join(objective.split()).strip()
    async with async_session_maker() as session:
        existing_query = select(AgentTask).where(
            AgentTask.thread_id == thread_id,
            AgentTask.create_idempotency_key == idempotency_key,
        )
        existing_query = existing_query.where(AgentTask.user_id.is_(None)) if user_id is None else existing_query.where(AgentTask.user_id == user_id)
        existing = (await session.execute(existing_query)).scalar_one_or_none()
        if existing:
            return existing, True
        task = AgentTask(
            thread_id=thread_id,
            project_id=project_id,
            user_id=user_id,
            workflow_id=workflow_id,
            objective=objective,
            objective_hash=hashlib.sha256(objective.casefold().encode("utf-8")).hexdigest(),
            create_idempotency_key=idempotency_key,
            config_json=config,
            budgets_json={
                "model_tokens": 0,
                "model_calls": 0,
                "tool_calls": 0,
                "subagent_attempts": 0,
                "artifact_bytes": 0,
                "elapsed_active_ms": 0,
            },
            expires_at=utc_now() + timedelta(hours=24),
        )
        session.add(task)
        try:
            await session.flush()
            await _append_event(session, task, "task.created", payload={"status": task.status})
            await session.commit()
            await session.refresh(task)
            return task, False
        except IntegrityError:
            await session.rollback()
            existing = (await session.execute(existing_query)).scalar_one_or_none()
            if existing:
                return existing, True
            raise


async def get_task(
    task_id: str,
    *,
    thread_id: Optional[str] = None,
    include_deleted: bool = False,
) -> Optional[AgentTask]:
    async with async_session_maker() as session:
        query = select(AgentTask).where(AgentTask.id == task_id)
        if thread_id is not None:
            query = query.where(AgentTask.thread_id == thread_id)
        if not include_deleted:
            query = query.where(AgentTask.deletion_requested_at.is_(None))
        return (await session.execute(query)).scalar_one_or_none()


async def get_task_run(task_id: str) -> Optional[AgentRun]:
    async with async_session_maker() as session:
        task = await session.get(AgentTask, task_id)
        return await session.get(AgentRun, task.active_run_id) if task and task.active_run_id else None


async def list_task_runs(task_id: str) -> list[AgentRun]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentRun)
            .where(AgentRun.task_id == task_id)
            .order_by(AgentRun.task_attempt, AgentRun.started_at, AgentRun.id)
        )
        return list(result.scalars().all())


async def task_cancel_requested(task_id: str) -> bool:
    task = await get_task(task_id)
    return bool(task and task.status in {AgentTaskStatus.CANCELLING.value, AgentTaskStatus.CANCELLED.value})


async def consume_budget(
    task_id: str,
    *,
    model_calls: int = 0,
    model_tokens: int = 0,
    tool_calls: int = 0,
) -> Dict[str, Any]:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            limits = (task.config_json or {}).get("limits") or {}
            usage = dict(task.budgets_json or {})
            next_model_calls = int(usage.get("model_calls") or 0) + model_calls
            next_model_tokens = int(usage.get("model_tokens") or 0) + model_tokens
            next_tool_calls = int(usage.get("tool_calls") or 0) + tool_calls
            if next_model_calls > int(limits.get("max_model_calls", 10_000)):
                raise AgentTaskConflict("model_call_budget_exhausted", "Task model-call budget exhausted")
            if next_model_tokens > int(limits.get("max_model_tokens", 500_000)):
                raise AgentTaskConflict("model_token_budget_exhausted", "Task model-token budget exhausted")
            if next_tool_calls > int(limits.get("max_tool_calls", 100)):
                raise AgentTaskConflict("tool_call_budget_exhausted", "Task tool-call budget exhausted")
            usage["model_calls"] = next_model_calls
            usage["model_tokens"] = next_model_tokens
            usage["tool_calls"] = next_tool_calls
            replace_jsonb_field(task, "budgets_json", usage)
            task.version += 1
            await _append_event(session, task, "task.budget_updated", agent_run_id=task.active_run_id, payload={
                "model_calls": usage["model_calls"], "model_tokens": usage["model_tokens"],
                "tool_calls": usage["tool_calls"],
            })
        return usage


async def queue_task_after_interrupt(
    task_id: str,
    *,
    reason: str,
    interrupt_id: Optional[str] = None,
    action: Optional[str] = None,
) -> Optional[AgentTask]:
    """Queue the same checkpoint thread after the canonical interrupt resolver succeeds."""
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None:
                return None
            if task.status in TERMINAL_TASK_STATUSES:
                return task
            task.status = AgentTaskStatus.QUEUED.value
            task.current_phase = "continuing"
            task.queued_at = utc_now()
            task.lease_owner = None
            task.lease_expires_at = None
            task.version += 1
            await _append_event(session, task, "task.continuation_queued", agent_run_id=task.active_run_id, payload={"reason": reason, "version": task.version})
            if interrupt_id and action:
                await _append_event(
                    session,
                    task,
                    "task.approval_resolved",
                    agent_run_id=task.active_run_id,
                    payload={"interrupt_id": interrupt_id, "action": action, "version": task.version},
                )
        await session.refresh(task)
        return task


async def requeue_after_wake(task_id: str, *, reason: str) -> Optional[AgentTask]:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None or task.status in TERMINAL_TASK_STATUSES:
                return task
            task.status = AgentTaskStatus.QUEUED.value
            task.current_phase = "continuation_queued"
            task.queued_at = utc_now()
            task.expires_at = utc_now() + timedelta(hours=24)
            task.lease_owner = None
            task.lease_expires_at = None
            task.version += 1
            await _append_event(session, task, "task.wake_budget_reached", agent_run_id=task.active_run_id, payload={
                "reason": reason, "version": task.version,
            })
        return task


async def list_tasks(thread_id: str, *, limit: int = 50) -> list[AgentTask]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTask)
            .where(AgentTask.thread_id == thread_id, AgentTask.deletion_requested_at.is_(None))
            .order_by(AgentTask.created_at.desc(), AgentTask.id.desc())
            .limit(max(1, min(limit, 100)))
        )
        return list(result.scalars().all())


async def list_todos(task_id: str) -> list[AgentTaskTodo]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTaskTodo)
            .where(AgentTaskTodo.task_id == task_id)
            .order_by(AgentTaskTodo.priority.desc(), AgentTaskTodo.created_at, AgentTaskTodo.id)
        )
        return list(result.scalars().all())


async def get_latest_plan(task_id: str, *, agent_run_id: Optional[str] = None) -> Optional[AgentTaskPlanRevision]:
    async with async_session_maker() as session:
        query = select(AgentTaskPlanRevision).where(AgentTaskPlanRevision.task_id == task_id)
        if agent_run_id is not None:
            query = query.where(AgentTaskPlanRevision.agent_run_id == agent_run_id)
        return (await session.execute(query.order_by(AgentTaskPlanRevision.revision.desc()).limit(1))).scalar_one_or_none()


async def list_plans(task_id: str, *, agent_run_id: Optional[str] = None) -> list[AgentTaskPlanRevision]:
    async with async_session_maker() as session:
        query = select(AgentTaskPlanRevision).where(AgentTaskPlanRevision.task_id == task_id)
        if agent_run_id is not None:
            query = query.where(AgentTaskPlanRevision.agent_run_id == agent_run_id)
        result = await session.execute(query.order_by(AgentTaskPlanRevision.revision, AgentTaskPlanRevision.created_at))
        return list(result.scalars().all())


async def list_artifacts(task_id: str, *, agent_run_id: Optional[str] = None) -> list[AgentTaskArtifact]:
    async with async_session_maker() as session:
        query = select(AgentTaskArtifact).where(
            AgentTaskArtifact.task_id == task_id,
            AgentTaskArtifact.validity != "deleted",
        )
        if agent_run_id is not None:
            query = query.where(AgentTaskArtifact.agent_run_id == agent_run_id)
        result = await session.execute(query.order_by(AgentTaskArtifact.created_at, AgentTaskArtifact.id))
        return list(result.scalars().all())


async def get_artifact(task_id: str, artifact_id: str) -> Optional[AgentTaskArtifact]:
    async with async_session_maker() as session:
        return (await session.execute(select(AgentTaskArtifact).where(
            AgentTaskArtifact.task_id == task_id,
            AgentTaskArtifact.id == artifact_id,
            AgentTaskArtifact.validity != "deleted",
        ))).scalar_one_or_none()


async def list_artifacts_for_threads(thread_ids: Iterable[str]) -> list[AgentTaskArtifact]:
    ids = {str(value) for value in thread_ids if value}
    if not ids:
        return []
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTaskArtifact)
            .join(AgentTask, AgentTask.id == AgentTaskArtifact.task_id)
            .where(AgentTask.thread_id.in_(ids), AgentTaskArtifact.validity != "deleted")
        )
        return list(result.scalars().all())


async def list_task_checkpoint_ids_for_threads(thread_ids: Iterable[str]) -> list[str]:
    ids = {str(value) for value in thread_ids if value}
    if not ids:
        return []
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentRun.checkpoint_thread_id)
            .where(AgentRun.thread_id.in_(ids), AgentRun.task_id.is_not(None), AgentRun.framework == "langgraph", AgentRun.checkpoint_thread_id.is_not(None))
        )
        return [str(value) for value in result.scalars().all() if value]


async def list_terminal_task_checkpoint_ids_before(cutoff: Any, *, limit: int = 100) -> list[str]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentRun.checkpoint_thread_id)
            .where(
                AgentRun.task_id.is_not(None),
                AgentRun.framework == "langgraph",
                AgentRun.completed_at.is_not(None),
                AgentRun.completed_at <= cutoff,
                AgentRun.status.in_(["completed", "failed", "expired", "cancelled", "rejected"]),
                AgentRun.checkpoint_thread_id.is_not(None),
            )
            .order_by(AgentRun.completed_at, AgentRun.id)
            .limit(max(1, min(limit, 500)))
        )
        return [str(value) for value in result.scalars().all() if value]


async def clear_task_checkpoint_ids(checkpoint_thread_ids: Iterable[str]) -> int:
    ids = {str(value) for value in checkpoint_thread_ids if value}
    if not ids:
        return 0
    async with async_session_maker() as session:
        async with session.begin():
            rows = list((await session.execute(
                select(AgentRun).where(
                    AgentRun.task_id.is_not(None),
                    AgentRun.checkpoint_thread_id.in_(ids),
                ).with_for_update()
            )).scalars().all())
            for run in rows:
                run.checkpoint_thread_id = None
            return len(rows)


async def release_stale_task_leases(*, limit: int = 100) -> int:
    now = utc_now()
    async with async_session_maker() as session:
        async with session.begin():
            rows = list((await session.execute(
                select(AgentTask)
                .where(
                    AgentTask.lease_owner.is_not(None),
                    AgentTask.lease_expires_at.is_not(None),
                    AgentTask.lease_expires_at < now,
                    AgentTask.status.in_([
                        AgentTaskStatus.RUNNING.value,
                        AgentTaskStatus.CANCELLING.value,
                    ]),
                )
                .order_by(AgentTask.lease_expires_at, AgentTask.id)
                .with_for_update(skip_locked=True)
                .limit(max(1, min(limit, 500)))
            )).scalars().all())
            for task in rows:
                task.lease_owner = None
                task.lease_expires_at = None
                task.updated_at = now
                await _append_event(
                    session,
                    task,
                    "task.lease_recovered",
                    agent_run_id=task.active_run_id,
                    payload={"status": task.status},
                )
            return len(rows)


async def mark_artifact_deleted(task_id: str, artifact_id: str) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            artifact = await session.get(AgentTaskArtifact, artifact_id)
            if task is None or artifact is None or artifact.task_id != task_id or artifact.validity == "deleted":
                return
            artifact.validity = "deleted"
            artifact.deleted_at = utc_now()
            await _append_event(session, task, "artifact.deleted", agent_run_id=artifact.agent_run_id, artifact_id=artifact.id, payload={"sha256": artifact.sha256})


async def mark_artifact_invalid(task_id: str, artifact_id: str, *, reason: str) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            artifact = await session.get(AgentTaskArtifact, artifact_id)
            if task is None or artifact is None or artifact.task_id != task_id or artifact.validity != "valid":
                return
            artifact.validity = "invalid"
            await _append_event(
                session,
                task,
                "artifact.invalidated",
                agent_run_id=artifact.agent_run_id,
                artifact_id=artifact.id,
                payload={"reason": reason, "sha256": artifact.sha256},
            )


async def list_expired_artifacts(*, limit: int = 100) -> list[AgentTaskArtifact]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTaskArtifact)
            .where(
                AgentTaskArtifact.validity != "deleted",
                AgentTaskArtifact.retention_until.is_not(None),
                AgentTaskArtifact.retention_until <= utc_now(),
            )
            .order_by(AgentTaskArtifact.retention_until, AgentTaskArtifact.id)
            .limit(max(1, min(limit, 500)))
        )
        return list(result.scalars().all())


async def list_live_artifacts(*, limit: int = 10_000) -> list[AgentTaskArtifact]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTaskArtifact)
            .where(AgentTaskArtifact.validity != "deleted")
            .order_by(AgentTaskArtifact.created_at, AgentTaskArtifact.id)
            .limit(max(1, min(limit, 10_000)))
        )
        return list(result.scalars().all())


async def invalidate_context_summaries(task_id: str, *, source_hash: str) -> int:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            rows = list((await session.execute(select(AgentTaskArtifact).where(
                AgentTaskArtifact.task_id == task_id,
                AgentTaskArtifact.kind == "context_summary",
                AgentTaskArtifact.validity == "valid",
            ).with_for_update())).scalars().all())
            invalidated = 0
            for artifact in rows:
                if (artifact.provenance_json or {}).get("source_hash") == source_hash:
                    continue
                artifact.validity = "invalid"
                invalidated += 1
                await _append_event(session, task, "artifact.invalidated", agent_run_id=artifact.agent_run_id, artifact_id=artifact.id, payload={
                    "reason": "source_hash_changed", "replacement_source_hash": source_hash,
                })
            return invalidated


async def list_subagent_runs(task_id: str, *, agent_run_id: Optional[str] = None) -> list[AgentTaskSubagentRun]:
    async with async_session_maker() as session:
        query = select(AgentTaskSubagentRun).where(AgentTaskSubagentRun.task_id == task_id)
        if agent_run_id is not None:
            query = query.where(AgentTaskSubagentRun.agent_run_id == agent_run_id)
        result = await session.execute(query.order_by(AgentTaskSubagentRun.created_at, AgentTaskSubagentRun.id))
        return list(result.scalars().all())


async def list_events(
    task_id: str,
    *,
    agent_run_id: Optional[str] = None,
    after_sequence: int = 0,
    limit: int = 500,
) -> list[AgentTaskEvent]:
    async with async_session_maker() as session:
        query = select(AgentTaskEvent).where(
            AgentTaskEvent.task_id == task_id,
            AgentTaskEvent.sequence > max(0, after_sequence),
        )
        if agent_run_id is not None:
            query = query.where(AgentTaskEvent.agent_run_id == agent_run_id)
        result = await session.execute(query.order_by(AgentTaskEvent.sequence).limit(max(1, min(limit, 1000))))
        return list(result.scalars().all())


async def _append_event(
    session,
    task: AgentTask,
    event_type: str,
    *,
    actor_type: str = "system",
    actor_id: Optional[str] = None,
    agent_run_id: Optional[str] = None,
    todo_id: Optional[str] = None,
    subagent_run_id: Optional[str] = None,
    artifact_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    policy_hash: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> AgentTaskEvent:
    latest = await session.execute(
        select(func.coalesce(func.max(AgentTaskEvent.sequence), 0))
        .where(AgentTaskEvent.task_id == task.id)
    )
    normalized_type, source_metadata = normalize_product_event_kind(event_type)
    event_sequence = int(latest.scalar_one()) + 1
    event = AgentTaskEvent(
        task_id=task.id,
        sequence=event_sequence,
        event_id=f"{task.id}:{event_sequence}",
        event_type=normalized_type,
        actor_type=actor_type,
        actor_id=actor_id,
        agent_run_id=agent_run_id,
        todo_id=todo_id,
        subagent_run_id=subagent_run_id,
        artifact_id=artifact_id,
        payload_json=sanitize_trace_detail(payload or {})[0],
        policy_hash=policy_hash,
        config_hash=config_hash,
        occurred_at=utc_now(),
        terminal=normalized_type in TERMINAL_RUNTIME_EVENT_KINDS,
        source_metadata_json=source_metadata,
    )
    session.add(event)
    await session.flush()
    return event


async def append_event(task_id: str, event_type: str, **kwargs) -> AgentTaskEvent:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            event = await _append_event(session, task, event_type, **kwargs)
        await session.refresh(event)
        return event


async def get_task_web_access(task_id: str) -> str:
    async with async_session_maker() as session:
        event = (await session.execute(
            select(AgentTaskEvent)
            .where(
                AgentTaskEvent.task_id == task_id,
                AgentTaskEvent.event_type == "approval.responded",
            )
            .order_by(AgentTaskEvent.sequence.desc())
            .limit(1)
        )).scalar_one_or_none()
        return str((event.payload_json or {}).get("status") or "undecided") if event else "undecided"


async def set_task_web_access(
    task_id: str,
    status: str,
    *,
    agent_run_id: str,
    interrupt_id: str,
    actor_id: Optional[str] = None,
) -> AgentTask:
    if status not in {WEB_ACCESS_ALLOWED, WEB_ACCESS_DENIED}:
        raise ValueError("unknown task web-access status")
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(
                select(AgentTask).where(AgentTask.id == task_id).with_for_update()
            )).scalar_one()
            prior_events = list((await session.execute(
                select(AgentTaskEvent)
                .where(
                    AgentTaskEvent.task_id == task_id,
                    AgentTaskEvent.event_type == "approval.responded",
                )
                .order_by(AgentTaskEvent.sequence.desc())
                .limit(100)
            )).scalars().all())
            if any(
                str((event.payload_json or {}).get("interrupt_id") or "") == interrupt_id
                and str((event.payload_json or {}).get("status") or "") == status
                for event in prior_events
            ):
                return task
            task.version += 1
            task.updated_at = utc_now()
            await _append_event(
                session,
                task,
                f"{WEB_ACCESS_EVENT_PREFIX}{status}",
                actor_type="user",
                actor_id=actor_id,
                agent_run_id=agent_run_id,
                payload={"interrupt_id": interrupt_id, "scope": "task", "status": status, "version": task.version},
            )
        await session.refresh(task)
        return task


COMMAND_TRANSITIONS = {
    "start": ({AgentTaskStatus.CREATED.value}, AgentTaskStatus.QUEUED.value),
    "pause": ({AgentTaskStatus.QUEUED.value, AgentTaskStatus.RUNNING.value}, AgentTaskStatus.PAUSING.value),
    "resume": ({AgentTaskStatus.PAUSED.value}, AgentTaskStatus.QUEUED.value),
    "cancel": ({*ACTIVE_TASK_STATUSES, AgentTaskStatus.CREATED.value}, AgentTaskStatus.CANCELLING.value),
    "retry": ({AgentTaskStatus.FAILED.value, AgentTaskStatus.EXPIRED.value}, AgentTaskStatus.QUEUED.value),
    "expire": ({AgentTaskStatus.PAUSED.value, AgentTaskStatus.AWAITING_APPROVAL.value}, AgentTaskStatus.EXPIRED.value),
}


async def apply_command(
    task_id: str,
    *,
    action: str,
    idempotency_key: str,
    expected_version: int,
    actor_id: Optional[str] = None,
) -> tuple[AgentTask, AgentTaskCommand, bool]:
    if action not in COMMAND_TRANSITIONS:
        raise AgentTaskConflict("task_command_unknown", f"Unsupported task command: {action}")
    async with async_session_maker() as session:
        async with session.begin():
            duplicate = (await session.execute(select(AgentTaskCommand).where(
                AgentTaskCommand.task_id == task_id,
                AgentTaskCommand.action == action,
                AgentTaskCommand.idempotency_key == idempotency_key,
            ))).scalar_one_or_none()
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None:
                raise AgentTaskConflict("task_not_found", "Agent task not found")
            if duplicate:
                return task, duplicate, True
            if task.version != expected_version:
                raise AgentTaskConflict("task_version_conflict", "Task version is stale", current_version=task.version)
            allowed, target = COMMAND_TRANSITIONS[action]
            if task.status not in allowed:
                raise AgentTaskConflict("task_transition_invalid", f"Cannot {action} task from {task.status}", current_version=task.version)
            command = AgentTaskCommand(
                task_id=task.id,
                action=action,
                idempotency_key=idempotency_key,
                expected_version=expected_version,
                actor_id=actor_id,
            )
            session.add(command)
            now = utc_now()
            if action == "pause" and task.status == AgentTaskStatus.QUEUED.value:
                target = AgentTaskStatus.PAUSED.value
            elif action == "cancel" and task.status in {
                AgentTaskStatus.CREATED.value,
                AgentTaskStatus.QUEUED.value,
                AgentTaskStatus.PAUSED.value,
                AgentTaskStatus.AWAITING_APPROVAL.value,
            }:
                target = AgentTaskStatus.CANCELLED.value
            task.status = target
            task.current_phase = target
            task.version += 1
            task.updated_at = now
            if target == AgentTaskStatus.QUEUED.value:
                task.queued_at = now
                task.lease_owner = None
                task.lease_expires_at = None
                task.expires_at = now + timedelta(hours=24)
                if action == "retry":
                    task.completed_at = None
                    task.paused_at = None
                    task.terminal_reason = None
                    todos = list((await session.execute(
                        select(AgentTaskTodo).where(AgentTaskTodo.task_id == task.id).with_for_update()
                    )).scalars().all())
                    for todo in todos:
                        if todo.status not in {"failed", "blocked", "cancelled"}:
                            continue
                        todo.status = "pending"
                        todo.attempt = 0
                        todo.progress = 0
                        todo.result_summary = None
                        todo.terminal_reason = None
                        todo.current_subagent_run_id = None
                        replace_jsonb_field(todo, "artifact_ids_json", [])
                        replace_jsonb_field(todo, "evidence_ids_json", [])
                        todo.version += 1
                        todo.updated_at = now
                    completed = sum(1 for todo in todos if todo.status == "completed")
                    task.completed_todos = completed
                    task.total_todos = len(todos)
                    task.progress = int((completed * 100) / len(todos)) if todos else 0
            if target == AgentTaskStatus.EXPIRED.value:
                task.completed_at = now
                task.terminal_reason = "approval_or_pause_expired"
            if target == AgentTaskStatus.PAUSED.value:
                task.paused_at = now
                task.lease_owner = None
                task.lease_expires_at = None
                task.expires_at = now + timedelta(days=7)
            if target == AgentTaskStatus.CANCELLED.value:
                task.completed_at = now
                task.terminal_reason = "cancelled_by_user"
                task.lease_owner = None
                task.lease_expires_at = None
            command.status = "completed"
            command.result_version = task.version
            replace_jsonb_field(command, "result_json", {"task_id": task.id, "status": task.status, "version": task.version})
            command.completed_at = now
            await _append_event(
                session,
                task,
                f"task.{action}_requested",
                actor_type="user",
                actor_id=actor_id,
                agent_run_id=task.active_run_id,
                payload={"status": task.status, "version": task.version},
            )
        await session.refresh(task)
        await session.refresh(command)
        return task, command, False


async def complete_control_command(command_id: str, *, result: dict[str, Any] | None = None, rejected: bool = False) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            command = await session.get(AgentTaskCommand, command_id, with_for_update=True)
            if command is not None:
                command.status = "rejected" if rejected else "completed"
                command.result_json = dict(result or {})
                command.completed_at = utc_now()


async def request_task_deletion(
    task_id: str,
    *,
    idempotency_key: str,
    expected_version: int,
    actor_id: Optional[str] = None,
) -> tuple[AgentTask, AgentTaskCommand, bool]:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None:
                raise AgentTaskConflict("task_not_found", "Agent task not found")
            duplicate = (await session.execute(select(AgentTaskCommand).where(
                AgentTaskCommand.task_id == task_id,
                AgentTaskCommand.action == "delete",
                AgentTaskCommand.idempotency_key == idempotency_key,
            ))).scalar_one_or_none()
            if duplicate:
                return task, duplicate, True
            if task.version != expected_version:
                raise AgentTaskConflict("task_version_conflict", "Task version is stale", current_version=task.version)
            if task.status not in TERMINAL_TASK_STATUSES:
                raise AgentTaskConflict("task_delete_nonterminal", "Only terminal tasks can be deleted", current_version=task.version)
            now = utc_now()
            command = AgentTaskCommand(
                task_id=task.id,
                action="delete",
                idempotency_key=idempotency_key,
                expected_version=expected_version,
                actor_id=actor_id,
                status="completed",
                result_version=task.version + 1,
                result_json={"task_id": task.id, "hidden": True},
                completed_at=now,
            )
            session.add(command)
            task.deletion_requested_at = task.deletion_requested_at or now
            task.version += 1
            task.updated_at = now
            await _append_event(session, task, "task.deletion_requested", actor_type="user", actor_id=actor_id)
        await session.refresh(task)
        await session.refresh(command)
        return task, command, False


async def list_pending_task_deletions(*, limit: int = 25) -> list[str]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTask.id)
            .where(AgentTask.deletion_requested_at.is_not(None), AgentTask.deletion_completed_at.is_(None))
            .order_by(AgentTask.deletion_requested_at)
            .limit(max(1, min(limit, 100)))
        )
        return [str(value) for value in result.scalars().all()]


async def mark_task_deletion_completed(task_id: str) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None or task.deletion_completed_at is not None:
                return
            task.deletion_completed_at = utc_now()
            task.updated_at = task.deletion_completed_at
            task.version += 1
            await _append_event(session, task, "task.deletion_completed")


async def claim_next_task(worker_id: str, *, lease_seconds: int = 60) -> Optional[AgentTask]:
    now = utc_now()
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(
                select(AgentTask)
                .where(
                    AgentTask.status.in_([
                        AgentTaskStatus.QUEUED.value,
                        AgentTaskStatus.RUNNING.value,
                        AgentTaskStatus.CANCELLING.value,
                    ]),
                    or_(AgentTask.lease_expires_at.is_(None), AgentTask.lease_expires_at < now),
                )
                .order_by(AgentTask.queued_at, AgentTask.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )).scalar_one_or_none()
            if task is None:
                return None
            was_queued = task.status == AgentTaskStatus.QUEUED.value
            is_cancelling = task.status == AgentTaskStatus.CANCELLING.value
            if not is_cancelling:
                task.status = AgentTaskStatus.RUNNING.value
                task.current_phase = "executing"
            task.lease_owner = worker_id
            task.heartbeat_at = now
            task.lease_expires_at = now + timedelta(seconds=max(15, lease_seconds))
            task.started_at = task.started_at or now
            if not is_cancelling:
                task.expires_at = None
            if was_queued:
                task.completed_at = None
                task.terminal_reason = None
            task.version += 1
            task.updated_at = now
            await _append_event(session, task, "task.claimed", agent_run_id=task.active_run_id, payload={"worker_id": worker_id, "version": task.version})
        await session.refresh(task)
        return task


async def heartbeat_task(task_id: str, worker_id: str, *, lease_seconds: int = 60) -> bool:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None or task.lease_owner != worker_id or task.status != AgentTaskStatus.RUNNING.value:
                return False
            now = utc_now()
            await _accrue_active_runtime(session, task, now=now, cap_ms=max(15, lease_seconds) * 1000)
            task.heartbeat_at = now
            task.lease_expires_at = now + timedelta(seconds=max(15, lease_seconds))
            task.updated_at = now
            return task.status == AgentTaskStatus.RUNNING.value


async def release_task_lease(task_id: str, worker_id: str, *, lease_seconds: int = 60) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is not None and task.lease_owner == worker_id:
                now = utc_now()
                await _accrue_active_runtime(session, task, now=now, cap_ms=max(15, lease_seconds) * 1000)
                task.lease_owner = None
                task.lease_expires_at = None
                task.heartbeat_at = now


async def _accrue_active_runtime(session: Any, task: AgentTask, *, now: Any, cap_ms: int) -> int:
    previous = task.heartbeat_at
    if previous is None or now <= previous:
        return 0
    increment = min(max(0, int((now - previous).total_seconds() * 1000)), max(1, cap_ms))
    if increment <= 0:
        return 0
    budgets = dict(task.budgets_json or {})
    elapsed = int(budgets.get("elapsed_active_ms") or 0) + increment
    budgets["elapsed_active_ms"] = elapsed
    replace_jsonb_field(task, "budgets_json", budgets)
    limits = (task.config_json or {}).get("limits") or {}
    maximum = int(limits.get("max_active_runtime_ms", 3_600_000))
    if elapsed >= maximum and task.status == AgentTaskStatus.RUNNING.value:
        task.status = AgentTaskStatus.CANCELLING.value
        task.current_phase = "active_runtime_budget_exhausted"
        task.terminal_reason = "active_runtime_budget_exhausted"
    task.version += 1
    await _append_event(
        session,
        task,
        "task.budget_updated",
        agent_run_id=task.active_run_id,
        payload={"elapsed_active_ms": elapsed, "max_active_runtime_ms": maximum},
    )
    return increment


async def active_runtime_budget_exhausted(task_id: str) -> bool:
    task = await get_task(task_id)
    if task is None:
        return True
    limits = (task.config_json or {}).get("limits") or {}
    return int((task.budgets_json or {}).get("elapsed_active_ms") or 0) >= int(
        limits.get("max_active_runtime_ms", 3_600_000)
    )


async def attach_run(task_id: str, run: AgentRun, *, parent_run_id: Optional[str] = None) -> AgentRun:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            stored_run = await session.get(AgentRun, run.id)
            if stored_run is None:
                raise AgentTaskConflict("task_run_missing", "Agent run does not exist")

            # The API eagerly prepares a run after the start command while a
            # worker may claim the task at the same time. Serialize attachment
            # on the task row and converge both callers on the run that won.
            # Query by task_id as well as active_run_id so this also repairs a
            # stale task pointer left by an interrupted attachment.
            existing_active = (await session.execute(
                select(AgentRun).where(
                    AgentRun.task_id == task.id,
                    AgentRun.status.in_([
                        AgentRunStatus.RUNNING.value,
                        AgentRunStatus.AWAITING_HUMAN.value,
                    ]),
                ).order_by(AgentRun.task_attempt.desc(), AgentRun.started_at.desc()).limit(1)
            )).scalar_one_or_none()
            if existing_active is not None and existing_active.id == stored_run.id:
                task.active_run_id = existing_active.id
                task.primary_run_id = task.primary_run_id or existing_active.id
                task.latest_run_attempt = max(task.latest_run_attempt, existing_active.task_attempt)
                return existing_active
            if existing_active is not None and existing_active.id != stored_run.id:
                stored_run.status = AgentRunStatus.CANCELLED.value
                stored_run.completed_at = utc_now()
                stored_run.error_json = {
                    "code": "concurrent_task_run_superseded",
                    "retryable": False,
                    "active_run_id": existing_active.id,
                }
                task.active_run_id = existing_active.id
                task.primary_run_id = task.primary_run_id or existing_active.id
                task.latest_run_attempt = max(task.latest_run_attempt, existing_active.task_attempt)
                return existing_active

            next_attempt = task.latest_run_attempt + 1
            stored_run.task_id = task.id
            stored_run.parent_run_id = parent_run_id
            stored_run.task_attempt = next_attempt
            task.primary_run_id = task.primary_run_id or stored_run.id
            task.active_run_id = stored_run.id
            task.latest_run_attempt = next_attempt
            task.version += 1
            await _append_event(session, task, "task.run_attached", agent_run_id=stored_run.id, payload={"attempt": next_attempt})
        await session.refresh(task)
        await session.refresh(stored_run)
        return stored_run


async def persist_plan(
    task_id: str,
    proposal: DeepResearchPlanProposal,
    *,
    agent_run_id: str,
    reason: str,
    planner_visit: int,
) -> tuple[AgentTaskPlanRevision, list[AgentTaskTodo]]:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            latest = int((await session.execute(select(func.coalesce(func.max(AgentTaskPlanRevision.revision), 0)).where(AgentTaskPlanRevision.task_id == task_id))).scalar_one())
            revision_number = latest + 1
            limits = (task.config_json or {}).get("limits") or {}
            run_revision_count = int((await session.execute(
                select(func.count(AgentTaskPlanRevision.id)).where(
                    AgentTaskPlanRevision.task_id == task_id,
                    AgentTaskPlanRevision.agent_run_id == agent_run_id,
                )
            )).scalar_one())
            if run_revision_count >= int(limits.get("max_plan_revisions", 8)):
                raise AgentTaskConflict("plan_revision_budget_exhausted", "Plan revision limit reached")
            enabled_profiles = set((task.config_json or {}).get("enabled_profiles") or [])
            for todo in proposal.todos:
                if todo.profile_id.value == "evidence_critic":
                    raise AgentTaskConflict("plan_profile_not_schedulable", "Evidence critic is reserved for final review")
                if todo.profile_id.value not in enabled_profiles:
                    raise AgentTaskConflict("plan_profile_not_allowed", f"Profile {todo.profile_id.value} is disabled")
            revision = AgentTaskPlanRevision(
                task_id=task.id,
                agent_run_id=agent_run_id,
                revision=revision_number,
                planner_visit=planner_visit,
                reason=reason,
                objective=proposal.objective,
                completion_criteria_json=proposal.success_criteria,
                ordered_todo_ids_json=[todo.id for todo in proposal.todos],
                plan_json=proposal.model_dump(mode="json"),
                provenance_json={"config_hash": canonical_hash(task.config_json)},
                content_hash=proposal.content_hash(),
            )
            session.add(revision)
            existing = {
                todo.id: todo
                for todo in (await session.execute(select(AgentTaskTodo).where(AgentTaskTodo.task_id == task.id))).scalars().all()
            }
            persisted: list[AgentTaskTodo] = []
            proposed_ids = {value.id for value in proposal.todos}
            for value in proposal.todos:
                current = existing.get(value.id)
                if current and current.status == "completed":
                    persisted.append(current)
                    continue
                if current is None:
                    current = AgentTaskTodo(
                        id=value.id,
                        task_id=task.id,
                        title=value.title,
                        description=value.description,
                        completion_criteria=value.completion_criteria,
                        priority=value.priority,
                        required=value.required,
                        dependency_ids_json=value.dependency_ids,
                        profile_id=value.profile_id.value,
                        max_attempts=int(limits.get("max_attempts_per_todo", 2)),
                        created_revision=revision_number,
                        updated_revision=revision_number,
                    )
                    session.add(current)
                else:
                    current.title = value.title
                    current.description = value.description
                    current.completion_criteria = value.completion_criteria
                    current.priority = value.priority
                    current.required = value.required
                    replace_jsonb_field(current, "dependency_ids_json", value.dependency_ids)
                    current.profile_id = value.profile_id.value
                    current.updated_revision = revision_number
                    current.version += 1
                    current.updated_at = utc_now()
                persisted.append(current)
            for omitted in existing.values():
                if omitted.id in proposed_ids or omitted.status == "completed":
                    continue
                if omitted.status == "running":
                    raise AgentTaskConflict("plan_supersedes_running_todo", "A plan revision cannot supersede running work")
                omitted.status = "skipped"
                omitted.required = False
                omitted.terminal_reason = f"superseded_by_plan_revision:{revision_number}"
                omitted.updated_revision = revision_number
                omitted.version += 1
                omitted.updated_at = utc_now()
                persisted.append(omitted)
            await session.flush()
            task.total_todos = len(persisted)
            task.current_phase = "planned"
            task.version += 1
            await _append_event(session, task, "plan.revised", agent_run_id=agent_run_id, payload={"revision": revision_number, "todo_count": len(persisted), "content_hash": revision.content_hash})
        await session.refresh(revision)
        return revision, persisted


async def schedule_ready_todos(task_id: str, *, limit: int) -> list[AgentTaskTodo]:
    """Atomically project dependency-ready todos and claim a bounded batch."""
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            todos = list((await session.execute(
                select(AgentTaskTodo).where(AgentTaskTodo.task_id == task_id).with_for_update()
            )).scalars().all())
            by_id = {todo.id: todo for todo in todos}
            terminal_success = {todo.id for todo in todos if todo.status in {"completed", "skipped"}}
            changed: list[AgentTaskTodo] = []
            for todo in todos:
                dependencies = [str(value) for value in (todo.dependency_ids_json or [])]
                if todo.status == "pending" and all(value in terminal_success for value in dependencies):
                    todo.status = "ready"
                    todo.version += 1
                    todo.updated_at = utc_now()
                    changed.append(todo)
                elif todo.status == "pending" and any(by_id.get(value) and by_id[value].status in {"failed", "cancelled", "blocked"} for value in dependencies):
                    todo.status = "blocked"
                    todo.terminal_reason = "dependency_failed"
                    todo.version += 1
                    todo.updated_at = utc_now()
                    changed.append(todo)
            # A scheduler replay can occur after todos were atomically claimed
            # but before the graph checkpoint committed the dispatch result.
            # Re-emit only attempts that have not started a subagent execution.
            claimed = [todo for todo in todos if todo.status == "running" and not todo.current_subagent_run_id]
            ready = sorted(
                [*claimed, *(todo for todo in todos if todo.status == "ready")],
                key=lambda value: (-value.priority, value.created_at, value.id),
            )[:max(1, limit)]
            for todo in ready:
                if todo.status == "ready":
                    todo.status = "running"
                    todo.attempt += 1
                todo.progress = max(1, todo.progress)
                todo.version += 1
                todo.updated_at = utc_now()
                await _append_event(session, task, "todo.started", agent_run_id=task.active_run_id, todo_id=todo.id, payload={"attempt": todo.attempt, "profile_id": todo.profile_id})
            completed = sum(1 for todo in todos if todo.status == "completed")
            task.completed_todos = completed
            task.total_todos = len(todos)
            task.progress = int((completed * 100) / len(todos)) if todos else 0
            task.current_phase = "dispatching" if ready else "controlling"
            task.version += 1
            for todo in changed:
                await _append_event(session, task, f"todo.{todo.status}", agent_run_id=task.active_run_id, todo_id=todo.id, payload={"version": todo.version})
        for todo in ready:
            await session.refresh(todo)
        return ready


async def block_todos(task_id: str, todo_ids: Iterable[str], *, reason: str) -> list[AgentTaskTodo]:
    ids = {str(value) for value in todo_ids if value}
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            rows = list((await session.execute(select(AgentTaskTodo).where(
                AgentTaskTodo.task_id == task_id,
                AgentTaskTodo.id.in_(ids),
            ).with_for_update())).scalars().all())
            for todo in rows:
                if todo.status in {"pending", "ready", "running"}:
                    todo.status = "blocked" if todo.required else "skipped"
                    todo.terminal_reason = reason
                    todo.current_subagent_run_id = None
                    todo.version += 1
                    todo.updated_at = utc_now()
                    await _append_event(session, task, f"todo.{todo.status}", agent_run_id=task.active_run_id, todo_id=todo.id, payload={"reason": reason})
            task.version += 1
        return rows


async def record_subagent_started(
    *, task_id: str, agent_run_id: str, todo_id: str, profile_id: str,
    plan_revision: int, attempt: int, timeout_ms: int, tool_policy_hash: str,
) -> tuple[AgentTaskSubagentRun, bool]:
    execution_key = canonical_hash({
        "task_id": task_id, "todo_id": todo_id, "profile_id": profile_id,
        "plan_revision": plan_revision, "attempt": attempt,
    })
    async with async_session_maker() as session:
        async with session.begin():
            existing = (await session.execute(select(AgentTaskSubagentRun).where(AgentTaskSubagentRun.execution_key == execution_key))).scalar_one_or_none()
            if existing:
                return existing, True
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            row = AgentTaskSubagentRun(
                task_id=task_id,
                agent_run_id=agent_run_id,
                todo_id=todo_id,
                execution_key=execution_key,
                profile_id=profile_id,
                plan_revision=plan_revision,
                attempt=attempt,
                status="running",
                tool_policy_hash=tool_policy_hash,
                timeout_ms=timeout_ms,
                started_at=utc_now(),
            )
            session.add(row)
            await session.flush()
            todo = await session.get(AgentTaskTodo, (task_id, todo_id))
            if todo is not None:
                todo.current_subagent_run_id = row.id
            budgets = dict(task.budgets_json or {})
            budgets["subagent_attempts"] = int(budgets.get("subagent_attempts") or 0) + 1
            replace_jsonb_field(task, "budgets_json", budgets)
            await _append_event(session, task, "subagent.started", agent_run_id=agent_run_id, todo_id=todo_id, subagent_run_id=row.id, payload={"profile_id": profile_id, "attempt": attempt})
        await session.refresh(row)
        return row, False


async def record_subagent_result(
    *, task_id: str, todo_id: str, subagent_run_id: str, status: str,
    summary: str, artifact_ids: list[str], usage: Dict[str, Any], error: Optional[Dict[str, Any]], retryable: bool,
) -> AgentTaskTodo:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            todo = (await session.execute(select(AgentTaskTodo).where(
                AgentTaskTodo.task_id == task_id, AgentTaskTodo.id == todo_id
            ).with_for_update())).scalar_one()
            subagent = await session.get(AgentTaskSubagentRun, subagent_run_id)
            if subagent is None or subagent.task_id != task_id:
                raise AgentTaskConflict("subagent_run_mismatch", "Subagent execution does not belong to task")
            if subagent.completed_at is not None and subagent.status in {"completed", "failed", "timed_out", "cancelled"}:
                return todo
            subagent.status = status
            subagent.completed_at = utc_now()
            replace_jsonb_field(subagent, "usage_json", usage)
            replace_jsonb_field(subagent, "output_artifact_ids_json", artifact_ids)
            if error is not None:
                replace_jsonb_field(subagent, "error_json", error)
            todo.result_summary = summary[:12_000]
            replace_jsonb_field(todo, "artifact_ids_json", list(dict.fromkeys([*(todo.artifact_ids_json or []), *artifact_ids])))
            if status == "completed":
                todo.status = "completed"
                todo.progress = 100
            elif status == "cancelled":
                todo.status = "cancelled"
                todo.terminal_reason = "task_cancelled"
            elif retryable and todo.attempt < todo.max_attempts:
                todo.status = "ready"
                todo.terminal_reason = None
            else:
                todo.status = "failed"
                todo.terminal_reason = str((error or {}).get("code") or status)
            todo.current_subagent_run_id = None
            todo.version += 1
            todo.updated_at = utc_now()
            todos = list((await session.execute(select(AgentTaskTodo).where(AgentTaskTodo.task_id == task_id))).scalars().all())
            task.completed_todos = sum(1 for value in todos if value.status == "completed")
            task.total_todos = len(todos)
            task.progress = int((task.completed_todos * 100) / len(todos)) if todos else 0
            task.version += 1
            await _append_event(session, task, f"subagent.{status}", agent_run_id=subagent.agent_run_id, todo_id=todo.id, subagent_run_id=subagent.id, payload={"todo_status": todo.status, "retryable": retryable, "artifact_ids": artifact_ids})
            await _append_event(session, task, f"todo.{todo.status}", agent_run_id=subagent.agent_run_id, todo_id=todo.id, payload={"attempt": todo.attempt, "progress": todo.progress})
        await session.refresh(todo)
        return todo


async def register_artifact(metadata: AgentTaskArtifact) -> tuple[AgentTaskArtifact, bool]:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == metadata.task_id).with_for_update())).scalar_one()
            duplicate = (await session.execute(select(AgentTaskArtifact).where(
                AgentTaskArtifact.agent_run_id == metadata.agent_run_id,
                AgentTaskArtifact.ownership_key == metadata.ownership_key,
                AgentTaskArtifact.sha256 == metadata.sha256,
                AgentTaskArtifact.kind == metadata.kind,
                AgentTaskArtifact.validity == "valid",
            ))).scalar_one_or_none()
            if duplicate:
                return duplicate, True
            limits = (task.config_json or {}).get("limits") or {}
            artifact_count = int((await session.execute(select(func.count(AgentTaskArtifact.id)).where(
                AgentTaskArtifact.task_id == task.id,
                AgentTaskArtifact.validity != "deleted",
            ))).scalar_one())
            budgets = dict(task.budgets_json or {})
            if artifact_count >= int(limits.get("max_artifacts", 200)):
                raise AgentTaskConflict("artifact_count_budget_exhausted", "Task artifact count limit reached")
            if metadata.byte_size > int(limits.get("max_single_artifact_bytes", 10_485_760)):
                raise AgentTaskConflict("artifact_size_budget_exhausted", "Task artifact exceeds its configured size limit")
            if int(budgets.get("artifact_bytes") or 0) + metadata.byte_size > int(limits.get("max_artifact_bytes", 104_857_600)):
                raise AgentTaskConflict("artifact_bytes_budget_exhausted", "Task artifact byte budget exhausted")
            session.add(metadata)
            await session.flush()
            budgets["artifact_bytes"] = int(budgets.get("artifact_bytes") or 0) + metadata.byte_size
            replace_jsonb_field(task, "budgets_json", budgets)
            await _append_event(session, task, "artifact.created", agent_run_id=metadata.agent_run_id, todo_id=metadata.todo_id, subagent_run_id=metadata.subagent_run_id, artifact_id=metadata.id, payload={"kind": metadata.kind, "byte_size": metadata.byte_size, "sha256": metadata.sha256})
        await session.refresh(metadata)
        return metadata, False


async def complete_task(task_id: str, *, status: str, reason: Optional[str] = None, final_artifact_id: Optional[str] = None) -> AgentTask:
    if status not in TERMINAL_TASK_STATUSES:
        raise ValueError("task completion requires a terminal status")
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            if task.status in TERMINAL_TASK_STATUSES:
                return task
            task.status = status
            task.current_phase = status
            task.terminal_reason = reason
            task.completed_at = utc_now()
            task.expires_at = None
            task.lease_owner = None
            task.lease_expires_at = None
            task.version += 1
            await _append_event(session, task, f"task.{status}", agent_run_id=task.active_run_id, artifact_id=final_artifact_id, payload={"reason": reason, "version": task.version})
        await session.refresh(task)
        return task


async def set_task_runtime_status(task_id: str, status: str, *, phase: Optional[str] = None, reason: Optional[str] = None) -> AgentTask:
    if status not in {value.value for value in AgentTaskStatus}:
        raise ValueError("unknown task status")
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            if task.status in TERMINAL_TASK_STATUSES:
                return task
            task.status = status
            task.current_phase = phase or status
            task.terminal_reason = reason
            task.updated_at = utc_now()
            task.version += 1
            if status == AgentTaskStatus.PAUSED.value:
                task.paused_at = utc_now()
                task.expires_at = utc_now() + timedelta(days=7)
            elif status == AgentTaskStatus.AWAITING_APPROVAL.value:
                task.expires_at = utc_now() + timedelta(days=7)
            if status != AgentTaskStatus.RUNNING.value:
                task.lease_owner = None
                task.lease_expires_at = None
            await _append_event(session, task, f"task.{status}", agent_run_id=task.active_run_id, payload={"phase": task.current_phase, "reason": reason, "version": task.version})
        await session.refresh(task)
        return task


async def expire_stale_tasks(*, limit: int = 100) -> int:
    now = utc_now()
    async with async_session_maker() as session:
        async with session.begin():
            rows = list((await session.execute(
                select(AgentTask)
                .where(
                    AgentTask.status.in_([
                        AgentTaskStatus.CREATED.value,
                        AgentTaskStatus.QUEUED.value,
                        AgentTaskStatus.PAUSED.value,
                        AgentTaskStatus.AWAITING_APPROVAL.value,
                    ]),
                    AgentTask.expires_at.is_not(None),
                    AgentTask.expires_at < now,
                )
                .with_for_update(skip_locked=True)
                .limit(max(1, min(limit, 1000)))
            )).scalars().all())
            for task in rows:
                task.status = AgentTaskStatus.EXPIRED.value
                task.current_phase = "expired"
                task.terminal_reason = "idle_or_approval_expired"
                task.completed_at = now
                task.lease_owner = None
                task.lease_expires_at = None
                task.version += 1
                if task.active_run_id:
                    run = await session.get(AgentRun, task.active_run_id)
                    if run is not None and run.status in {"running", "awaiting_human"}:
                        run.status = "expired"
                        run.completed_at = now
                        pending = dict(run.pending_interrupt_json or {})
                        if pending:
                            pending["status"] = "expired"
                            pending["decision"] = {"action": "expire", "reason": task.terminal_reason}
                            replace_jsonb_field(run, "pending_interrupt_json", pending)
                        replace_jsonb_field(run, "error_json", {
                            "code": "agent_task_expired", "raw_message": task.terminal_reason, "retryable": False,
                        })
                await _append_event(session, task, "task.expired", agent_run_id=task.active_run_id, payload={"reason": task.terminal_reason})
            return len(rows)
