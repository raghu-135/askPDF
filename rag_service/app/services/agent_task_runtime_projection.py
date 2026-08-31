"""Apply framework-neutral runtime task deltas to product-owned records."""

from __future__ import annotations

from typing import Any, Mapping

from app.models.deep_research import DeepResearchPlanProposal
from app.services import agent_task_repository as tasks
from runtime_protocol.contracts import TaskOrchestrationDelta


class RuntimeTaskProjectionConflict(RuntimeError):
    """The runtime based its delta on stale product state."""


async def apply_runtime_task_delta(
    *,
    task_id: str,
    agent_run_id: str,
    delta: TaskOrchestrationDelta,
    artifact_id_map: Mapping[str, str],
    observed_version_verified: bool = False,
) -> None:
    if delta.attempt_id != agent_run_id:
        raise RuntimeTaskProjectionConflict("runtime delta attempt does not match the product run")

    plans = await tasks.list_plans(task_id)
    already_applied = any(
        str((plan.provenance_json or {}).get("runtime_idempotency_key") or "") == delta.idempotency_key
        for plan in plans
    )
    task = await tasks.get_task(task_id)
    if task is None:
        raise RuntimeTaskProjectionConflict("runtime delta targets a missing task")
    if task.version != delta.observed_task_version and not already_applied and not observed_version_verified:
        raise RuntimeTaskProjectionConflict(
            f"stale runtime delta: observed task version {delta.observed_task_version}, current version {task.version}"
        )

    plan_revision = delta.observed_plan_revision
    if delta.plan is not None:
        revision, _ = await tasks.persist_plan(
            task_id,
            DeepResearchPlanProposal.model_validate(dict(delta.plan)),
            agent_run_id=agent_run_id,
            reason="runtime_projection",
            planner_visit=max(1, delta.observed_plan_revision + 1),
            idempotency_key=delta.idempotency_key,
        )
        plan_revision = revision.revision

    todos = {todo.id: todo for todo in await tasks.list_todos(task_id)}
    for packet in delta.subagent_changes:
        todo_id = str(packet.get("todo_id") or "")
        todo = todos.get(todo_id)
        if todo is None:
            raise RuntimeTaskProjectionConflict(f"runtime delta references unknown todo {todo_id!r}")
        attempt = max(1, int(todo.attempt or packet.get("attempt") or 1))
        subagent, _ = await tasks.record_subagent_started(
            task_id=task_id,
            agent_run_id=agent_run_id,
            todo_id=todo_id,
            profile_id=str(todo.profile_id or packet.get("profile_id") or "general_researcher"),
            plan_revision=max(1, plan_revision),
            attempt=attempt,
            timeout_ms=int(packet.get("timeout_ms") or 180_000),
            tool_policy_hash=str(packet.get("tool_policy_hash") or delta.idempotency_key),
        )
        artifact_ids = [
            artifact_id_map.get(str(value), str(value))
            for value in packet.get("artifact_ids") or []
        ]
        await tasks.record_subagent_result(
            task_id=task_id,
            todo_id=todo_id,
            subagent_run_id=subagent.id,
            status=str(packet.get("status") or "failed"),
            summary=str(packet.get("summary") or ""),
            artifact_ids=artifact_ids,
            usage=dict(packet.get("usage") or {}),
            error=dict(packet["error"]) if isinstance(packet.get("error"), Mapping) else None,
            retryable=bool(packet.get("retryable")),
        )
