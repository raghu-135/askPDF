"""Atomically apply framework-neutral runtime task deltas to product records."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import timedelta
from typing import Any, Mapping

from sqlalchemy import func
from sqlalchemy.future import select

from app.agent_workflows.interrupts import normalize_pending_interrupt_payload, run_interrupt_resume_guard
from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import AgentRunStatus
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import (
    AgentRun, AgentTask, AgentTaskArtifact, AgentTaskCommand, AgentTaskPlanRevision,
    AgentTaskRuntimeDelta, AgentTaskSubagentRun, AgentTaskTodo,
)
from app.models.deep_research import AgentTaskStatus, DeepResearchPlanProposal
from app.services import agent_task_repository as tasks
from app.services.agent_task_budgets import normalize_budget_state
from app.services.content_store import get_content_store, task_artifact_content_key
from app.time_utils import utc_now
from runtime_protocol.contracts import TaskOrchestrationDelta


class RuntimeTaskProjectionConflict(RuntimeError):
    """The runtime delta is stale, malformed, or causally inconsistent."""


def _payload_hash(delta: TaskOrchestrationDelta) -> str:
    encoded = json.dumps(delta.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def runtime_delta_conflict_details(
    *, task: Any, agent_run_id: str, delta: TaskOrchestrationDelta, current_plan_revision: int,
) -> dict[str, Any] | None:
    active_run_id = str(getattr(task, "active_run_id", "") or "")
    if active_run_id != agent_run_id:
        return {"reason": "active_run_changed", "active_run_id": active_run_id or None}
    attempt_prefix = f"{agent_run_id}:attempt:"
    attempt_number = delta.attempt_id.removeprefix(attempt_prefix)
    if (
        not delta.attempt_id.startswith(attempt_prefix)
        or not attempt_number.isdigit()
        or int(attempt_number) < 1
    ):
        return {"reason": "runtime_attempt_identity_invalid"}
    expected_event_id = f"{delta.attempt_id}:operation:{delta.operation_id}:result"
    if delta.event_id != expected_event_id:
        return {"reason": "boundary_event_identity_invalid", "expected_event_id": expected_event_id}
    # Product-owned events, leases, and commands may advance the task version
    # while the same runtime operation is active.  Plan revision and active-run
    # identity are the orchestration conflict guards; a runtime claiming a
    # future product version is always invalid.
    if delta.observed_task_version > int(task.version):
        return {"reason": "observed_version_is_ahead"}
    if int(current_plan_revision) != delta.observed_plan_revision:
        return {
            "reason": "plan_revision_changed", "current_plan_revision": int(current_plan_revision),
            "observed_plan_revision": delta.observed_plan_revision,
        }
    return None


async def _stage_artifacts(
    *, task_id: str, agent_run_id: str, artifacts: tuple[Mapping[str, Any], ...]
) -> list[dict[str, Any]]:
    staged: list[dict[str, Any]] = []
    store = get_content_store()
    for source in artifacts:
        runtime_id = str(source.get("artifact_id") or source.get("id") or "").strip()
        content = source.get("content")
        if not runtime_id or content is None:
            raise RuntimeTaskProjectionConflict("runtime artifact requires artifact_id and content")
        if not isinstance(content, (str, bytes, bytearray)):
            raise RuntimeTaskProjectionConflict("runtime artifact content must be text or bytes")
        body = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        if len(body) > 10_485_760:
            raise RuntimeTaskProjectionConflict("runtime artifact exceeds the per-object limit")
        digest = hashlib.sha256(body).hexdigest()
        if str(source.get("sha256") or digest) != digest:
            raise RuntimeTaskProjectionConflict("runtime artifact digest mismatch")
        artifact_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"askpdf:{task_id}:{agent_run_id}:{runtime_id}:{digest}"))
        object_key = task_artifact_content_key(task_id, agent_run_id, artifact_id)
        await store.put(object_key, body, expected_sha256=digest)
        staged.append({**dict(source), "runtime_id": runtime_id, "id": artifact_id, "object_key": object_key, "sha256": digest, "byte_size": len(body)})
    return staged


def _merge_budget(
    current: Mapping[str, Any],
    incoming: Mapping[str, Any],
    limits: Mapping[str, Any],
    *,
    authorized_tranche_increment: bool = False,
) -> dict[str, Any]:
    existing = normalize_budget_state(current, limits)
    candidate = normalize_budget_state(incoming, limits)
    old_tranche, new_tranche = int(existing.get("tranche_index") or 1), int(candidate.get("tranche_index") or 1)
    if new_tranche < old_tranche or new_tranche > old_tranche + 1:
        raise RuntimeTaskProjectionConflict("runtime budget tranche is stale or skips a tranche")
    if new_tranche > old_tranche and not authorized_tranche_increment:
        raise RuntimeTaskProjectionConflict("runtime budget tranche increment is not product-authorized")
    for key, old_value in (existing.get("lifetime_usage") or {}).items():
        if int((candidate.get("lifetime_usage") or {}).get(key) or 0) < int(old_value or 0):
            raise RuntimeTaskProjectionConflict(f"runtime lifetime budget counter regressed: {key}")
    if new_tranche == old_tranche:
        for key, old_value in (existing.get("tranche_usage") or {}).items():
            if int((candidate.get("tranche_usage") or {}).get(key) or 0) < int(old_value or 0):
                raise RuntimeTaskProjectionConflict(f"runtime tranche budget counter regressed: {key}")
    candidate["tranche_limits"] = existing["tranche_limits"]
    return candidate


async def apply_runtime_task_delta(
    *, task_id: str, agent_run_id: str, delta: TaskOrchestrationDelta,
    artifact_id_map: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply one runtime boundary exactly once in one product DB transaction."""

    staged = await _stage_artifacts(task_id=task_id, agent_run_id=agent_run_id, artifacts=delta.artifacts) if delta.artifacts else []
    artifact_ids = {str(value["runtime_id"]): str(value["id"]) for value in staged}
    artifact_ids.update({str(key): str(value) for key, value in (artifact_id_map or {}).items()})
    payload_sha256 = _payload_hash(delta)

    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            run = (await session.execute(select(AgentRun).where(AgentRun.id == agent_run_id, AgentRun.task_id == task_id).with_for_update())).scalar_one_or_none()
            if task is None or run is None:
                raise RuntimeTaskProjectionConflict("runtime delta targets a missing task or run")
            existing = list((await session.execute(select(AgentTaskRuntimeDelta).where(
                (AgentTaskRuntimeDelta.task_id == task_id) &
                ((AgentTaskRuntimeDelta.idempotency_key == delta.idempotency_key) |
                 ((AgentTaskRuntimeDelta.agent_run_id == agent_run_id) & (AgentTaskRuntimeDelta.event_id == delta.event_id)))
            ).with_for_update())).scalars().all())
            if existing:
                if any(value.payload_sha256 != payload_sha256 for value in existing):
                    raise RuntimeTaskProjectionConflict("runtime delta identity was reused with different content")
                return artifact_ids
            if task.status in tasks.TERMINAL_TASK_STATUSES or task.deletion_requested_at is not None:
                raise RuntimeTaskProjectionConflict("runtime delta targets an immutable task")
            plans = list((await session.execute(select(AgentTaskPlanRevision).where(AgentTaskPlanRevision.task_id == task_id))).scalars().all())
            current_plan_revision = max((int(value.revision) for value in plans), default=0)
            conflict = runtime_delta_conflict_details(task=task, agent_run_id=agent_run_id, delta=delta, current_plan_revision=current_plan_revision)
            if conflict is not None:
                raise RuntimeTaskProjectionConflict(f"stale runtime delta: {conflict}")

            plan_revision = current_plan_revision
            applied_runtime_plan_revision = delta.observed_plan_revision
            expected_runtime_revision = delta.observed_plan_revision + 1
            current_todos = {row.id: row for row in (await session.execute(
                select(AgentTaskTodo).where(AgentTaskTodo.task_id == task_id).with_for_update()
            )).scalars().all()}
            applied_correction_ids: set[str] = set()
            for plan_change in delta.plan_changes:
                if plan_change.runtime_revision != expected_runtime_revision:
                    raise RuntimeTaskProjectionConflict("runtime plan lineage contains a revision gap")
                if plan_change.parent_runtime_revision != plan_change.runtime_revision - 1:
                    raise RuntimeTaskProjectionConflict("runtime plan lineage contains a conflicting parent")
                if plan_change.acknowledged_product_revision != delta.observed_plan_revision:
                    raise RuntimeTaskProjectionConflict("runtime plan lineage has a mismatched product baseline")
                proposal = DeepResearchPlanProposal.model_validate(dict(plan_change.plan))
                plan_revision += 1
                limits = dict((task.config_json or {}).get("limits") or {})
                if len(proposal.todos) > int(limits.get("max_todos", 50)):
                    raise RuntimeTaskProjectionConflict("runtime plan exceeds the product todo limit")
                if plan_revision > int(limits.get("max_plan_revisions", 8)):
                    raise RuntimeTaskProjectionConflict("runtime plan exceeds the product revision limit")
                enabled_profiles = set((task.config_json or {}).get("enabled_profiles") or [])
                for item in proposal.todos:
                    if item.profile_id.value == "evidence_critic" or item.profile_id.value not in enabled_profiles:
                        raise RuntimeTaskProjectionConflict(f"runtime plan contains disallowed profile {item.profile_id.value}")
                revision = AgentTaskPlanRevision(
                    task_id=task_id, agent_run_id=agent_run_id, revision=plan_revision,
                    planner_visit=plan_change.planner_visit, reason=plan_change.reason,
                    objective=proposal.objective, completion_criteria_json=proposal.success_criteria,
                    ordered_todo_ids_json=[item.id for item in proposal.todos], plan_json=proposal.model_dump(mode="json"),
                    provenance_json={
                        "config_hash": tasks.canonical_hash(task.config_json),
                        "runtime_delta_id": delta.idempotency_key,
                        "runtime_revision": plan_change.runtime_revision,
                        "parent_runtime_revision": plan_change.parent_runtime_revision,
                        "correction_ids": list(plan_change.correction_ids),
                    },
                    content_hash=proposal.content_hash(),
                )
                session.add(revision)
                proposed_ids = {item.id for item in proposal.todos}
                for item in proposal.todos:
                    todo = current_todos.get(item.id)
                    if todo is not None and todo.status == "completed":
                        continue
                    if todo is None:
                        todo = AgentTaskTodo(
                            id=item.id, task_id=task_id, title=item.title, description=item.description,
                            completion_criteria=item.completion_criteria, priority=item.priority, required=item.required,
                            dependency_ids_json=list(item.dependency_ids), profile_id=item.profile_id.value,
                            max_attempts=int(limits.get("max_attempts_per_todo", 2)), created_revision=plan_revision,
                            updated_revision=plan_revision,
                        )
                        session.add(todo)
                        current_todos[todo.id] = todo
                    else:
                        todo.title, todo.description, todo.completion_criteria = item.title, item.description, item.completion_criteria
                        todo.priority, todo.required, todo.profile_id = item.priority, item.required, item.profile_id.value
                        replace_jsonb_field(todo, "dependency_ids_json", list(item.dependency_ids))
                        todo.updated_revision, todo.version = plan_revision, todo.version + 1
                for todo in current_todos.values():
                    if todo.id not in proposed_ids and todo.status == "running":
                        raise RuntimeTaskProjectionConflict("runtime plan cannot supersede running work")
                    if todo.id not in proposed_ids and todo.status != "completed":
                        todo.status, todo.required, todo.terminal_reason = "skipped", False, f"superseded_by_plan_revision:{plan_revision}"
                        todo.updated_revision, todo.version = plan_revision, todo.version + 1
                await tasks._append_event(session, task, "plan.revised", agent_run_id=agent_run_id, payload={"revision": plan_revision, "content_hash": revision.content_hash})
                applied_runtime_plan_revision = plan_change.runtime_revision
                expected_runtime_revision += 1
                applied_correction_ids.update(plan_change.correction_ids)

            limits = dict((task.config_json or {}).get("limits") or {})
            valid_todo_ids = set((await session.execute(select(AgentTaskTodo.id).where(
                AgentTaskTodo.task_id == task_id,
            ))).scalars().all())
            artifact_count = int((await session.execute(select(func.count(AgentTaskArtifact.id)).where(
                AgentTaskArtifact.task_id == task_id,
                AgentTaskArtifact.validity != "deleted",
            ))).scalar_one())
            new_artifact_bytes = 0
            for value in staged:
                kind = str(value.get("kind") or "tool_output")
                if kind not in {"tool_output", "intermediate_report", "context_summary", "final_report"}:
                    raise RuntimeTaskProjectionConflict(f"runtime artifact has unsupported kind {kind!r}")
                if value.get("todo_id") is not None and str(value["todo_id"]) not in valid_todo_ids:
                    raise RuntimeTaskProjectionConflict("runtime artifact references an unknown todo")
                ownership_key = f"runtime:{value['runtime_id']}"
                duplicate = (await session.execute(select(AgentTaskArtifact).where(
                    AgentTaskArtifact.agent_run_id == agent_run_id,
                    AgentTaskArtifact.ownership_key == ownership_key,
                    AgentTaskArtifact.validity == "valid",
                ))).scalar_one_or_none()
                if duplicate is not None:
                    if duplicate.sha256 != value["sha256"] or duplicate.kind != kind:
                        raise RuntimeTaskProjectionConflict("runtime artifact identity was reused with different content")
                    artifact_ids[str(value["runtime_id"])] = duplicate.id
                    continue
                if artifact_count >= int(limits.get("max_artifacts", 200)):
                    raise RuntimeTaskProjectionConflict("runtime artifact count exceeds the product limit")
                if value["byte_size"] > int(limits.get("max_single_artifact_bytes", 10_485_760)):
                    raise RuntimeTaskProjectionConflict("runtime artifact exceeds the product per-object limit")
                artifact_count += 1
                new_artifact_bytes += int(value["byte_size"])
                artifact = AgentTaskArtifact(
                    id=value["id"], task_id=task_id, agent_run_id=agent_run_id, todo_id=value.get("todo_id"),
                    ownership_key=ownership_key, kind=kind,
                    object_key=value["object_key"], media_type=str(value.get("media_type") or "text/plain"),
                    byte_size=value["byte_size"], sha256=value["sha256"],
                    provenance_json={**dict(value.get("provenance") or {}), "runtime_artifact_id": value["runtime_id"], "runtime_subagent_run_id": value.get("subagent_run_id")},
                    source_refs_json=dict(value.get("source_refs") or {}), retention_until=utc_now() + timedelta(days=30),
                )
                session.add(artifact)
                await tasks._append_event(session, task, "artifact.created", agent_run_id=agent_run_id, todo_id=artifact.todo_id, artifact_id=artifact.id, payload={"kind": artifact.kind, "byte_size": artifact.byte_size, "sha256": artifact.sha256})

            current_budget = normalize_budget_state(task.budgets_json, limits)
            if int(current_budget["lifetime_usage"].get("artifact_bytes") or 0) + new_artifact_bytes > int(limits.get("max_artifact_bytes", 104_857_600)):
                raise RuntimeTaskProjectionConflict("runtime artifact bytes exceed the product limit")
            if new_artifact_bytes:
                if not delta.budget_usage:
                    raise RuntimeTaskProjectionConflict("runtime artifacts require a cumulative budget snapshot")
                incoming_budget = normalize_budget_state(delta.budget_usage, limits)
                minimum_artifact_bytes = int(current_budget["lifetime_usage"].get("artifact_bytes") or 0) + new_artifact_bytes
                if int(incoming_budget["lifetime_usage"].get("artifact_bytes") or 0) < minimum_artifact_bytes:
                    raise RuntimeTaskProjectionConflict("runtime artifact budget does not account for projected content")

            referenced_artifact_ids = {
                str(item)
                for packet in (*delta.subagent_changes, *delta.todo_changes)
                for item in (packet.get("artifact_ids") or [])
                if str(item) not in artifact_ids
            }
            boundary_artifact_ids = [
                (delta.result or {}).get("final_artifact_id"),
                ((delta.pending_interrupt or {}).get("value") or {}).get("provisional_artifact_id")
                if isinstance((delta.pending_interrupt or {}).get("value"), Mapping)
                else None,
            ]
            referenced_artifact_ids.update(
                str(item) for item in boundary_artifact_ids
                if item and str(item) not in artifact_ids
            )
            existing_artifact_ids = set()
            if referenced_artifact_ids:
                existing_artifact_ids = set((await session.execute(select(AgentTaskArtifact.id).where(
                    AgentTaskArtifact.task_id == task_id,
                    AgentTaskArtifact.id.in_(referenced_artifact_ids),
                    AgentTaskArtifact.validity == "valid",
                ))).scalars().all())
                missing = referenced_artifact_ids - existing_artifact_ids
                if missing:
                    raise RuntimeTaskProjectionConflict(f"runtime delta references unknown artifacts: {sorted(missing)}")

            def translated_artifacts(values: Any) -> list[str]:
                return [artifact_ids.get(str(item), str(item)) for item in values or []]

            def translated_value(value: Any) -> Any:
                if isinstance(value, str):
                    return artifact_ids.get(value, value)
                if isinstance(value, list):
                    return [translated_value(item) for item in value]
                if isinstance(value, tuple):
                    return [translated_value(item) for item in value]
                if isinstance(value, Mapping):
                    return {str(key): translated_value(item) for key, item in value.items()}
                return value

            todos = {row.id: row for row in (await session.execute(select(AgentTaskTodo).where(AgentTaskTodo.task_id == task_id).with_for_update())).scalars().all()}
            allowed_subagent_statuses = {"queued", "running", "completed", "failed", "cancelled", "timed_out"}
            terminal_subagent_statuses = {"completed", "failed", "cancelled", "timed_out"}
            for packet in delta.subagent_changes:
                todo_id = str(packet.get("todo_id") or "")
                if todo_id not in todos:
                    raise RuntimeTaskProjectionConflict(f"runtime delta references unknown todo {todo_id!r}")
                status = str(packet.get("status") or "failed")
                if status not in allowed_subagent_statuses:
                    raise RuntimeTaskProjectionConflict(f"runtime delta contains invalid subagent status {status!r}")
                attempt = max(1, int(packet.get("attempt") or todos[todo_id].attempt or 1))
                if attempt > int(todos[todo_id].max_attempts):
                    raise RuntimeTaskProjectionConflict("runtime subagent attempt exceeds the product limit")
                timeout_ms = int(packet.get("timeout_ms") or 180_000)
                if timeout_ms > int(limits.get("subagent_timeout_ms", 180_000)) or timeout_ms <= 0:
                    raise RuntimeTaskProjectionConflict("runtime subagent timeout exceeds the product limit")
                execution_key = tasks.canonical_hash({"task_id": task_id, "todo_id": todo_id, "plan_revision": max(1, plan_revision), "attempt": attempt})
                subagent = (await session.execute(select(AgentTaskSubagentRun).where(AgentTaskSubagentRun.execution_key == execution_key))).scalar_one_or_none()
                if subagent is None:
                    subagent = AgentTaskSubagentRun(
                        task_id=task_id, agent_run_id=agent_run_id, todo_id=todo_id, execution_key=execution_key,
                        profile_id=str(packet.get("profile_id") or todos[todo_id].profile_id), plan_revision=max(1, plan_revision),
                        attempt=attempt, status=status, usage_json=dict(packet.get("usage") or {}),
                        tool_policy_hash=str(packet.get("tool_policy_hash") or delta.idempotency_key), timeout_ms=timeout_ms,
                        output_artifact_ids_json=translated_artifacts(packet.get("artifact_ids")),
                        error_json=dict(packet["error"]) if isinstance(packet.get("error"), Mapping) else None,
                        started_at=utc_now(), completed_at=utc_now() if status in terminal_subagent_statuses else None,
                    )
                    session.add(subagent)
                    await session.flush()
                    await tasks._append_event(session, task, f"subagent.{subagent.status}", agent_run_id=agent_run_id, todo_id=todo_id, subagent_run_id=subagent.id, payload={"attempt": attempt})
                elif subagent.status not in terminal_subagent_statuses:
                    changed = subagent.status != status
                    subagent.status = status
                    replace_jsonb_field(subagent, "usage_json", dict(packet.get("usage") or subagent.usage_json or {}))
                    replace_jsonb_field(subagent, "output_artifact_ids_json", translated_artifacts(packet.get("artifact_ids")))
                    subagent.error_json = dict(packet["error"]) if isinstance(packet.get("error"), Mapping) else None
                    subagent.completed_at = utc_now() if status in terminal_subagent_statuses else None
                    if changed:
                        await tasks._append_event(session, task, f"subagent.{status}", agent_run_id=agent_run_id, todo_id=todo_id, subagent_run_id=subagent.id, payload={"attempt": attempt})

            allowed_statuses = {"pending", "ready", "running", "blocked", "completed", "failed", "skipped", "cancelled"}
            for change in delta.todo_changes:
                todo_id = str(change.get("id") or "")
                todo = todos.get(todo_id)
                if todo is None:
                    raise RuntimeTaskProjectionConflict(f"runtime delta references unknown todo {todo_id!r}")
                status = str(change.get("status") or todo.status)
                if status not in allowed_statuses:
                    raise RuntimeTaskProjectionConflict(f"runtime delta contains invalid todo status {status!r}")
                if todo.status == "completed" and status != "completed":
                    continue
                changed = status != todo.status
                todo.status = status
                todo.progress = 100 if status == "completed" else max(0, min(100, int(change.get("progress") or todo.progress)))
                todo.attempt = max(todo.attempt, int(change.get("attempt") or 0))
                todo.result_summary = str(change.get("result_summary") or todo.result_summary or "")[:12_000] or None
                replace_jsonb_field(todo, "artifact_ids_json", translated_artifacts(change.get("artifact_ids") or todo.artifact_ids_json or []))
                todo.updated_revision, todo.version, todo.updated_at = max(todo.updated_revision, plan_revision), todo.version + 1, utc_now()
                if changed:
                    await tasks._append_event(session, task, f"todo.{status}", agent_run_id=agent_run_id, todo_id=todo_id, payload={"attempt": todo.attempt, "progress": todo.progress})

            all_todos = list(todos.values())
            task.completed_todos = sum(1 for item in all_todos if item.status == "completed")
            task.total_todos = len(all_todos)
            task.progress = int(task.completed_todos * 100 / len(all_todos)) if all_todos else 0
            if delta.budget_usage:
                pending_decision = dict((run.pending_interrupt_json or {}).get("decision") or {})
                budget = _merge_budget(
                    task.budgets_json,
                    delta.budget_usage,
                    (task.config_json or {}).get("limits") or {},
                    authorized_tranche_increment=(
                        str((run.pending_interrupt_json or {}).get("response_operation") or "")
                        == "task.budget_review.respond"
                        and str(pending_decision.get("action") or "") in {"continue", "steer"}
                    ),
                )
                replace_jsonb_field(task, "budgets_json", budget)
                await tasks._append_event(session, task, "task.budget_updated", agent_run_id=agent_run_id, payload={"tranche_index": budget["tranche_index"], "tranche_usage": budget["tranche_usage"], "lifetime_usage": budget["lifetime_usage"]})
            if delta.web_access is not None:
                status, interrupt_id = str(delta.web_access.get("status") or ""), str(delta.web_access.get("interrupt_id") or "")
                pending, decision = dict(run.pending_interrupt_json or {}), dict((run.pending_interrupt_json or {}).get("decision") or {})
                if status not in {tasks.WEB_ACCESS_ALLOWED, tasks.WEB_ACCESS_DENIED} or not interrupt_id or pending.get("interrupt_id") != interrupt_id or not decision:
                    raise RuntimeTaskProjectionConflict("runtime web-access change is not backed by a product decision")
                await tasks._append_event(session, task, f"web_access.{status}", agent_run_id=agent_run_id, payload={"interrupt_id": interrupt_id, "status": status})
            if delta.pending_interrupt is not None:
                operation = str(delta.pending_interrupt.get("operation") or "")
                if operation == "set" and isinstance(delta.pending_interrupt.get("value"), Mapping):
                    pending = normalize_pending_interrupt_payload({
                        **translated_value(delta.pending_interrupt["value"]),
                        "resume_guard": run_interrupt_resume_guard(run),
                    })
                    replace_jsonb_field(run, "pending_interrupt_json", pending)
                    run.status = AgentRunStatus.AWAITING_HUMAN.value
                    task.status = AgentTaskStatus.PAUSED.value if pending.get("type") == "task_pause" else AgentTaskStatus.AWAITING_APPROVAL.value
                    task.current_phase = "checkpointed_interrupt"
                    task.paused_at = utc_now() if task.status == AgentTaskStatus.PAUSED.value else task.paused_at
                    task.expires_at = utc_now() + timedelta(days=7)
                    task.lease_owner = None
                    task.lease_expires_at = None
                    await tasks._append_event(session, task, "task.approval_requested" if task.status == AgentTaskStatus.AWAITING_APPROVAL.value else "task.paused", agent_run_id=agent_run_id, payload={"interrupt_id": pending.get("interrupt_id"), "type": pending.get("type")})
                elif operation == "clear":
                    replace_jsonb_field(run, "pending_interrupt_json", {})
                else:
                    raise RuntimeTaskProjectionConflict("runtime interrupt change must be set or clear")
            if delta.result is not None:
                if not str(delta.result.get("status") or ""):
                    raise RuntimeTaskProjectionConflict("runtime result status is required")
                metadata = dict(run.run_metadata_json or {})
                metadata["orchestration_result"] = translated_value(delta.result)
                projection = dict(metadata.get("projection") or {})
                projection.update({
                    "status": "applied",
                    "delta_event_id": delta.event_id,
                    "operation_id": delta.operation_id,
                    "applied_runtime_plan_revision": applied_runtime_plan_revision,
                })
                projection.pop("projection_error", None)
                metadata["projection"] = projection
                replace_jsonb_field(run, "run_metadata_json", metadata)

            if applied_correction_ids:
                now = utc_now()
                commands = list((await session.execute(select(AgentTaskCommand).where(
                    AgentTaskCommand.task_id == task_id,
                    AgentTaskCommand.action == "steer",
                    AgentTaskCommand.status == "accepted",
                ).with_for_update())).scalars().all())
                completed_corrections: list[str] = []
                for command in commands:
                    command_result = dict(command.result_json or {})
                    correction = dict(command_result.get("correction") or {})
                    correction_id = str(correction.get("correction_id") or correction.get("id") or "")
                    if correction_id not in applied_correction_ids:
                        continue
                    correction.update({"status": "applied", "applied_at": now.isoformat(), "plan_revision": plan_revision})
                    command_result.update({"correction": correction, "delivery_state": "applied", "plan_revision": plan_revision})
                    replace_jsonb_field(command, "result_json", command_result)
                    command.status = "completed"
                    command.result_version = task.version + 1
                    command.completed_at = now
                    completed_corrections.append(correction_id)
                if completed_corrections:
                    if task.current_phase == "budget_correction_delivery_pending":
                        task.current_phase = "budget_continuation_queued"
                    await tasks._append_event(
                        session, task, "task.course_correction_applied", agent_run_id=agent_run_id,
                        payload={"correction_ids": sorted(completed_corrections), "plan_revision": plan_revision},
                    )

            task.version, task.updated_at = task.version + 1, utc_now()
            await session.flush()
            session.add(AgentTaskRuntimeDelta(
                task_id=task_id, agent_run_id=agent_run_id, attempt_id=delta.attempt_id,
                operation_id=delta.operation_id, event_id=delta.event_id, idempotency_key=delta.idempotency_key,
                payload_sha256=payload_sha256, observed_task_version=delta.observed_task_version,
                observed_plan_revision=delta.observed_plan_revision, applied_task_version=task.version,
                applied_plan_revision=plan_revision,
                applied_runtime_plan_revision=applied_runtime_plan_revision,
            ))
    return artifact_ids
