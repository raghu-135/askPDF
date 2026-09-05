"""Atomically apply framework-neutral runtime task deltas to product records."""

from __future__ import annotations

import hashlib
import json
import uuid
from contextvars import ContextVar
from functools import wraps
from datetime import timedelta
from typing import Any, Mapping

from sqlalchemy import func
from sqlalchemy.future import select

from app.agent_workflows.interrupts import normalize_pending_interrupt_payload, run_interrupt_resume_guard
from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import AgentRunStatus
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import (
    AgentRun, AgentRuntimeOperation, AgentTask, AgentTaskArtifact, AgentTaskCommand, AgentTaskPlanRevision,
    AgentTaskRuntimeDelta, AgentTaskSubagentRun, AgentTaskTodo,
)
from app.models.deep_research import AgentTaskStatus, DeepResearchPlanProposal
from app.services import agent_task_repository as tasks
from app.services.agent_task_budgets import exhausted_dimensions, normalize_budget_state
from app.services.content_store import get_content_store, task_artifact_content_key
from app.time_utils import utc_now
from runtime_protocol.contracts import TaskOrchestrationDelta


class RuntimeTaskProjectionConflict(RuntimeError):
    """The runtime delta is stale, malformed, or causally inconsistent."""


_staged_content_keys: ContextVar[set[str] | None] = ContextVar("runtime_staged_content_keys", default=None)


async def _cleanup_staged_content() -> None:
    keys = _staged_content_keys.get() or set()
    if not keys:
        return
    store = get_content_store()
    for key in keys:
        try:
            await store.delete(key)
        except Exception:
            # Cleanup is retried by the normal content-store sweeper; projection
            # failures must not be hidden by a secondary storage failure.
            continue
    _staged_content_keys.set(set())


def _cleanup_projection_on_error(function):
    @wraps(function)
    async def wrapped(*args, **kwargs):
        token = _staged_content_keys.set(set())
        try:
            return await function(*args, **kwargs)
        except Exception:
            await _cleanup_staged_content()
            raise
        finally:
            _staged_content_keys.reset(token)
    return wrapped


def _register_staged_content(key: str) -> None:
    keys = _staged_content_keys.get()
    if keys is not None:
        keys.add(key)


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
        if await store.exists(object_key):
            existing = await store.stat(object_key)
            if existing.sha256 != digest or existing.size != len(body):
                raise RuntimeTaskProjectionConflict("runtime artifact object conflicts with existing content")
        else:
            created, existing = await store.put_if_absent(object_key, body, expected_sha256=digest)
            if not created and (existing.sha256 != digest or existing.size != len(body)):
                raise RuntimeTaskProjectionConflict("runtime artifact object conflicts with existing content")
            if created:
                _register_staged_content(object_key)
        staged.append({**dict(source), "runtime_id": runtime_id, "id": artifact_id, "object_key": object_key, "sha256": digest, "byte_size": len(body)})
    return staged


async def _stage_final_report(
    *, task_id: str, agent_run_id: str, operation_id: str, text: str
) -> dict[str, Any] | None:
    body = text.strip().encode("utf-8")
    if not body:
        return None
    digest = hashlib.sha256(body).hexdigest()
    artifact_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"askpdf:{task_id}:{agent_run_id}:final:{operation_id}:{digest}",
    ))
    object_key = task_artifact_content_key(task_id, agent_run_id, artifact_id)
    created, existing = await get_content_store().put_if_absent(object_key, body, expected_sha256=digest)
    if not created and (existing.sha256 != digest or existing.size != len(body)):
        raise RuntimeTaskProjectionConflict("final report object conflicts with existing content")
    if created:
        _register_staged_content(object_key)
    return {
        "id": artifact_id,
        "object_key": object_key,
        "sha256": digest,
        "byte_size": len(body),
    }


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
    # Runtime budget payloads are cumulative snapshots. A late event or a
    # terminal result can legitimately carry an older snapshot than one that
    # was already projected, especially after a failed graph invocation. Do
    # not regress product accounting in that case; merge counters
    # component-wise by maximum. Identity, task-version, plan-revision, and
    # tranche-transition conflicts remain strict guards above.
    lifetime = dict(candidate.get("lifetime_usage") or {})
    for key, old_value in (existing.get("lifetime_usage") or {}).items():
        lifetime[key] = max(int(old_value or 0), int(lifetime.get(key) or 0))
    candidate["lifetime_usage"] = lifetime
    if new_tranche == old_tranche:
        tranche = dict(candidate.get("tranche_usage") or {})
        for key, old_value in (existing.get("tranche_usage") or {}).items():
            tranche[key] = max(int(old_value or 0), int(tranche.get(key) or 0))
        candidate["tranche_usage"] = tranche
    candidate["tranche_limits"] = existing["tranche_limits"]
    return candidate


def _incomplete_disposition(run: AgentRun, task_result: Mapping[str, Any]) -> tuple[str, int, int]:
    orchestration = dict(
        (dict((run.resolved_spec_json or {}).get("config") or {}).get("task_policy") or {}).get("orchestration")
        or {}
    )
    policy = str(orchestration.get("incomplete_result_policy") or "review")
    max_rounds = min(5, max(0, int(orchestration.get("max_incomplete_review_rounds") or 3)))
    return policy, max_rounds, max(1, int(run.task_attempt or 1))


@_cleanup_projection_on_error
async def apply_neutral_task_completion(
    *,
    task_id: str,
    agent_run_id: str,
    operation_id: str,
    runtime_status: str,
    task_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically project a non-delta runtime's usage and product disposition."""

    usage = dict(task_result.get("usage") or {})
    canonical_usage = {
        "operation_id": str(usage.get("operation_id") or operation_id),
        "model_tokens": max(0, int(usage.get("model_tokens") or 0)),
        "model_calls": max(0, int(usage.get("model_calls") or 0)) if usage.get("model_calls") is not None else None,
        "tool_calls": max(0, int(usage.get("tool_calls") or 0)) if usage.get("tool_calls") is not None else None,
        "active_runtime_ms": max(0, int(usage.get("active_runtime_ms") or 0)) if usage.get("active_runtime_ms") is not None else None,
        "measured_dimensions": sorted(set(str(value) for value in usage.get("measured_dimensions") or [])),
        "cumulative": True,
    }
    if canonical_usage["operation_id"] != operation_id:
        raise RuntimeTaskProjectionConflict("runtime usage targets a different operation")
    fingerprint = hashlib.sha256(json.dumps(
        {"status": runtime_status, "task_result": dict(task_result), "usage": canonical_usage},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str,
    ).encode()).hexdigest()
    final_stage = None

    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(
                AgentTask.id == task_id,
            ).with_for_update())).scalar_one()
            run = (await session.execute(select(AgentRun).where(
                AgentRun.id == agent_run_id, AgentRun.task_id == task_id,
            ).with_for_update())).scalar_one()
            operation = (await session.execute(select(AgentRuntimeOperation).where(
                AgentRuntimeOperation.run_id == agent_run_id,
                AgentRuntimeOperation.operation == "task.completion.project",
                AgentRuntimeOperation.idempotency_key == operation_id,
            ).with_for_update())).scalar_one_or_none()
            if operation is not None and operation.status == "completed":
                if operation.request_fingerprint != fingerprint:
                    raise RuntimeTaskProjectionConflict("runtime completion identity was replayed with different content")
                return dict(operation.result_json or {})
            now = utc_now()
            if operation is None:
                operation = AgentRuntimeOperation(
                    run_id=agent_run_id,
                    operation="task.completion.project",
                    idempotency_key=operation_id,
                    request_fingerprint=fingerprint,
                    status="in_progress",
                    claimed_at=now,
                    claim_expires_at=now + timedelta(minutes=5),
                )
                session.add(operation)
            elif operation.request_fingerprint != fingerprint:
                raise RuntimeTaskProjectionConflict("runtime completion identity was replayed with different content")

            final_stage = await _stage_final_report(
                task_id=task_id, agent_run_id=agent_run_id, operation_id=operation_id,
                text=str(task_result.get("text") or ""),
            ) if runtime_status == "completed" else None

            limits = dict((task.config_json or {}).get("limits") or {})
            budget = normalize_budget_state(task.budgets_json, limits)
            dimension_map = {
                "model_tokens": "model_tokens", "model_calls": "model_calls",
                "tool_calls": "tool_calls", "active_runtime_ms": "elapsed_active_ms",
            }
            for source_key in canonical_usage["measured_dimensions"]:
                target_key = dimension_map.get(source_key)
                value = canonical_usage.get(source_key)
                if target_key is None or value is None:
                    continue
                budget["tranche_usage"][target_key] = int(budget["tranche_usage"].get(target_key) or 0) + int(value)
                budget["lifetime_usage"][target_key] = int(budget["lifetime_usage"].get(target_key) or 0) + int(value)

            final_artifact_id: str | None = None
            if final_stage is not None:
                existing_final = (await session.execute(select(AgentTaskArtifact).where(
                    AgentTaskArtifact.agent_run_id == agent_run_id,
                    AgentTaskArtifact.kind == "final_report",
                    AgentTaskArtifact.validity == "valid",
                    AgentTaskArtifact.deleted_at.is_(None),
                ).with_for_update())).scalar_one_or_none()
                if existing_final is not None:
                    if existing_final.sha256 != final_stage["sha256"]:
                        raise RuntimeTaskProjectionConflict("runtime completion returned conflicting final reports")
                    final_artifact_id = existing_final.id
                else:
                    artifact_count = int((await session.execute(select(func.count(AgentTaskArtifact.id)).where(
                        AgentTaskArtifact.task_id == task_id,
                        AgentTaskArtifact.validity != "deleted",
                    ))).scalar_one())
                    if artifact_count >= int(limits.get("max_artifacts", 200)):
                        raise RuntimeTaskProjectionConflict("final report exceeds the product artifact count limit")
                    if final_stage["byte_size"] > int(limits.get("max_single_artifact_bytes", 10_485_760)):
                        raise RuntimeTaskProjectionConflict("final report exceeds the product per-object limit")
                    if int(budget["lifetime_usage"].get("artifact_bytes") or 0) + final_stage["byte_size"] > int(limits.get("max_artifact_bytes", 104_857_600)):
                        raise RuntimeTaskProjectionConflict("final report exceeds the product artifact byte limit")
                    budget["lifetime_usage"]["artifact_bytes"] = int(
                        budget["lifetime_usage"].get("artifact_bytes") or 0
                    ) + final_stage["byte_size"]
                    artifact = AgentTaskArtifact(
                        id=final_stage["id"], task_id=task_id, agent_run_id=agent_run_id,
                        ownership_key=f"runtime-final:{operation_id}", kind="final_report",
                        object_key=final_stage["object_key"], media_type="text/plain",
                        byte_size=final_stage["byte_size"], sha256=final_stage["sha256"],
                        provenance_json={
                            "runtime_operation_id": operation_id,
                            "warnings": list(task_result.get("warnings") or []),
                            "gaps": list(task_result.get("gaps") or []),
                            "outcome": task_result.get("status"),
                        },
                        source_refs_json={}, retention_until=now + timedelta(days=30),
                    )
                    session.add(artifact)
                    final_artifact_id = artifact.id
                    await tasks._append_event(
                        session, task, "artifact.created", agent_run_id=agent_run_id,
                        artifact_id=artifact.id,
                        payload={"kind": "final_report", "byte_size": artifact.byte_size, "sha256": artifact.sha256},
                    )

            correction_outcomes = {
                str(value.get("correction_id")): dict(value)
                for value in task_result.get("correction_outcomes") or []
                if isinstance(value, Mapping) and value.get("correction_id")
            }
            linked_corrections = [
                dict(value) for value in (run.run_metadata_json or {}).get("course_corrections") or []
                if isinstance(value, Mapping)
            ]
            if linked_corrections:
                commands = list((await session.execute(select(AgentTaskCommand).where(
                    AgentTaskCommand.task_id == task_id,
                    AgentTaskCommand.action == "steer",
                    AgentTaskCommand.status == "accepted",
                ).with_for_update())).scalars().all())
                commands_by_correction = {
                    str((value.result_json or {}).get("correction", {}).get("correction_id") or
                        (value.result_json or {}).get("correction", {}).get("id") or ""): value
                    for value in commands
                }
                expected = {
                    str(value.get("correction_id") or value.get("id") or "")
                    for value in linked_corrections
                }
                if expected - set(correction_outcomes):
                    raise RuntimeTaskProjectionConflict(
                        "Hermes terminal result omitted correction coverage"
                    )
                for correction_id in expected:
                    command = commands_by_correction.get(correction_id)
                    if command is None:
                        raise RuntimeTaskProjectionConflict(
                            "Hermes result references a correction that is not active"
                        )
                    outcome = correction_outcomes[correction_id]
                    if outcome.get("state") not in {"satisfied", "unresolved"}:
                        raise RuntimeTaskProjectionConflict("Hermes correction outcome is invalid")
                    command_result = dict(command.result_json or {})
                    correction = dict(command_result.get("correction") or {})
                    correction.update({"status": "incorporated", "linked_run_id": agent_run_id})
                    command_result.update({
                        "correction": correction, "delivery_state": "incorporated",
                        "runtime_outcome": outcome,
                    })
                    await tasks._append_event(
                        session, task, "task.course_correction_incorporated", agent_run_id=agent_run_id,
                        payload={"correction_ids": [correction_id], "linked_run_id": agent_run_id},
                    )
                    if outcome["state"] == "satisfied":
                        correction.update({"status": "satisfied", "satisfied_at": now.isoformat()})
                        command_result["delivery_state"] = "satisfied"
                        command.status = "completed"
                        command.completed_at = now
                        command.result_version = task.version + 1
                    else:
                        correction["status"] = "unresolved"
                        command_result["delivery_state"] = "unresolved"
                    command_result["correction"] = correction
                    replace_jsonb_field(command, "result_json", command_result)
                    await tasks._append_event(
                        session, task, f"task.course_correction_{outcome['state']}", agent_run_id=agent_run_id,
                        payload={"correction_ids": [correction_id], "linked_run_id": agent_run_id},
                    )

            dimensions = exhausted_dimensions(budget)
            budget["boundary"] = {
                "dimensions": dimensions, "operation_id": operation_id, "observed_at": now.isoformat(),
            } if dimensions else None
            replace_jsonb_field(task, "budgets_json", budget)
            await tasks._append_event(
                session, task, "task.budget_updated", agent_run_id=agent_run_id,
                payload={
                    "operation_id": operation_id, "tranche_index": budget["tranche_index"],
                    "tranche_usage": budget["tranche_usage"], "lifetime_usage": budget["lifetime_usage"],
                    "measured_dimensions": canonical_usage["measured_dimensions"],
                },
            )

            warnings = [dict(value) for value in task_result.get("warnings") or [] if isinstance(value, Mapping)]
            gaps = list(dict.fromkeys(str(value) for value in task_result.get("gaps") or [] if str(value).strip()))
            disposition = runtime_status
            pending: dict[str, Any] | None = None
            if runtime_status == "completed" and dimensions:
                disposition = "budget_review"
                pending = {
                    "interrupt_id": f"budget-review:{agent_run_id}:{budget['tranche_index']}",
                    "type": "budget_review", "response_operation": "task.budget_review.respond",
                    "status": "pending", "title": "Research budget reached",
                    "allowed_actions": ["continue", "accept_partial", "steer"],
                    "continuation_semantics": "linked_run", "preserves_run_id": False,
                    "provisional_artifact_id": final_artifact_id,
                    "provisional_answer": str(task_result.get("text") or ""),
                    "warnings": warnings, "gaps": gaps,
                    "usage": {**budget, "exhausted_dimensions": dimensions},
                }
                event_type = "task.budget_review_requested"
                phase = "budget_review"
            elif runtime_status == "completed":
                policy, max_rounds, review_round = _incomplete_disposition(run, task_result)
                needs_review = bool(warnings or gaps or task_result.get("status") == "completed_with_warnings")
                if needs_review and policy == "review" and review_round <= max_rounds:
                    disposition = "result_review"
                    pending = {
                        "interrupt_id": f"result-review:{agent_run_id}:{operation_id}",
                        "type": "incomplete_result_review", "kind": "approval", "status": "pending",
                        "response_operation": "task.result_review.respond",
                        "title": "Review incomplete result",
                        "allowed_actions": ["accept", "retry_with_input"],
                        "provisional_artifact_id": final_artifact_id,
                        "provisional_answer": str(task_result.get("text") or ""),
                        "warnings": warnings, "gaps": gaps,
                        "review_round": review_round, "max_review_rounds": max_rounds,
                    }
                    event_type = "task.result_review_requested"
                    phase = "awaiting_result_review"
                elif needs_review and policy == "fail":
                    disposition = "failed"

            if pending is not None:
                replace_jsonb_field(run, "pending_interrupt_json", pending)
                run.status = AgentRunStatus.AWAITING_HUMAN.value
                task.status = AgentTaskStatus.AWAITING_APPROVAL.value
                task.current_phase = phase
                task.terminal_reason = "budget_exhausted" if disposition == "budget_review" else "incomplete_result"
                task.completed_at = None
                task.expires_at = now + timedelta(days=7)
                task.lease_owner = None
                task.lease_expires_at = None
                await tasks._append_event(
                    session, task, event_type, agent_run_id=agent_run_id,
                    artifact_id=final_artifact_id,
                    causal_key=f"run:{agent_run_id}:{disposition}:{budget['tranche_index'] if disposition == 'budget_review' else operation_id}",
                    payload={
                        "interrupt_id": pending["interrupt_id"], "warnings": warnings,
                        "gaps": gaps, "exhausted_dimensions": dimensions,
                    },
                )
            else:
                cancelled = runtime_status in {"cancelled", "canceled"}
                failed = runtime_status == "failed" or disposition == "failed"
                run.status = AgentRunStatus.CANCELLED.value if cancelled else AgentRunStatus.FAILED.value if failed else AgentRunStatus.COMPLETED.value
                run.completed_at = now
                task.status = AgentTaskStatus.CANCELLED.value if cancelled else AgentTaskStatus.FAILED.value if failed else AgentTaskStatus.COMPLETED.value
                task.current_phase = task.status
                task.terminal_reason = "incomplete_result_rejected" if disposition == "failed" and runtime_status == "completed" else runtime_status
                task.completed_at = now
                task.expires_at = None
                task.lease_owner = None
                task.lease_expires_at = None
                await tasks._append_event(
                    session, task, f"task.{task.status}", agent_run_id=agent_run_id,
                    artifact_id=final_artifact_id,
                    payload={"reason": task.terminal_reason, "operation_id": operation_id},
                )

            metadata = dict(run.run_metadata_json or {})
            projection = dict(metadata.get("projection") or {})
            projection.update({
                "status": "applied", "reconciliation_status": "projected",
                "operation_id": operation_id, "usage_fingerprint": fingerprint,
                "final_artifact_id": final_artifact_id, "product_disposition": disposition,
            })
            projection.pop("projection_error", None)
            metadata["projection"] = projection
            replace_jsonb_field(run, "run_metadata_json", metadata)
            task.version += 1
            result = {
                "usage_fingerprint": fingerprint, "usage": canonical_usage,
                "final_artifact_id": final_artifact_id, "product_disposition": disposition,
                "exhausted_dimensions": dimensions,
            }
            operation.status = "completed"
            replace_jsonb_field(operation, "result_json", result)
            operation.completed_at = now
            return result


@_cleanup_projection_on_error
async def apply_runtime_task_delta(
    *, task_id: str, agent_run_id: str, delta: TaskOrchestrationDelta,
    artifact_id_map: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply one runtime boundary exactly once in one product DB transaction."""

    staged: list[dict[str, Any]] = []
    result_envelope = dict(delta.result or {})
    task_result = dict(result_envelope.get("task_result") or {}) if isinstance(result_envelope.get("task_result"), Mapping) else {}
    if task_result:
        outer_warnings = [dict(value) for value in result_envelope.get("warnings") or [] if isinstance(value, Mapping)]
        outer_gaps = [str(value) for value in result_envelope.get("incomplete_reasons") or []]
        if outer_warnings != [dict(value) for value in task_result.get("warnings") or [] if isinstance(value, Mapping)]:
            raise RuntimeTaskProjectionConflict("runtime result warnings disagree with the canonical task result")
        if outer_gaps != [str(value) for value in task_result.get("gaps") or []]:
            raise RuntimeTaskProjectionConflict("runtime result gaps disagree with the canonical task result")
        if result_envelope.get("result_outcome") != task_result.get("status"):
            raise RuntimeTaskProjectionConflict("runtime result outcome disagrees with the canonical task result")
        task_outcomes = [
            dict(value) for value in task_result.get("correction_outcomes") or []
            if isinstance(value, Mapping)
        ]
        if task_outcomes != [value.to_dict() for value in delta.correction_outcomes]:
            raise RuntimeTaskProjectionConflict(
                "runtime delta correction outcomes disagree with the canonical task result"
            )
    final_stage = None
    artifact_ids = {str(key): str(value) for key, value in (artifact_id_map or {}).items()}
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

            staged = await _stage_artifacts(
                task_id=task_id, agent_run_id=agent_run_id, artifacts=delta.artifacts
            ) if delta.artifacts else []
            artifact_ids.update({str(value["runtime_id"]): str(value["id"]) for value in staged})
            final_stage = await _stage_final_report(
                task_id=task_id,
                agent_run_id=agent_run_id,
                operation_id=delta.operation_id,
                text=str(task_result.get("text") or ""),
            ) if str(result_envelope.get("status") or "") == "completed" else None

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
            referenced_artifact_ids.update(
                str(item)
                for outcome in delta.correction_outcomes
                for item in outcome.artifact_ids
                if str(item) not in artifact_ids
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
            final_artifact_id: str | None = None
            if final_stage is not None:
                existing_final = (await session.execute(select(AgentTaskArtifact).where(
                    AgentTaskArtifact.agent_run_id == agent_run_id,
                    AgentTaskArtifact.kind == "final_report",
                    AgentTaskArtifact.validity == "valid",
                    AgentTaskArtifact.deleted_at.is_(None),
                ).with_for_update())).scalar_one_or_none()
                if existing_final is not None:
                    if existing_final.sha256 != final_stage["sha256"]:
                        raise RuntimeTaskProjectionConflict("runtime operation returned conflicting final reports")
                    final_artifact_id = existing_final.id
                else:
                    if artifact_count >= int(limits.get("max_artifacts", 200)):
                        raise RuntimeTaskProjectionConflict("final report exceeds the product artifact count limit")
                    if final_stage["byte_size"] > int(limits.get("max_single_artifact_bytes", 10_485_760)):
                        raise RuntimeTaskProjectionConflict("final report exceeds the product per-object limit")
                    budget = normalize_budget_state(task.budgets_json, limits)
                    if int(budget["lifetime_usage"].get("artifact_bytes") or 0) + final_stage["byte_size"] > int(limits.get("max_artifact_bytes", 104_857_600)):
                        raise RuntimeTaskProjectionConflict("final report exceeds the product artifact byte limit")
                    budget["lifetime_usage"]["artifact_bytes"] = int(
                        budget["lifetime_usage"].get("artifact_bytes") or 0
                    ) + final_stage["byte_size"]
                    replace_jsonb_field(task, "budgets_json", budget)
                    final_artifact = AgentTaskArtifact(
                        id=final_stage["id"], task_id=task_id, agent_run_id=agent_run_id,
                        ownership_key=f"runtime-final:{delta.operation_id}", kind="final_report",
                        object_key=final_stage["object_key"], media_type="text/plain",
                        byte_size=final_stage["byte_size"], sha256=final_stage["sha256"],
                        provenance_json={
                            "runtime_operation_id": delta.operation_id,
                            "warnings": list(task_result.get("warnings") or []),
                            "gaps": list(task_result.get("gaps") or []),
                            "outcome": task_result.get("status"),
                        },
                        source_refs_json={}, retention_until=utc_now() + timedelta(days=30),
                    )
                    session.add(final_artifact)
                    final_artifact_id = final_artifact.id
                    artifact_count += 1
                    await tasks._append_event(
                        session, task, "artifact.created", agent_run_id=agent_run_id,
                        artifact_id=final_artifact.id,
                        payload={"kind": "final_report", "byte_size": final_artifact.byte_size, "sha256": final_artifact.sha256},
                    )
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
                    "reconciliation_status": "projected",
                    "delta_event_id": delta.event_id,
                    "operation_id": delta.operation_id,
                    "applied_runtime_plan_revision": applied_runtime_plan_revision,
                    "final_artifact_id": final_artifact_id,
                })
                projection.pop("projection_error", None)
                metadata["projection"] = projection
                replace_jsonb_field(run, "run_metadata_json", metadata)

                result_status = str(delta.result.get("status") or "")
                if result_status == "completed" and task_result:
                    warnings = [dict(value) for value in task_result.get("warnings") or [] if isinstance(value, Mapping)]
                    gaps = list(dict.fromkeys(str(value) for value in task_result.get("gaps") or [] if str(value).strip()))
                    policy, max_rounds, review_round = _incomplete_disposition(run, task_result)
                    needs_review = bool(warnings or gaps or task_result.get("status") == "completed_with_warnings")
                    if needs_review and policy == "review" and review_round <= max_rounds:
                        pending = {
                            "interrupt_id": f"result-review:{run.id}:{delta.operation_id}",
                            "type": "incomplete_result_review",
                            "kind": "approval",
                            "status": "pending",
                            "title": "Review incomplete result",
                            "body": "The agent returned usable output with warnings or unresolved gaps.",
                            "response_operation": "task.result_review.respond",
                            "allowed_actions": ["accept", "retry_with_input"],
                            "response_schema": {"type": "object", "properties": {"followup_input": {"type": "string", "maxLength": 20000}}},
                            "review_round": review_round,
                            "max_review_rounds": max_rounds,
                            "provisional_artifact_id": final_artifact_id,
                            "provisional_answer": str(task_result.get("text") or ""),
                            "warnings": warnings,
                            "gaps": gaps,
                        }
                        replace_jsonb_field(run, "pending_interrupt_json", pending)
                        run.status = AgentRunStatus.AWAITING_HUMAN.value
                        task.status = AgentTaskStatus.AWAITING_APPROVAL.value
                        task.current_phase = "awaiting_result_review"
                        task.terminal_reason = "incomplete_result"
                        task.completed_at = None
                        task.expires_at = utc_now() + timedelta(days=7)
                        task.lease_owner = None
                        task.lease_expires_at = None
                        await tasks._append_event(
                            session, task, "task.result_review_requested", agent_run_id=agent_run_id,
                            artifact_id=final_artifact_id,
                            causal_key=f"run:{agent_run_id}:result-review:{review_round}",
                            payload={
                                "interrupt_id": pending["interrupt_id"], "result_outcome": "completed_with_warnings",
                                "warnings": warnings, "gaps": gaps, "review_round": review_round,
                            },
                        )
                    else:
                        now = utc_now()
                        rejected = needs_review and policy == "fail"
                        run.status = AgentRunStatus.FAILED.value if rejected else AgentRunStatus.COMPLETED.value
                        run.completed_at = now
                        task.status = AgentTaskStatus.FAILED.value if rejected else AgentTaskStatus.COMPLETED.value
                        task.current_phase = task.status
                        task.terminal_reason = "incomplete_result_rejected" if rejected else "completed_with_warnings" if needs_review else "completed"
                        task.completed_at = now
                        task.expires_at = None
                        task.lease_owner = None
                        task.lease_expires_at = None
                        await tasks._append_event(
                            session, task, f"task.{task.status}", agent_run_id=agent_run_id,
                            artifact_id=final_artifact_id,
                            payload={"reason": task.terminal_reason, "runtime_event_id": delta.event_id},
                        )
                elif result_status in {"failed", "cancelled", "canceled"}:
                    now = utc_now()
                    cancelled = result_status in {"cancelled", "canceled"}
                    run.status = AgentRunStatus.CANCELLED.value if cancelled else AgentRunStatus.FAILED.value
                    run.completed_at = now
                    task.status = AgentTaskStatus.CANCELLED.value if cancelled else AgentTaskStatus.FAILED.value
                    task.current_phase = task.status
                    task.terminal_reason = str((delta.result.get("error") or {}).get("code") or result_status)
                    task.completed_at = now
                    task.expires_at = None
                    task.lease_owner = None
                    task.lease_expires_at = None
                    await tasks._append_event(
                        session, task, f"task.{task.status}", agent_run_id=agent_run_id,
                        payload={"reason": task.terminal_reason, "runtime_event_id": delta.event_id},
                    )

            correction_outcomes = {value.correction_id: value for value in delta.correction_outcomes}
            correction_ids_to_lock = applied_correction_ids | set(correction_outcomes)
            if correction_ids_to_lock:
                now = utc_now()
                commands = list((await session.execute(select(AgentTaskCommand).where(
                    AgentTaskCommand.task_id == task_id,
                    AgentTaskCommand.action == "steer",
                    AgentTaskCommand.status == "accepted",
                ).with_for_update())).scalars().all())
                incorporated_corrections: list[str] = []
                satisfied_corrections: list[str] = []
                unresolved_corrections: list[str] = []
                known_corrections: set[str] = set()
                for command in commands:
                    command_result = dict(command.result_json or {})
                    correction = dict(command_result.get("correction") or {})
                    correction_id = str(correction.get("correction_id") or correction.get("id") or "")
                    known_corrections.add(correction_id)
                    if correction_id not in correction_ids_to_lock:
                        continue
                    if correction_id in applied_correction_ids:
                        correction.update({
                            "status": "incorporated", "incorporated_at": now.isoformat(),
                            "plan_revision": plan_revision,
                        })
                        command_result.update({
                            "correction": correction, "delivery_state": "incorporated",
                            "plan_revision": plan_revision,
                        })
                        incorporated_corrections.append(correction_id)
                    outcome = correction_outcomes.get(correction_id)
                    if outcome is not None:
                        if outcome.state == "satisfied" and correction.get("status") != "incorporated":
                            raise RuntimeTaskProjectionConflict(
                                "runtime cannot satisfy a correction it did not incorporate"
                            )
                        missing_todos = set(outcome.todo_ids) - valid_todo_ids
                        missing_artifacts = {
                            value for value in outcome.artifact_ids
                            if value not in artifact_ids and value not in existing_artifact_ids
                        }
                        if missing_todos or missing_artifacts:
                            raise RuntimeTaskProjectionConflict(
                                "runtime correction outcome references unknown product state"
                            )
                        outcome_payload = translated_value(outcome.to_dict())
                        command_result["runtime_outcome"] = outcome_payload
                        if outcome.state == "satisfied":
                            correction.update({"status": "satisfied", "satisfied_at": now.isoformat()})
                            command_result["delivery_state"] = "satisfied"
                            command.status = "completed"
                            command.result_version = task.version + 1
                            command.completed_at = now
                            satisfied_corrections.append(correction_id)
                        elif outcome.state == "unresolved":
                            correction.update({"status": "unresolved"})
                            command_result["delivery_state"] = "unresolved"
                            unresolved_corrections.append(correction_id)
                    command_result["correction"] = correction
                    replace_jsonb_field(command, "result_json", command_result)
                unknown = correction_ids_to_lock - known_corrections
                if unknown:
                    raise RuntimeTaskProjectionConflict(
                        f"runtime delta references unknown corrections: {sorted(unknown)}"
                    )
                if incorporated_corrections:
                    if task.current_phase == "budget_correction_delivery_pending":
                        task.current_phase = "budget_continuation_queued"
                    await tasks._append_event(
                        session, task, "task.course_correction_incorporated", agent_run_id=agent_run_id,
                        payload={"correction_ids": sorted(incorporated_corrections), "plan_revision": plan_revision},
                    )
                for state, values in (("satisfied", satisfied_corrections), ("unresolved", unresolved_corrections)):
                    if values:
                        await tasks._append_event(
                            session, task, f"task.course_correction_{state}", agent_run_id=agent_run_id,
                            payload={"correction_ids": sorted(values), "runtime_event_id": delta.event_id},
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
