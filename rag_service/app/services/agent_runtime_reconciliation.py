"""Control-plane recovery for uncertain external-runtime outcomes."""

from __future__ import annotations

import hashlib
import json
import argparse
import asyncio
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Mapping

from app.agent_workflows.repository import AgentWorkflowRepository
from app.time_utils import utc_now
from runtime_protocol.contracts import AgentRuntimeRequest, RuntimeCourseCorrection
from runtime_protocol.errors import RuntimeError as AgentRuntimeError


def result_hash(result: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(result), sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


async def record_runtime_event(run: Any, event: Any) -> bool:
    """Persist the last neutral event idempotently in existing run metadata."""

    metadata = dict(getattr(run, "run_metadata_json", None) or {})
    projection = dict(metadata.get("projection") or {})
    event_id = str(getattr(event, "event_id", None) or "")
    if event_id and event_id == projection.get("last_event_id"):
        return False
    projection.update({
        "status": projection.get("status") or "pending",
        "last_event_id": event_id,
        "last_event_sequence": int(getattr(event, "sequence", 0) or 0),
        "terminal_event_id": event_id if getattr(event, "terminal", False) else projection.get("terminal_event_id"),
    })
    await AgentWorkflowRepository().update_runtime_projection(run.id, projection)
    return True


async def record_terminal_result(run: Any, result: Mapping[str, Any], *, terminal_event_id: str | None = None) -> dict[str, Any]:
    """Record a bounded terminal snapshot before product projection."""

    repository = AgentWorkflowRepository()
    fresh_run = await repository.get_run(str(run.id))
    metadata = dict(
        getattr(fresh_run, "run_metadata_json", None)
        or getattr(run, "run_metadata_json", None)
        or {}
    )
    projection = dict(metadata.get("projection") or {})
    digest = result_hash(result)
    existing = projection.get("result_hash")
    if existing and existing != digest:
        raise ValueError("runtime_terminal_result_conflict")
    projection.update({
        "status": projection.get("status") or "pending",
        "result_hash": digest,
        "terminal_event_id": terminal_event_id or projection.get("terminal_event_id"),
        "runtime_result": dict(result),
    })
    await repository.update_runtime_projection(run.id, projection)
    return projection


async def reconcile_known_result(run: Any, result: Mapping[str, Any] | None, projector: Any) -> Any:
    """Project a known result without changing paused or terminal identity."""

    if not result:
        return run
    status = str(result.get("status") or "")
    if status in {"awaiting_human", "paused", "running", "no_continuation"}:
        return run
    await record_terminal_result(run, result, terminal_event_id=result.get("terminal_event_id"))
    return await projector.reconcile_run(run=run, result=result)


async def reconcile_request(run: Any, adapter: Any, request: AgentRuntimeRequest, projector: Any) -> Any:
    """Inspect an uncertain runtime continuation and preserve resumability."""

    inspection = await adapter.inspect_state(request)
    if inspection.get("continuation_available"):
        return run
    projection = dict((getattr(run, "run_metadata_json", None) or {}).get("projection") or {})
    result = projection.get("runtime_result")
    return await reconcile_known_result(run, result, projector)


async def reconcile_run_by_id(run_id: str, *, dry_run: bool = False) -> str:
    """Reconcile one persisted run without creating a replacement run."""
    from app.agent_workflows.repository import AgentWorkflowRepository
    from app.runtime.catalog import continuation_from_run, definition_from_run
    from app.runtime.registry import get_runtime_registry
    from app.runtime.adapter import RuntimeInvocationContext
    from app.services.agent_runtime_projection import AgentRuntimeProjection
    from app.services import agent_task_repository as tasks
    from app.services.agent_run_cancellation import confirm_task_cancellation, request_task_cancellation

    repository = AgentWorkflowRepository()
    run = await repository.get_run(run_id)
    if run is None:
        return "missing"
    projection = dict((run.run_metadata_json or {}).get("projection") or {})
    if dry_run:
        return "candidate"
    definition = definition_from_run(run)
    adapter = get_runtime_registry().get(definition)
    result = projection.get("runtime_result")
    request = AgentRuntimeRequest(
        run_id=str(run.id),
        thread_id=str(run.thread_id),
        definition_id=definition.definition_id,
        framework=definition.framework,
        builder_id=definition.builder_id,
        input={"question": str((result or {}).get("question") or "")},
        task_id=getattr(run, "task_id", None),
        continuation=continuation_from_run(run),
    )
    context = RuntimeInvocationContext(
        request_payload={"question": request.input.get("question", ""), "runtime_execution_mode": True},
        resolved_spec=dict(run.resolved_spec_json or {}),
        agent_run_context={"agent_run_id": run.id, "agent_workflow_id": run.workflow_id},
        task_id=getattr(run, "task_id", None),
    )
    status = "preserved"
    task = await tasks.get_task(str(run.task_id)) if getattr(run, "task_id", None) else None
    known_status = str((result or {}).get("status") or "") if isinstance(result, Mapping) else ""
    if task is not None and isinstance(result, Mapping) and result.get("orchestration_delta"):
        from app.services.agent_task_runtime_projection import apply_runtime_task_delta
        from runtime_protocol.transport import result_from_dict

        wire_result = dict(result)
        if not isinstance(wire_result.get("task_result"), Mapping) and isinstance(
            wire_result.get("runtime_task_result"), Mapping
        ):
            wire_result["task_result"] = dict(wire_result["runtime_task_result"])
        runtime_result = result_from_dict(wire_result)
        delta = runtime_result.orchestration_delta
        if delta is None:
            return "deferred"
        artifact_ids = await apply_runtime_task_delta(
            task_id=str(task.id), agent_run_id=str(run.id), delta=delta,
        )
        delta_sha256 = hashlib.sha256(
            json.dumps(delta.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        final_runtime_artifact = str((delta.result or {}).get("final_artifact_id") or "")
        await tasks.finalize_reconciled_runtime_task(
            str(task.id), str(run.id), delta_event_id=delta.event_id,
            payload_sha256=delta_sha256, runtime_status=runtime_result.status,
            result=dict(result),
            final_artifact_id=artifact_ids.get(final_runtime_artifact, final_runtime_artifact or None),
        )
        return "projected"
    if (
        task is not None
        and isinstance(result, Mapping)
        and (result.get("runtime_metadata") or {}).get("runtime_behavior", {}).get("continuation_semantics") == "linked_run"
    ):
        from app.services.agent_task_runtime_projection import apply_neutral_task_completion
        from runtime_protocol.transport import result_from_dict

        wire_result = dict(result)
        if not isinstance(wire_result.get("task_result"), Mapping) and isinstance(
            wire_result.get("runtime_task_result"), Mapping
        ):
            wire_result["task_result"] = dict(wire_result["runtime_task_result"])
        runtime_result = result_from_dict(wire_result)
        if runtime_result.task_result is None:
            projection.update({
                "reconciliation_status": "manual_required",
                "projection_error": {
                    "code": "runtime_task_result_missing",
                    "message": "A task-backed Hermes result has no neutral task result",
                    "retryable": False,
                },
            })
            await repository.update_runtime_projection(run.id, projection)
            return "deferred"
        task_result = runtime_result.task_result.to_dict()
        usage = dict(task_result.get("usage") or {})
        operation_id = str(
            usage.get("operation_id")
            or projection.get("operation_id")
            or f"task:{task.id}:run:{run.id}:reconcile"
        )
        await apply_neutral_task_completion(
            task_id=str(task.id), agent_run_id=str(run.id),
            operation_id=operation_id, runtime_status=runtime_result.status,
            task_result=task_result,
        )
        return "projected"
    if (
        task is not None and str(task.status) != "cancelling"
        and isinstance(result, Mapping)
        and bool((result.get("runtime_metadata") or {}).get("runtime_behavior", {}).get("supports_orchestration_delta"))
    ):
        projection.update({
            "reconciliation_status": "manual_required",
            "projection_error": {
                "code": "runtime_task_delta_missing",
                "message": "A task-backed LangGraph result cannot be projected by the chat projector",
                "retryable": False,
            },
        })
        await repository.update_runtime_projection(run.id, projection)
        return "deferred"
    if (
        task is not None
        and str(task.status) == "cancelling"
        and known_status in {"completed", "failed", "cancelled", "canceled", "no_continuation"}
    ):
        await reconcile_known_result(run, result, AgentRuntimeProjection())
        result = None
        status = "projected"
    elif task is not None and str(task.status) == "cancelling" and str(run.status) in {"running", "awaiting_human"}:
        inspection = await adapter.inspect_state(request)
        runtime_status = str(inspection.get("status") or (inspection.get("result") or {}).get("status") or "")
        if runtime_status in {"cancelled", "canceled"}:
            cancelled_result = inspection.get("result") if isinstance(inspection.get("result"), Mapping) else {
                "status": "cancelled",
                "error": {"code": "run_cancelled", "message": "Runtime cancellation confirmed", "retryable": False},
            }
            await confirm_task_cancellation(
                task,
                run,
                result=cancelled_result,
                terminal_event_id=str(inspection.get("terminal_event_id") or "") or None,
            )
            status = "projected"
        elif runtime_status in {"completed", "failed", "no_continuation"}:
            terminal_result = (
                dict(inspection.get("result"))
                if isinstance(inspection.get("result"), Mapping)
                else {
                    "status": runtime_status,
                    "error": dict(inspection.get("error") or {}),
                }
            )
            terminal_result.setdefault("status", runtime_status)
            await reconcile_known_result(run, terminal_result, AgentRuntimeProjection())
            status = "projected"
        else:
            await request_task_cancellation(task, run)
            status = "deferred"
    if result:
        await reconcile_known_result(run, result, AgentRuntimeProjection())
        status = "projected"
    elif status == "preserved":
        inspection = await adapter.inspect_state(request)
        if inspection.get("continuation_available"):
            status = "preserved"
        else:
            status = "deferred"
    refreshed = await AgentWorkflowRepository().get_run(run.id)
    projection = dict(((refreshed.run_metadata_json if refreshed is not None else {}) or {}).get("projection") or projection)
    await AgentWorkflowRepository().update_runtime_projection(
        run.id,
        {**projection, "reconciliation_status": status},
    )
    return status


async def reconcile_task_attempt(task_id: str, run_id: str, *, dry_run: bool = False) -> str:
    """Reconcile a task attempt through its owning AgentRun."""
    from app.agent_workflows.repository import AgentWorkflowRepository

    run = await AgentWorkflowRepository().get_run(run_id)
    if run is None or str(getattr(run, "task_id", "")) != str(task_id):
        return "missing"
    return await reconcile_run_by_id(run_id, dry_run=dry_run)


async def run_runtime_reconciliation(*, batch_size: int = 100, dry_run: bool = False) -> dict[str, int]:
    from app.agent_workflows.repository import AgentWorkflowRepository
    from app.runtime.catalog import definition_from_run
    from app.runtime.registry import adapter_for_definition
    from app.services import agent_task_repository as tasks
    from app.services.agent_task_runtime import ensure_task_run

    candidates = await AgentWorkflowRepository().list_runtime_reconciliation_candidates(limit=batch_size)
    counts = {"inspected": 0, "projected": 0, "preserved": 0, "failed": 0, "deferred": 0, "corrections": 0}
    for run in candidates:
        counts["inspected"] += 1
        try:
            status = await reconcile_run_by_id(run.id, dry_run=dry_run)
            counts[status] = counts.get(status, 0) + 1
        except Exception as exc:
            failed_run = await AgentWorkflowRepository().get_run(run.id)
            projection = dict(((failed_run.run_metadata_json if failed_run is not None else {}) or {}).get("projection") or {})
            failure_count = int(projection.get("reconciliation_failure_count") or 0) + 1
            projection.update({
                "reconciliation_status": "manual_required" if failure_count >= 3 else "failed",
                "reconciliation_failure_count": failure_count,
                "last_reconciliation_error": {
                    "type": type(exc).__name__, "message": str(exc)[:1000],
                },
                "next_retry_at": None if failure_count >= 3 else (
                    utc_now() + timedelta(seconds=min(300, 2 ** failure_count * 5))
                ).isoformat(),
            })
            await AgentWorkflowRepository().update_runtime_projection(run.id, projection)
            counts["failed"] += 1
    for command in await tasks.list_pending_course_correction_commands(limit=batch_size):
        if dry_run:
            counts["corrections"] += 1
            continue
        try:
            result = dict(command.result_json or {})
            correction = dict(result.get("correction") or {})
            task = await tasks.get_task(command.task_id)
            if task is None:
                continue
            if task.status in {"cancelling", "cancelled"} or task.deletion_requested_at is not None:
                await tasks.reject_course_correction(
                    command.id,
                    error={"code": "course_correction_cancelled", "retryable": False},
                )
                counts["corrections"] += 1
                continue
            run = await AgentWorkflowRepository().get_run(
                str(result.get("source_run_id") or correction.get("source_run_id") or "")
            )
            if run is None:
                await tasks.reject_course_correction(
                    command.id,
                    error={"code": "course_correction_source_run_missing", "retryable": False},
                )
                counts["corrections"] += 1
                continue
            if str(result.get("delivery_mode") or "") == "linked_run" or str(run.status) in {"completed", "failed", "cancelled", "expired", "rejected"}:
                await tasks.set_course_correction_delivery_mode(command.id, delivery_mode="linked_run")
                if str(run.status) in tasks.TERMINAL_TASK_RUN_STATUSES:
                    await tasks.queue_linked_course_correction(task.id, run_id=run.id)
                    await ensure_task_run(task.id)
                    counts["corrections"] += 1
                continue
            definition = definition_from_run(run)
            receipt = await adapter_for_definition(definition).submit_course_correction(
                AgentRuntimeRequest(
                    run_id=run.id,
                    thread_id=run.thread_id,
                    definition_id=definition.definition_id,
                    framework=definition.framework,
                    builder_id=definition.builder_id,
                    task_id=task.id,
                ),
                RuntimeCourseCorrection(
                    correction_id=str(correction.get("correction_id") or correction.get("id")),
                    operation_id=command.id,
                    instruction=str(correction.get("instruction") or ""),
                    scope=str(correction.get("scope") or "remaining_work"),
                    observed_task_version=int(correction.get("observed_task_version") or command.expected_version),
                    observed_plan_revision=int(correction.get("observed_plan_revision") or 0),
                    submitted_at=correction.get("submitted_at"),
                ),
            )
            if receipt.status == "terminal":
                await tasks.set_course_correction_delivery_mode(command.id, delivery_mode="linked_run", receipt=receipt.to_dict())
            elif receipt.status == "applied":
                await tasks.mark_course_corrections_runtime_applied(
                    task.id,
                    [receipt.correction_id],
                    plan_revision=int(receipt.plan_revision or 0),
                )
            else:
                await tasks.mark_course_correction_delivered(command.id, receipt=receipt.to_dict())
            counts["corrections"] += 1
        except AgentRuntimeError as exc:
            if not exc.retryable:
                await tasks.reject_course_correction(command.id, error=exc.to_dict())
            counts["failed"] += 1
        except Exception:
            counts["failed"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile uncertain external agent runtime outcomes")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_runtime_reconciliation(batch_size=args.batch_size, dry_run=args.dry_run)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
