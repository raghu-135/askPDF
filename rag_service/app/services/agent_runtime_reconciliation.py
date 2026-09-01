"""Control-plane recovery for uncertain external-runtime outcomes."""

from __future__ import annotations

import hashlib
import json
import argparse
import asyncio
from types import SimpleNamespace
from typing import Any, Mapping

from app.agent_workflows.repository import AgentWorkflowRepository
from runtime_protocol.contracts import AgentRuntimeRequest


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

    candidates = await AgentWorkflowRepository().list_runtime_reconciliation_candidates(limit=batch_size)
    counts = {"inspected": 0, "projected": 0, "preserved": 0, "failed": 0, "deferred": 0}
    for run in candidates:
        counts["inspected"] += 1
        try:
            status = await reconcile_run_by_id(run.id, dry_run=dry_run)
            counts[status] = counts.get(status, 0) + 1
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
