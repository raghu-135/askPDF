"""Single control-plane owner for durable agent-run cancellation."""

from __future__ import annotations

import os
from typing import Any, Mapping

from app.db.enums import AgentRunStatus
from app.models.deep_research import AgentTaskStatus
from app.product_orchestration.interrupts import product_owned_review_pending
from app.runtime.capability_resolver import require_capability
from app.time_utils import utc_now
from app.runtime.catalog import continuation_from_run, definition_from_run
from runtime_protocol.contracts import AgentRuntimeRequest, RuntimeOperationId
from runtime_protocol.errors import RuntimeError
from runtime_protocol.events import create_runtime_event
from app.runtime.registry import RuntimeRegistry, get_runtime_registry
from app.runtime.termination import cancellation_reason, confirmed_cancellation_details


def _request(task: Any, run: Any) -> AgentRuntimeRequest:
    definition = definition_from_run(run)
    return AgentRuntimeRequest(
        run_id=str(run.id),
        thread_id=str(run.thread_id),
        definition_id=definition.definition_id,
        framework=definition.framework,
        builder_id=definition.builder_id,
        task_id=str(task.id),
        continuation=continuation_from_run(run),
    )


CANCEL_ABANDON_ERROR_CODES = frozenset({
    "runtime_run_not_found",
    "runtime_binding_missing",
    "runtime_binding_unavailable",
})


def cancel_confirm_timeout_seconds() -> float:
    return float(os.environ["AGENT_RUNTIME_CANCEL_CONFIRM_TIMEOUT_SECONDS"])


def cancellation_confirm_deadline_elapsed(task: Any, run: Any | None) -> bool:
    updated_at = getattr(task, "updated_at", None)
    if updated_at is None:
        return False
    if updated_at.tzinfo is None:
        from app.time_utils import parse_datetime_utc

        updated_at = parse_datetime_utc(str(updated_at))
    return (utc_now() - updated_at).total_seconds() >= cancel_confirm_timeout_seconds()


def should_finalize_cancel_on_terminal_runtime_ack(task: Any, run: Any) -> bool:
    if str(getattr(task, "status", "")) != AgentTaskStatus.CANCELLING.value:
        return False
    if not product_owned_review_pending(run):
        return False
    metadata = dict(getattr(run, "run_metadata_json", None) or {})
    projection = dict(metadata.get("projection") or {})
    return projection.get("status") == "applied" or str(getattr(run, "status", "")) == AgentRunStatus.AWAITING_HUMAN.value


def should_abandon_runtime_cancellation(task: Any, run: Any, *, error: RuntimeError | None = None) -> bool:
    if run is None:
        return True
    metadata = dict(getattr(run, "run_metadata_json", None) or {})
    if metadata.get("runtime_started") is False:
        return True
    if error is not None and error.code in CANCEL_ABANDON_ERROR_CODES:
        return True
    return cancellation_confirm_deadline_elapsed(task, run)


async def abandon_pending_cancellation(task: Any, run: Any, *, reason: str = "cancelled_by_user") -> Any:
    """Finalize a cancelling task when the runtime cannot confirm cancellation."""

    from app.services import agent_task_repository as tasks

    if product_owned_review_pending(run):
        await tasks.clear_product_review_interrupt(str(run.id))
    return await confirm_task_cancellation(
        task,
        run,
        result={"status": "cancelled", "runtime_status": "cancelled", "reason": reason},
        terminal_event_id=f"{run.id}:cancelled",
    )


async def require_task_cancellation(task: Any, run: Any | None, *, registry: RuntimeRegistry | None = None) -> None:
    """Fail before task mutation when an active runtime cannot be cancelled."""

    if run is None:
        return
    selected = registry or get_runtime_registry()
    await require_capability(
        definition_from_run(run),
        RuntimeOperationId.RUN_CANCEL,
        registry=selected,
        run=run,
    )


async def request_task_cancellation(
    task: Any,
    run: Any | None,
    *,
    registry: RuntimeRegistry | None = None,
) -> Mapping[str, Any]:
    """Submit cancellation; acknowledgement is deliberately nonterminal."""

    if run is None:
        return {"status": "cancelled", "task_id": str(task.id), "runtime_confirmation": "not_required"}
    selected = registry or get_runtime_registry()
    await require_task_cancellation(task, run, registry=selected)
    adapter = selected.get(definition_from_run(run))
    try:
        result = await adapter.cancel(_request(task, run))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError.from_exception(
            exc,
            code="runtime_transport_error",
            retryable=True,
            safe_message="Runtime cancellation could not be submitted",
            details={"operation_id": RuntimeOperationId.RUN_CANCEL.value, "run_id": str(run.id)},
        ) from exc
    value = dict(result) if isinstance(result, Mapping) else {"result": result}
    upstream_status = str(value.get("status") or value.get("run_status") or "").strip()
    return {
        **value,
        "status": "cancelling",
        "run_id": str(run.id),
        "runtime_status": upstream_status or None,
        "runtime_confirmation": (
            "terminal" if upstream_status in {"completed", "failed", "cancelled", "canceled"}
            else "pending"
        ),
    }


async def confirm_task_cancellation(
    task: Any,
    run: Any,
    *,
    result: Mapping[str, Any] | None = None,
    terminal_event_id: str | None = None,
) -> Any:
    """Atomically project one authoritative runtime cancellation."""

    from app.services import agent_task_repository as tasks
    from app.services.agent_runtime_reconciliation import record_terminal_result, reconcile_run_by_id

    status = str((result or {}).get("runtime_status") or (result or {}).get("status") or "")
    if status in {"completed", "failed"}:
        if should_finalize_cancel_on_terminal_runtime_ack(task, run):
            return await abandon_pending_cancellation(task, run, reason="cancelled_by_user")
        # An acknowledgement of an already terminal execution is not evidence
        # of cancellation. Fetch/project its real output, including task deltas.
        return await reconcile_run_by_id(str(run.id))
    if status not in {"cancelled", "canceled"}:
        raise ValueError("runtime cancellation requires confirmed cancelled status")

    cancelled_result = dict(result or {})
    event_id = terminal_event_id or str(cancelled_result.get("terminal_event_id") or f"{run.id}:cancelled")
    await record_terminal_result(run, cancelled_result, terminal_event_id=event_id)
    # Keep the runtime snapshot untouched (its hash is used for replay).
    error = dict(cancelled_result.get("error") or {
        "code": "run_cancelled", "message": "Runtime cancellation confirmed", "retryable": False,
    })
    terminal = create_runtime_event(
        event_id=event_id,
        run_id=str(run.id),
        sequence=1,
        kind="run.cancelled",
        payload={"status": "cancelled", "error": error, "cancellation": confirmed_cancellation_details(task, run)},
    )
    finalized = await tasks.finalize_task_run(
        str(task.id),
        str(run.id),
        run_status="cancelled",
        task_status="cancelled",
        metrics=dict(getattr(run, "metrics_json", None) or {}),
        error=error,
        debug_trace=dict(getattr(run, "debug_trace_json", None) or {}),
        terminal_reason=cancellation_reason(task, run),
        terminal_event=terminal,
    )
    await tasks.complete_pending_cancel_commands(
        str(task.id),
        result={
            "status": "cancelled",
            "task_id": str(task.id),
            "run_id": str(run.id),
            "runtime_confirmation": "confirmed",
        },
    )
    return finalized
