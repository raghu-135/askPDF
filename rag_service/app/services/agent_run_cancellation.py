"""Single control-plane owner for durable agent-run cancellation."""

from __future__ import annotations

from typing import Any, Mapping

from app.runtime.capability_resolver import RuntimeCapabilityDisabledReason, require_capability
from app.runtime.catalog import continuation_from_run, definition_from_run
from runtime_protocol.contracts import AgentRuntimeRequest, AgentRuntimeResult, RuntimeOperationId
from runtime_protocol.errors import RuntimeError
from runtime_protocol.events import create_runtime_event
from app.runtime.registry import RuntimeRegistry, get_runtime_registry
from app.runtime.termination import cancellation_reason, confirmed_cancellation_details

_CANCELLATION_DEFERRAL_CODES = frozenset({
    "runtime_run_not_found",
    "runtime_binding_missing",
    "runtime_binding_unavailable",
    "runtime_capability_unavailable",
})


def cancellation_submission_deferred(exc: AgentRuntimeError) -> bool:
    """Return True when runtime cancel should be retried by the worker, not rolled back."""

    if exc.code in _CANCELLATION_DEFERRAL_CODES - {"runtime_capability_unavailable"}:
        return True
    if exc.code != "runtime_capability_unavailable":
        return False
    disabled_reason = str((exc.details or {}).get("disabled_reason") or "").strip()
    return disabled_reason in {
        "runtime_binding_unavailable",
        RuntimeCapabilityDisabledReason.RUNTIME_BINDING_UNAVAILABLE.value,
    }


def runtime_cancel_submitted(run: Any | None) -> bool:
    metadata = getattr(run, "run_metadata_json", None) if run is not None else None
    return isinstance(metadata, Mapping) and metadata.get("runtime_started") is True


async def honor_product_cancellation_before_runtime_start(task: Any, run: Any) -> AgentRuntimeResult | None:
    """Stop a fresh product run before transport when cancellation is already durable."""

    if run is None or not getattr(run, "_fresh_runtime_run", False):
        return None
    if str(getattr(task, "status", "") or "") != "cancelling":
        return None
    metadata = dict(getattr(run, "run_metadata_json", None) or {})
    if metadata.get("runtime_started") is True:
        return None
    await finalize_product_local_cancellation(task, run)
    return AgentRuntimeResult(
        status="cancelled",
        error={
            "code": "run_cancelled",
            "message": "Task cancellation honoured before runtime start",
            "retryable": False,
        },
    )


async def finalize_product_local_cancellation(task: Any, run: Any) -> Any:
    """Terminalize task and run without a runtime round-trip."""

    from app.services import agent_task_repository as tasks
    from runtime_protocol.events import create_runtime_event

    event_id = f"{run.id}:cancelled"
    terminal = create_runtime_event(
        event_id=event_id,
        run_id=str(run.id),
        sequence=1,
        kind="run.cancelled",
        payload={
            "status": "cancelled",
            "error": {
                "code": "run_cancelled",
                "message": "Task cancelled before runtime execution started",
                "retryable": False,
            },
            "cancellation": confirmed_cancellation_details(task, run),
        },
    )
    finalized = await tasks.finalize_task_run(
        str(task.id),
        str(run.id),
        run_status="cancelled",
        task_status="cancelled",
        metrics=dict(getattr(run, "metrics_json", None) or {}),
        error={
            "code": "run_cancelled",
            "message": "Task cancelled before runtime execution started",
            "retryable": False,
        },
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


async def record_scheduled_task_cancellation(task: Any, run: Any | None) -> None:
    """Persist cooperative cancellation while runtime admission is still pending."""

    if run is None:
        return
    from app.db.jsonb_utils import replace_jsonb_field
    from app.db.connection_sqlmodel import async_session_maker
    from app.db.models_sqlmodel import AgentRun
    from sqlalchemy import select

    async with async_session_maker() as session:
        async with session.begin():
            stored = (
                await session.execute(select(AgentRun).where(AgentRun.id == run.id).with_for_update())
            ).scalar_one_or_none()
            if stored is None or stored.status in {"completed", "failed", "cancelled"}:
                return
            metadata = dict(stored.run_metadata_json or {})
            metadata["cancel_requested"] = True
            metadata["cancellation_request"] = {
                **dict(metadata.get("cancellation_request") or {}),
                "task_id": str(task.id),
                "run_id": str(run.id),
                "reason": "cancelled_by_user",
                "runtime_confirmation": "scheduled",
            }
            replace_jsonb_field(stored, "run_metadata_json", metadata)


def scheduled_cancellation_result(task: Any, run: Any | None) -> Mapping[str, Any]:
    return {
        "status": "cancelling",
        "task_id": str(task.id),
        "run_id": None if run is None else str(run.id),
        "runtime_confirmation": "scheduled",
    }


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
