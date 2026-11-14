"""Single control-plane owner for durable agent-run cancellation."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

from app.db.enums import AgentRunStatus
from app.models.deep_research import AgentTaskStatus
from app.product_orchestration.interrupts import product_owned_review_pending
from app.runtime.capability_resolver import RuntimeCapabilityDisabledReason, require_capability
from app.time_utils import utc_now
from app.runtime.catalog import continuation_from_run, definition_from_run
from runtime_protocol.contracts import AgentRuntimeRequest, AgentRuntimeResult, RuntimeOperationId
from runtime_protocol.errors import RuntimeError as AgentRuntimeError
from runtime_protocol.events import create_runtime_event
from app.runtime.registry import RuntimeRegistry, get_runtime_registry
from app.runtime.termination import cancellation_reason, confirmed_cancellation_details

logger = logging.getLogger(__name__)

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

    return await tasks.finalize_product_local_cancellation(str(task.id), str(run.id))


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
                "requested_at": utc_now().isoformat(),
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


CANCEL_ABANDON_ERROR_CODES = frozenset({
    "runtime_run_not_found",
    "runtime_binding_missing",
    "runtime_binding_unavailable",
})


def cancel_confirm_timeout_seconds() -> float:
    return float(os.environ["AGENT_RUNTIME_CANCEL_CONFIRM_TIMEOUT_SECONDS"])


def _cancellation_deadline_anchor(task: Any, run: Any | None):
    """Anchor confirm-timeout to the original cancel request, not lease churn."""

    from app.time_utils import parse_datetime_utc

    if run is not None:
        metadata = dict(getattr(run, "run_metadata_json", None) or {})
        request = dict(metadata.get("cancellation_request") or {})
        requested_at = request.get("requested_at")
        if requested_at:
            return parse_datetime_utc(str(requested_at))
    updated_at = getattr(task, "updated_at", None)
    if updated_at is None:
        return None
    if updated_at.tzinfo is None:
        return parse_datetime_utc(str(updated_at))
    return updated_at


def cancellation_confirm_deadline_elapsed(task: Any, run: Any | None) -> bool:
    anchor = _cancellation_deadline_anchor(task, run)
    if anchor is None:
        return False
    return (utc_now() - anchor).total_seconds() >= cancel_confirm_timeout_seconds()


def should_finalize_cancel_on_terminal_runtime_ack(task: Any, run: Any) -> bool:
    if str(getattr(task, "status", "")) != AgentTaskStatus.CANCELLING.value:
        return False
    if not product_owned_review_pending(run):
        return False
    metadata = dict(getattr(run, "run_metadata_json", None) or {})
    projection = dict(metadata.get("projection") or {})
    return projection.get("status") == "applied" or str(getattr(run, "status", "")) == AgentRunStatus.AWAITING_HUMAN.value


def cancellation_scheduled_on_run(run: Any | None) -> bool:
    if run is None:
        return False
    metadata = dict(getattr(run, "run_metadata_json", None) or {})
    if metadata.get("runtime_started") is True:
        return False
    request = dict(metadata.get("cancellation_request") or {})
    return bool(metadata.get("cancel_requested") or request.get("runtime_confirmation") == "scheduled")


def should_abandon_runtime_cancellation(task: Any, run: Any, *, error: AgentRuntimeError | None = None) -> bool:
    if run is None:
        return True
    if cancellation_scheduled_on_run(run):
        return False
    metadata = dict(getattr(run, "run_metadata_json", None) or {})
    if metadata.get("runtime_started") is False:
        return False
    if error is not None and error.code in CANCEL_ABANDON_ERROR_CODES:
        return cancellation_confirm_deadline_elapsed(task, run)
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
    except AgentRuntimeError:
        raise
    except Exception as exc:
        raise AgentRuntimeError.from_exception(
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


_LIVE_RUNTIME_STATUSES = frozenset({"running", "awaiting_human", "paused", "pausing"})
_TERMINAL_RUN_STATUSES = frozenset({
    AgentRunStatus.COMPLETED.value,
    AgentRunStatus.FAILED.value,
    AgentRunStatus.CANCELLED.value,
    AgentRunStatus.EXPIRED.value,
    AgentRunStatus.REJECTED.value,
})
_ACTIVE_RUN_STATUSES = frozenset({
    AgentRunStatus.RUNNING.value,
    AgentRunStatus.AWAITING_HUMAN.value,
})


async def _finalize_cancelling_against_terminal_run(task: Any, run: Any) -> str:
    """Align a cancelling task when its active run already reached a terminal state."""

    from app.services import agent_task_repository as tasks
    from app.services.agent_runtime_reconciliation import reconcile_run_by_id

    if str(run.status) == "cancelled":
        refreshed = await tasks.get_task(str(task.id))
        if refreshed is not None and refreshed.status == AgentTaskStatus.CANCELLED.value:
            return "not_required"
        await abandon_pending_cancellation(task, run)
        return "finalized"
    status = await reconcile_run_by_id(str(run.id))
    refreshed = await tasks.get_task(str(task.id))
    if refreshed is not None and refreshed.status != AgentTaskStatus.CANCELLING.value:
        return "projected" if status == "projected" else "finalized"
    # Run ended without a projectable snapshot; treat as user cancellation.
    if cancellation_confirm_deadline_elapsed(task, run):
        await abandon_pending_cancellation(task, run)
        return "finalized"
    return "deferred"


async def _reconcile_cancelling_from_runtime_inspection(task: Any, run: Any) -> str:
    """Inspect authoritative runtime state and project or abandon a stale cancellation."""

    from app.services.agent_runtime_reconciliation import reconcile_run_by_id

    metadata = dict(getattr(run, "run_metadata_json", None) or {})
    if metadata.get("runtime_started") is not True:
        await finalize_product_local_cancellation(task, run)
        return "finalized"
    try:
        adapter = get_runtime_registry().get(definition_from_run(run))
        inspection = await adapter.inspect_state(_request(task, run))
    except AgentRuntimeError as exc:
        if should_abandon_runtime_cancellation(task, run, error=exc):
            await abandon_pending_cancellation(task, run, reason=str(exc.code or "cancelled_by_user"))
            return "finalized"
        return "deferred"
    except Exception:
        return "deferred"

    runtime_status = str(
        inspection.get("status") or (inspection.get("result") or {}).get("status") or ""
    ).strip()
    if inspection.get("continuation_available") and runtime_status in _LIVE_RUNTIME_STATUSES:
        return "deferred"
    if runtime_status in {"cancelled", "canceled"}:
        terminal_result = inspection.get("result")
        if not isinstance(terminal_result, Mapping):
            terminal_result = {"status": "cancelled", "runtime_status": "cancelled"}
        await confirm_task_cancellation(
            task,
            run,
            result=terminal_result,
            terminal_event_id=inspection.get("terminal_event_id"),
        )
        return "finalized"
    if runtime_status in {"completed", "failed", "no_continuation"}:
        await reconcile_run_by_id(str(run.id))
        return "projected"
    await abandon_pending_cancellation(task, run)
    return "finalized"


async def recover_cancelling_task(
    task: Any,
    run: Any | None,
    *,
    allow_runtime_submit: bool = False,
) -> str:
    """Shared worker and maintenance path for cancelling task recovery."""

    from app.services import agent_task_repository as tasks
    from app.services.agent_runtime_reconciliation import reconcile_run_by_id

    task_id = str(getattr(task, "id", "") or "")
    if not task_id:
        return "missing"
    refreshed = await tasks.get_task(task_id)
    if refreshed is None:
        return "missing"
    task = refreshed
    if str(task.status) != AgentTaskStatus.CANCELLING.value:
        return "not_required"

    if run is None:
        run = await tasks.get_task_run(task_id)

    if run is not None and str(run.status) in _TERMINAL_RUN_STATUSES:
        return await _finalize_cancelling_against_terminal_run(task, run)

    stale = cancellation_confirm_deadline_elapsed(task, run)

    if run is None:
        await tasks.complete_task(task_id, status=AgentTaskStatus.CANCELLED.value, reason="cancelled_by_user")
        return "finalized"

    if cancellation_scheduled_on_run(run):
        await finalize_product_local_cancellation(task, run)
        return "finalized"

    if str(run.status) in _ACTIVE_RUN_STATUSES:
        if allow_runtime_submit and not stale:
            try:
                cancellation = await request_task_cancellation(task, run)
                if cancellation.get("runtime_confirmation") == "terminal":
                    runtime_status = str(cancellation.get("runtime_status") or "")
                    if should_finalize_cancel_on_terminal_runtime_ack(task, run) and runtime_status in {
                        "completed",
                        "failed",
                    }:
                        await abandon_pending_cancellation(task, run)
                        return "finalized"
                    await reconcile_run_by_id(str(run.id))
                    return "projected"
            except AgentRuntimeError as exc:
                if should_abandon_runtime_cancellation(task, run, error=exc):
                    await abandon_pending_cancellation(task, run, reason=exc.code)
                    return "finalized"
                logger.warning(
                    "Runtime cancellation remains pending after recovery | task_id=%s run_id=%s code=%s",
                    task_id,
                    run.id,
                    exc.code,
                )
                return "deferred"
            return "deferred"
        if not stale:
            return "fresh"
        return await _reconcile_cancelling_from_runtime_inspection(task, run)

    if stale:
        await abandon_pending_cancellation(task, run)
        return "finalized"
    return "fresh"


async def reconcile_stale_cancelling_task(task: Any, run: Any | None = None) -> str:
    """Maintenance entrypoint: never resubmits runtime cancel while the deadline is fresh."""

    return await recover_cancelling_task(task, run, allow_runtime_submit=False)


async def run_stale_cancellation_reconciliation(*, batch_size: int = 100, dry_run: bool = False) -> dict[str, int]:
    """Reconcile leftover cancelling tasks during periodic maintenance."""

    from app.services import agent_task_repository as tasks

    counts = {"inspected": 0, "fresh": 0, "finalized": 0, "projected": 0, "deferred": 0, "not_required": 0, "missing": 0, "failed": 0}
    for task in await tasks.list_stale_cancelling_tasks(limit=batch_size):
        counts["inspected"] += 1
        if dry_run:
            counts["deferred"] += 1
            continue
        try:
            run = await tasks.get_task_run(task.id)
            status = await reconcile_stale_cancelling_task(task, run)
            counts[status] = counts.get(status, 0) + 1
        except Exception:
            counts["failed"] += 1
    return counts
