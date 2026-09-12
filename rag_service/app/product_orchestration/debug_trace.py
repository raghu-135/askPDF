from __future__ import annotations

from copy import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence

from runtime_protocol.contracts import AgentRuntimeEvent

from app.product_orchestration.trace_payloads import (
    DEBUG_PAYLOAD_VERSION,
    append_interrupt_event_to_debug_payload,
    append_runtime_event_to_debug_payload,
    build_interrupt_trace_event,
    build_runtime_trace_event,
    merge_debug_payloads,
)
from app.product_orchestration.trace_recorder import AgentTraceRecorder, TRACE_SCHEMA_VERSION


def _journal_event_to_runtime(event: Any, run: Any) -> AgentRuntimeEvent:
    if isinstance(event, AgentRuntimeEvent):
        return event
    occurred_at = getattr(event, "occurred_at", None)
    payload = getattr(event, "payload_json", None)
    if payload is None:
        payload = getattr(event, "payload", None)
    source_metadata = getattr(event, "source_metadata_json", None)
    if source_metadata is None:
        source_metadata = getattr(event, "source_metadata", None)
    return AgentRuntimeEvent(
        event_id=str(getattr(event, "event_id", "") or ""),
        run_id=str(getattr(event, "agent_run_id", None) or getattr(event, "run_id", None) or run.id),
        sequence=int(getattr(event, "sequence", 0) or 0),
        attempt=int(getattr(event, "attempt", 1) or 1),
        kind=str(getattr(event, "kind", "runtime.event") or "runtime.event"),
        payload=payload if isinstance(payload, dict) else {},
        occurred_at=str(occurred_at) if occurred_at else None,
        terminal=bool(getattr(event, "terminal", False)),
        source_metadata=source_metadata if isinstance(source_metadata, dict) else {},
    )


def build_debug_payload_from_journal(
    run: Any,
    events: Sequence[Any],
    *,
    result: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Project a debug payload from the durable runtime event journal.

    Used for in-flight inspection. This does not persist the payload; callers
    that own a terminal transition still write the finalized trace separately.
    """

    if not events:
        return None
    recorder = AgentTraceRecorder(run)
    for event in events:
        recorder.record_agent_runtime_event(_journal_event_to_runtime(event, run))
    result_payload = dict(result or {})
    return finalize_and_merge_debug_payload(
        recorder=recorder,
        run=run,
        metrics=dict(getattr(run, "metrics_json", None) or {}),
        result=result_payload or None,
        chat_turn_id=result_payload.get("chat_turn_id"),
        route=result_payload.get("route"),
        route_reason=result_payload.get("route_reason"),
        error=result_payload.get("agent_error") or getattr(run, "error_json", None),
        run_status=str(result_payload.get("status") or getattr(run, "status", "")),
        completed_at=getattr(run, "completed_at", None),
    )


def finalize_and_merge_debug_payload(
    *,
    recorder: AgentTraceRecorder,
    run: Any,
    metrics: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
    chat_turn_id: Optional[str] = None,
    route: Any = None,
    route_reason: Any = None,
    error: Any = None,
    run_status: Optional[str] = None,
    completed_at: Any = None,
) -> Dict[str, Any]:
    """Finalize one execution phase and merge it with an earlier persisted phase."""

    status = run_status or getattr(run, "status", None)
    final_run = copy(run)
    if run_status is not None:
        setattr(final_run, "status", run_status)
    if completed_at is not None:
        setattr(final_run, "completed_at", completed_at)
    incoming = recorder.finalize(
        run=final_run,
        chat_turn_id=chat_turn_id,
        metrics=metrics,
        route=route,
        route_reason=route_reason,
        error=error,
        result=result,
    )
    existing = getattr(run, "debug_trace_json", None)
    if not isinstance(existing, dict):
        return incoming
    return merge_debug_payloads(
        existing,
        incoming,
        resolved_spec=getattr(run, "resolved_spec_json", None)
        if isinstance(getattr(run, "resolved_spec_json", None), dict)
        else {},
        run_status=status,
        completed_at=completed_at if completed_at is not None else getattr(run, "completed_at", None),
        chat_turn_id=chat_turn_id,
        metrics=metrics,
    )
