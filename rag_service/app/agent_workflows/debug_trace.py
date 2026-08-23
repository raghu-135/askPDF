from __future__ import annotations

from copy import copy
from typing import Any, Dict, List, Optional

from app.agent_workflows.trace_payloads import (
    DEBUG_PAYLOAD_VERSION,
    append_interrupt_event_to_debug_payload,
    append_runtime_event_to_debug_payload,
    build_interrupt_trace_event,
    build_runtime_trace_event,
    merge_debug_payloads,
)
from app.agent_workflows.trace_recorder import AgentTraceRecorder, TRACE_SCHEMA_VERSION


def build_debug_payload(
    *,
    run: Any,
    chat_turn_id: Optional[str] = None,
    node_events: List[Dict[str, Any]],
    tool_events: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    route: Any = None,
    route_reason: Any = None,
    error: Any = None,
) -> Dict[str, Any]:
    recorder = AgentTraceRecorder(run)
    for event in node_events:
        if isinstance(event, dict):
            recorder.record_node_event(event)
    for event in tool_events:
        if isinstance(event, dict):
            recorder.record_tool_event(event)
    return recorder.finalize(
        run=run,
        chat_turn_id=chat_turn_id,
        metrics=metrics,
        route=route,
        route_reason=route_reason,
        error=error,
    )


def build_debug_trace(
    *,
    run: Any,
    chat_turn: Any = None,
    node_events: List[Dict[str, Any]],
    tool_events: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    route: Any = None,
    route_reason: Any = None,
    error: Any = None,
) -> Dict[str, Any]:
    """Build only the v1 trace document for tests and schema checks."""

    payload = build_debug_payload(
        run=run,
        chat_turn_id=getattr(chat_turn, "id", None) if chat_turn is not None else None,
        node_events=node_events,
        tool_events=tool_events,
        metrics=metrics,
        route=route,
        route_reason=route_reason,
        error=error,
    )
    return payload["trace"]


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
