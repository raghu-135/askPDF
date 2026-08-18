from __future__ import annotations

import time
from datetime import timedelta
from typing import Any, Callable, Dict, Optional

from langchain_core.runnables import RunnableConfig

from app.agent.tool_contract import compact_tool_event


def append_node_event(
    state: Dict[str, Any],
    node: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    started: Optional[float] = None,
    config: Optional[RunnableConfig] = None,
    runtime_node_id: Callable[[Optional[RunnableConfig], str], str],
    runtime_node_type: Callable[[Optional[RunnableConfig], str], str],
    runtime_visit_index: Callable[[Optional[RunnableConfig]], Optional[int]],
    utc_now: Callable[[], Any],
    iso_utc_z: Callable[[Any], str],
) -> list[Dict[str, Any]]:
    event = {
        "node": runtime_node_id(config, node),
        "node_type": runtime_node_type(config, node),
        **(data or {}),
    }
    visit_index = runtime_visit_index(config)
    if visit_index is not None:
        event["visit_index"] = visit_index
    if started is not None:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        completed_at = utc_now()
        event["elapsed_ms"] = elapsed_ms
        event.setdefault("start_time", iso_utc_z(completed_at - timedelta(milliseconds=elapsed_ms)))
        event.setdefault("end_time", iso_utc_z(completed_at))
    telemetry_sink = ((config or {}).get("configurable") or {}).get("telemetry_sink")
    if isinstance(telemetry_sink, dict):
        telemetry_sink.setdefault("node_events", []).append(dict(event))
    trace_recorder = ((config or {}).get("configurable") or {}).get("trace_recorder")
    if trace_recorder is not None and hasattr(trace_recorder, "record_node_event"):
        trace_recorder.record_node_event(dict(event))
    return [*state.get("node_events", []), event]


def append_tool_event(
    state: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    tool_input: Any = None,
    config: Optional[RunnableConfig] = None,
    runtime_node_type: Callable[[Optional[RunnableConfig], str], str],
    runtime_visit_index: Callable[[Optional[RunnableConfig]], Optional[int]],
) -> list[Dict[str, Any]]:
    event = compact_tool_event(payload, tool_input=tool_input)
    caller_node_type = event.get("caller_node_type") or runtime_node_type(config, str(event.get("caller_node") or ""))
    if caller_node_type:
        event["caller_node_type"] = caller_node_type
    visit_index = runtime_visit_index(config)
    if visit_index is not None:
        event["caller_visit_index"] = visit_index
    trace_recorder = ((config or {}).get("configurable") or {}).get("trace_recorder")
    if trace_recorder is not None and hasattr(trace_recorder, "record_tool_detail"):
        trace_recorder.record_tool_detail(
            payload={
                **dict(payload),
                "caller_node": event.get("caller_node"),
                "caller_node_type": event.get("caller_node_type"),
                "caller_visit_index": event.get("caller_visit_index"),
                "tool_name": event.get("tool_name"),
            },
            tool_input=tool_input,
        )
    telemetry_sink = ((config or {}).get("configurable") or {}).get("telemetry_sink")
    if isinstance(telemetry_sink, dict):
        telemetry_sink.setdefault("tool_events", []).append(dict(event))
    if trace_recorder is not None and hasattr(trace_recorder, "record_tool_event"):
        trace_recorder.record_tool_event(dict(event))
    execution_event_sink = ((config or {}).get("configurable") or {}).get("execution_event_sink")
    if execution_event_sink is not None and hasattr(execution_event_sink, "emit_nowait"):
        execution_event_sink.emit_nowait("tool.completed", event)
    return [*state.get("tool_events", []), event]
