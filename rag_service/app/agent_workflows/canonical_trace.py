from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence, TypedDict

from app.agent_workflows.trace_sanitization import _bounded_value
from app.runtime.contracts import AgentRuntimeEvent


class TraceVisualizationId(str, Enum):
    GENERIC_TIMELINE = "generic.timeline"
    LANGGRAPH_GRAPH = "langgraph.graph"
    HERMES_SESSION = "hermes.session"


class GenericTimelineVisualization(TypedDict):
    id: str


class LangGraphVisualization(TypedDict):
    id: str
    nodes: list[Any]
    edges: list[Any]
    execution_plan: list[Any]
    selected_route: Any
    visits: list[Any]


class HermesSessionVisualization(TypedDict):
    id: str
    session_id: Any
    upstream_run_id: Any
    reasoning: list[dict[str, Any]]
    approvals: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    subagents: list[dict[str, Any]]
    failures: list[dict[str, Any]]


TRACE_VISUALIZATION_GENERIC = TraceVisualizationId.GENERIC_TIMELINE.value
TRACE_VISUALIZATION_LANGGRAPH = TraceVisualizationId.LANGGRAPH_GRAPH.value
TRACE_VISUALIZATION_HERMES = TraceVisualizationId.HERMES_SESSION.value


def _number(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _framework_details(payload: Mapping[str, Any], framework: str) -> dict[str, Any]:
    details = payload.get("framework_details")
    result = dict(details) if isinstance(details, Mapping) else {}
    metadata = payload.get("framework_metadata")
    if framework and isinstance(metadata, Mapping) and framework not in result:
        result[framework] = dict(metadata)
    return result


def _timeline_event(event: AgentRuntimeEvent, framework: str) -> dict[str, Any]:
    payload = dict(event.payload)
    if event.kind.startswith("tool."):
        arguments = payload.get("arguments") or payload.get("args") or payload.get("input")
        if isinstance(arguments, Mapping):
            payload.setdefault("provided_argument_names", sorted(str(key) for key in arguments))
        for key in ("arguments", "args", "input"):
            payload.pop(key, None)
    return {
        "event_id": event.event_id,
        "sequence": event.sequence,
        "attempt": event.attempt,
        "kind": event.kind,
        "occurred_at": event.occurred_at,
        "operation_id": payload.get("operation_id"),
        "parent_operation_id": payload.get("parent_operation_id") or payload.get("parent_id"),
        "status": payload.get("status"),
        "payload": _bounded_value(payload),
        "framework_details": _bounded_value(_framework_details(payload, framework)),
    }


def _operation_key(payload: Mapping[str, Any]) -> tuple[str, int] | None:
    operation_id = str(payload.get("operation_id") or "").strip()
    return (operation_id, _number(payload.get("visit_index"))) if operation_id else None


def _duration_ms(started_at: Any, completed_at: Any) -> float | None:
    if not started_at or not completed_at:
        return None
    try:
        start = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(completed_at).replace("Z", "+00:00"))
        return max(0.0, round((end - start).total_seconds() * 1000, 3))
    except (TypeError, ValueError):
        return None


def _operations(events: Sequence[AgentRuntimeEvent], framework: str) -> list[dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for event in events:
        if not event.kind.startswith("operation."):
            continue
        payload = dict(event.payload)
        key = _operation_key(payload)
        if key is None:
            continue
        operation_id, visit_index = key
        row = rows.setdefault((operation_id, visit_index), {
            "operation_id": operation_id,
            "operation_type": str(payload.get("operation_type") or "runtime_operation"),
            "operation_label": str(payload.get("operation_label") or payload.get("label") or operation_id),
            "parent_operation_id": payload.get("parent_operation_id") or payload.get("parent_id"),
            "visit_index": visit_index,
            "attempt": _number(payload.get("attempt")),
            "status": "running",
            "started_at": None,
            "completed_at": None,
            "duration_ms": None,
            "input": {},
            "output": {},
            "error": None,
            "topology_ref": payload.get("topology_ref") if isinstance(payload.get("topology_ref"), Mapping) else None,
            "framework_details": {},
            "route": None,
            "route_reason": None,
            "execution_plan": [],
            "sequence": event.sequence,
        })
        row.update({
            "operation_type": str(payload.get("operation_type") or row["operation_type"]),
            "operation_label": str(payload.get("operation_label") or payload.get("label") or row["operation_label"]),
            "parent_operation_id": payload.get("parent_operation_id") or payload.get("parent_id") or row.get("parent_operation_id"),
            "attempt": _number(payload.get("attempt"), row["attempt"]),
            "topology_ref": payload.get("topology_ref") if isinstance(payload.get("topology_ref"), Mapping) else row.get("topology_ref"),
            "framework_details": {**dict(row.get("framework_details") or {}), **_framework_details(payload, framework)},
            "route": payload.get("route") or row.get("route"),
            "route_reason": payload.get("route_reason") or row.get("route_reason"),
            "execution_plan": _bounded_value(payload.get("execution_plan") or row.get("execution_plan") or []),
        })
        if event.kind == "operation.started":
            row["status"] = "running"
            row["started_at"] = payload.get("started_at") or payload.get("start_time") or event.occurred_at
            row["input"] = _bounded_value(payload.get("input") or payload.get("input_summary") or {})
        else:
            row["status"] = {
                "operation.completed": "completed",
                "operation.failed": "failed",
                "operation.skipped": "skipped",
            }.get(event.kind, str(payload.get("status") or row["status"]))
            row["completed_at"] = payload.get("completed_at") or payload.get("end_time") or event.occurred_at
            row["duration_ms"] = payload.get("duration_ms") or payload.get("elapsed_ms")
            if row["duration_ms"] is None:
                row["duration_ms"] = _duration_ms(row.get("started_at"), row.get("completed_at"))
            row["output"] = _bounded_value(payload.get("output") or payload.get("output_summary") or payload.get("detail") or {})
            row["error"] = _bounded_value(payload.get("error"))
    return sorted(rows.values(), key=lambda row: (int(row.get("sequence") or 0), str(row["operation_id"]), int(row["visit_index"])))


def _event_rows(events: Sequence[AgentRuntimeEvent], prefix: str, framework: str) -> list[dict[str, Any]]:
    return [_timeline_event(event, framework) for event in events if event.kind.startswith(prefix)]


def _failure_rows(events: Sequence[AgentRuntimeEvent], framework: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    terminal_failure_id: str | None = None
    for event in events:
        payload = dict(event.payload)
        status = str(payload.get("status") or "").lower()
        error = payload.get("error")
        failed = (
            event.kind == "run.failed"
            or event.kind.endswith(".failed")
            or status in {"failed", "failure", "error", "rejected"}
            or bool(error)
            or (event.kind == "tool.completed" and payload.get("ok") is False)
        )
        if not failed:
            continue
        if event.kind == "run.failed":
            terminal_failure_id = event.event_id
        normalized_error = dict(error) if isinstance(error, Mapping) else {}
        if error and not normalized_error:
            normalized_error["message"] = str(error)
        if not normalized_error:
            message = payload.get("message") or payload.get("reason")
            normalized_error = {"message": str(message)} if message else {}
        rows.append({
            **_timeline_event(event, framework),
            "error": _bounded_value(normalized_error),
            "classification": "terminal" if event.kind == "run.failed" else "contributing",
        })
    contributing_ids = [row["event_id"] for row in rows if row["event_id"] != terminal_failure_id]
    primary_failure_id = contributing_ids[0] if contributing_ids else terminal_failure_id
    for row in rows:
        if row["event_id"] == primary_failure_id and row["event_id"] != terminal_failure_id:
            row["classification"] = "primary"
        if row["event_id"] == terminal_failure_id:
            row["contributing_failure_event_ids"] = contributing_ids
            row["primary_failure_event_id"] = primary_failure_id
            row["failure_count"] = len(rows)
    return rows


def _langgraph_visualization(resolved_spec: Mapping[str, Any], operations: Sequence[Mapping[str, Any]], framework: str) -> LangGraphVisualization | None:
    if framework != "langgraph":
        return None
    config = resolved_spec.get("config") if isinstance(resolved_spec.get("config"), Mapping) else {}
    graph = config.get("graph") if isinstance(config.get("graph"), Mapping) else {}
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    if not nodes and not edges:
        return None
    visits = [
        {
            "operation_id": row.get("operation_id"),
            "visit_index": row.get("visit_index"),
            "status": row.get("status"),
            "topology_ref": row.get("topology_ref"),
            "framework_details": row.get("framework_details"),
        }
        for row in operations
        if isinstance(row.get("topology_ref"), Mapping)
    ]
    execution_plan = next((row.get("execution_plan") for row in reversed(operations) if row.get("execution_plan")), graph.get("execution_plan") or graph.get("executionPlan") or [])
    selected_route = next((row.get("route") for row in reversed(operations) if row.get("route")), graph.get("selected_route") or graph.get("selectedRoute"))
    return {
        "id": TRACE_VISUALIZATION_LANGGRAPH,
        "nodes": _bounded_value(nodes),
        "edges": _bounded_value(edges),
        "execution_plan": _bounded_value(execution_plan),
        "selected_route": selected_route,
        "visits": _bounded_value(visits),
    }


def _hermes_visualization(events: Sequence[AgentRuntimeEvent], failures: Sequence[Mapping[str, Any]]) -> HermesSessionVisualization | None:
    hermes_events = [event for event in events if (event.source_metadata or {}).get("framework") == "hermes"]
    if not hermes_events:
        return None
    session_id = None
    upstream_run_id = None
    for event in hermes_events:
        payload = event.payload
        session_id = payload.get("session_id") or session_id
        upstream_run_id = payload.get("upstream_run_id") or upstream_run_id
    subagent_events = [event for event in hermes_events if event.kind.startswith("subagent.")]
    if session_id is None and upstream_run_id is None and not subagent_events:
        return None
    hermes_event_ids = {event.event_id for event in hermes_events}
    return {
        "id": TRACE_VISUALIZATION_HERMES,
        "session_id": session_id,
        "upstream_run_id": upstream_run_id,
        "reasoning": [_timeline_event(event, "hermes") for event in hermes_events if event.kind == "reasoning.available"],
        "approvals": [_timeline_event(event, "hermes") for event in hermes_events if event.kind.startswith("approval.")],
        "tools": [_timeline_event(event, "hermes") for event in hermes_events if event.kind.startswith("tool.")],
        "subagents": [_timeline_event(event, "hermes") for event in subagent_events],
        "failures": _bounded_value([
            row for row in failures
            if row.get("event_id") in hermes_event_ids
            or row.get("kind") == "run.failed"
        ]),
    }


def build_canonical_trace_projection(
    *,
    events: Sequence[AgentRuntimeEvent],
    resolved_spec: Mapping[str, Any],
    framework: str,
) -> dict[str, Any]:
    ordered = sorted(events, key=lambda event: (event.sequence, event.event_id))
    operations = _operations(ordered, framework)
    failures = _failure_rows(ordered, framework)
    visualizations: dict[str, Any] = {
        TRACE_VISUALIZATION_GENERIC: {"id": TRACE_VISUALIZATION_GENERIC}
    }
    langgraph = _langgraph_visualization(resolved_spec, operations, framework)
    if langgraph is not None:
        visualizations[TRACE_VISUALIZATION_LANGGRAPH] = langgraph
    hermes = _hermes_visualization(ordered, failures)
    if hermes is not None:
        visualizations[TRACE_VISUALIZATION_HERMES] = hermes
    return {
        "events": [_timeline_event(event, framework) for event in ordered],
        "operations": operations,
        "tools": _event_rows(ordered, "tool.", framework),
        "approvals": _event_rows(ordered, "approval.", framework),
        "subagents": _event_rows(ordered, "subagent.", framework),
        "artifacts": _event_rows(ordered, "artifact.", framework),
        "failures": failures,
        "visualizations": visualizations,
    }
