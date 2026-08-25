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


class AgentTraceLocation(TypedDict, total=False):
    operation_id: str
    operation_label: str
    parent_operation_id: str
    tool_call_id: str
    tool_name: str
    subagent_id: str
    approval_id: str
    parallel_group_id: str
    attempt: int
    sequence: int
    topology_ref: dict[str, Any]


class AgentTraceFailure(TypedDict, total=False):
    event_id: str
    kind: str
    classification: str
    code: str
    message: str
    retryable: bool
    occurred_at: str
    location: AgentTraceLocation
    caused_by_event_id: str
    related_event_ids: list[str]
    details: dict[str, Any]


class AgentTraceDiagnostics(TypedDict):
    outcome: str
    summary: dict[str, Any]
    failures: list[AgentTraceFailure]
    groups: list[dict[str, Any]]
    observability_gaps: list[dict[str, Any]]


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
    framework_details = _framework_details(payload, framework)
    if event.kind.startswith("tool."):
        arguments = payload.get("arguments") or payload.get("args") or payload.get("input")
        if isinstance(arguments, Mapping):
            payload.setdefault("provided_argument_names", sorted(str(key) for key in arguments))
        for key in ("arguments", "args", "input"):
            payload.pop(key, None)
    for key in ("response", "runtime_binding", "runtime_metadata", "prompt", "messages", "headers", "framework_details", "framework_metadata"):
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
        "framework_details": _bounded_value(framework_details),
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


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if str(item)] if isinstance(value, (list, tuple, set)) else []


def build_trace_diagnostics(events: Sequence[AgentRuntimeEvent]) -> AgentTraceDiagnostics:
    rows: list[AgentTraceFailure] = []
    terminal: AgentTraceFailure | None = None
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
        cancelled = event.kind.endswith(".cancelled") or status in {"cancelled", "canceled"}
        if not failed and not cancelled:
            continue
        normalized_error = dict(error) if isinstance(error, Mapping) else {}
        if error and not normalized_error:
            normalized_error["message"] = str(error)
        if not normalized_error:
            message = payload.get("message") or payload.get("reason")
            normalized_error = {"message": str(message)} if message else {}
        location = {
            key: value
            for key, value in {
                "operation_id": payload.get("operation_id"),
                "operation_label": payload.get("operation_label") or payload.get("label"),
                "parent_operation_id": payload.get("parent_operation_id") or payload.get("parent_id"),
                "tool_call_id": payload.get("tool_call_id"),
                "tool_name": payload.get("tool_name"),
                "subagent_id": payload.get("subagent_id"),
                "approval_id": payload.get("approval_id"),
                "parallel_group_id": payload.get("parallel_group_id") or payload.get("dispatch_id") or payload.get("wave_id"),
                "attempt": event.attempt,
                "sequence": event.sequence,
                "topology_ref": payload.get("topology_ref") if isinstance(payload.get("topology_ref"), Mapping) else None,
            }.items()
            if value not in (None, "", [], {})
        }
        caused_by = str(payload.get("caused_by_event_id") or normalized_error.get("caused_by_event_id") or "")
        related = _string_list(payload.get("related_event_ids") or normalized_error.get("related_event_ids"))
        row: AgentTraceFailure = {
            "event_id": event.event_id,
            "kind": event.kind,
            "classification": "terminal_summary" if event.kind in {"run.failed", "run.cancelled"} else "cancellation" if cancelled else "contributing",
            "code": str(normalized_error.get("code") or payload.get("code") or event.kind.replace(".", "_")),
            "message": str(normalized_error.get("safe_message") or normalized_error.get("message") or normalized_error.get("raw_message") or payload.get("message") or payload.get("reason") or event.kind),
            "retryable": bool(normalized_error.get("retryable") or payload.get("retryable")),
            "location": _bounded_value(location),
        }
        if event.occurred_at:
            row["occurred_at"] = event.occurred_at
        if caused_by:
            row["caused_by_event_id"] = caused_by
        if related:
            row["related_event_ids"] = related
        details = normalized_error.get("details")
        if isinstance(details, Mapping):
            row["details"] = _bounded_value(dict(details))
        rows.append(row)
        if event.kind in {"run.failed", "run.cancelled"}:
            terminal = row

    by_id = {row["event_id"]: row for row in rows}
    non_terminal = [row for row in rows if row.get("classification") != "terminal_summary" and row.get("classification") != "cancellation"]
    explicit_primary_id = str((terminal or {}).get("caused_by_event_id") or "")
    primary = by_id.get(explicit_primary_id) if explicit_primary_id else None
    visited: set[str] = set()
    while primary is not None and primary.get("caused_by_event_id") and primary["event_id"] not in visited:
        visited.add(primary["event_id"])
        next_primary = by_id.get(str(primary["caused_by_event_id"]))
        if next_primary is None:
            break
        primary = next_primary
    if primary is None:
        primary = non_terminal[0] if non_terminal else terminal
    primary_basis = "explicit_cause" if explicit_primary_id and primary is not None else "earliest_observed"
    parallel_counts: dict[str, int] = {}
    for row in non_terminal:
        group_id = str((row.get("location") or {}).get("parallel_group_id") or "")
        if group_id:
            parallel_counts[group_id] = parallel_counts.get(group_id, 0) + 1
    for row in rows:
        if primary is not None and row["event_id"] == primary["event_id"] and row.get("classification") != "terminal_summary":
            row["classification"] = "primary"
        elif row.get("classification") not in {"terminal_summary", "cancellation"}:
            parallel_group_id = str((row.get("location") or {}).get("parallel_group_id") or "")
            row["classification"] = "downstream" if row.get("caused_by_event_id") else "concurrent" if parallel_group_id and parallel_counts.get(parallel_group_id, 0) > 1 else "contributing"

    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("classification") == "terminal_summary":
            continue
        location = row.get("location") or {}
        key = (
            str(row.get("code") or "runtime_failure"),
            str(location.get("operation_id") or ""),
            str(location.get("tool_name") or ""),
            str(location.get("subagent_id") or ""),
        )
        group = groups.setdefault(key, {
            "code": key[0],
            "location": location,
            "event_ids": [],
            "occurrence_count": 0,
            "classifications": [],
        })
        group["event_ids"].append(row["event_id"])
        group["occurrence_count"] += 1
        if row["classification"] not in group["classifications"]:
            group["classifications"].append(row["classification"])

    gaps = []
    if terminal is not None and terminal.get("kind") == "run.failed" and not non_terminal:
        gaps.append({
            "code": "terminal_failure_without_lower_level_events",
            "message": "The runtime reported a terminal failure without lower-level diagnostic events.",
            "terminal_event_id": terminal["event_id"],
        })
    outcome = str((terminal or {}).get("location", {}).get("status") or ("failed" if terminal and terminal.get("kind") == "run.failed" else "cancelled" if terminal else "completed"))
    summary_source = terminal or primary
    summary = {
        "code": str((summary_source or {}).get("code") or "run_completed"),
        "message": str((summary_source or {}).get("message") or "Run completed without a recorded failure."),
        "retryable": bool((summary_source or {}).get("retryable")),
        "primary_failure_event_id": (primary or {}).get("event_id"),
        "primary_basis": primary_basis if primary is not None else None,
        "location": (primary or {}).get("location") or {},
        "failure_count": len([row for row in rows if row.get("classification") != "cancellation"]),
        "cancellation_count": len([row for row in rows if row.get("classification") == "cancellation"]),
    }
    return {
        "outcome": outcome,
        "summary": _bounded_value(summary),
        "failures": rows,
        "groups": sorted(groups.values(), key=lambda group: (group["event_ids"][0], group["code"])),
        "observability_gaps": gaps,
    }


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
    diagnostics = build_trace_diagnostics(ordered)
    failures = diagnostics["failures"]
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
        "models": _event_rows(ordered, "llm.", framework),
        "approvals": _event_rows(ordered, "approval.", framework),
        "subagents": _event_rows(ordered, "subagent.", framework),
        "artifacts": _event_rows(ordered, "artifact.", framework),
        "diagnostics": diagnostics,
        "visualizations": visualizations,
    }
