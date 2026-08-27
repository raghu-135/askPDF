from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence, TypedDict

from app.agent_workflows.trace_sanitization import _bounded_value
from app.runtime.contracts import AgentRuntimeEvent


class TraceVisualizationId(str, Enum):
    GENERIC_TIMELINE = "generic.timeline"
    GENERIC_PARALLEL = "generic.parallel"
    LANGGRAPH_GRAPH = "langgraph.graph"
    HERMES_SESSION = "hermes.session"


class GenericTimelineVisualization(TypedDict):
    id: str


class GenericParallelVisualization(TypedDict):
    id: str
    group_ids: list[str]


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


class AgentTraceParallelAttempt(TypedDict, total=False):
    attempt: int
    status: str
    started_at: str
    completed_at: str
    duration_ms: float
    first_sequence: int
    last_sequence: int
    event_ids: list[str]
    failure_event_ids: list[str]
    caused_by_event_ids: list[str]
    related_event_ids: list[str]


class AgentTraceParallelMember(TypedDict, total=False):
    member_id: str
    work_id: str
    operation_id: str
    operation_label: str
    tool_call_id: str
    tool_name: str
    subagent_id: str
    ordinal: int
    status: str
    first_sequence: int
    last_sequence: int
    event_ids: list[str]
    attempts: list[AgentTraceParallelAttempt]


class AgentTraceParallelBarrier(TypedDict, total=False):
    status: str
    event_id: str
    sequence: int
    occurred_at: str
    result_count: int


class AgentTraceParallelAggregation(TypedDict, total=False):
    status: str
    event_id: str
    sequence: int
    occurred_at: str
    counts: dict[str, int]


class AgentTraceParallelGroup(TypedDict, total=False):
    group_id: str
    parent_operation_id: str
    topology_ref: dict[str, Any]
    status: str
    planned: int
    first_sequence: int
    last_sequence: int
    started_at: str
    completed_at: str
    duration_ms: float
    event_ids: list[str]
    members: list[AgentTraceParallelMember]
    barrier: AgentTraceParallelBarrier
    aggregation: AgentTraceParallelAggregation


TRACE_VISUALIZATION_GENERIC = TraceVisualizationId.GENERIC_TIMELINE.value
TRACE_VISUALIZATION_PARALLEL = TraceVisualizationId.GENERIC_PARALLEL.value
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
    group_id = _parallel_group_id(payload)
    return {
        "event_id": event.event_id,
        "sequence": event.sequence,
        "attempt": event.attempt,
        "kind": event.kind,
        "occurred_at": event.occurred_at,
        "operation_id": payload.get("operation_id"),
        "parent_operation_id": payload.get("parent_operation_id") or payload.get("parent_id"),
        "parallel_group_id": group_id,
        "parallel_member_id": payload.get("work_id") or (payload.get("operation_id") if group_id else None),
        "parallel_attempt": _strict_positive_int(payload.get("attempt") if payload.get("attempt") is not None else event.attempt, "attempt") if group_id else None,
        "status": payload.get("status"),
        "payload": _bounded_value(payload),
        "framework_details": _bounded_value(framework_details),
    }


_PARALLEL_EVENT_PREFIXES = ("dispatch.", "worker.", "aggregation.")
_PARALLEL_TERMINAL_MEMBER_STATUSES = {"completed", "skipped", "failed", "timed_out", "cancelled"}


class TraceProjectionError(ValueError):
    pass


def _parallel_group_id(payload: Mapping[str, Any]) -> str | None:
    for key in ("parallel_group_id", "dispatch_id", "wave_id"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _strict_positive_int(value: Any, field: str, *, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise TraceProjectionError(f"parallel event has invalid {field}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TraceProjectionError(f"parallel event has invalid {field}") from exc
    if result < 1:
        raise TraceProjectionError(f"parallel event has invalid {field}")
    return result


def _optional_nonnegative_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TraceProjectionError(f"parallel event has invalid {field}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TraceProjectionError(f"parallel event has invalid {field}") from exc
    if result < 0:
        raise TraceProjectionError(f"parallel event has invalid {field}")
    return result


def _parallel_status(kind: str, payload: Mapping[str, Any]) -> str:
    explicit = str(payload.get("status") or "").strip().lower()
    if explicit:
        return "cancelled" if explicit == "canceled" else explicit
    if kind.startswith("worker."):
        return {
            "worker.queued": "queued",
            "worker.started": "active",
            "worker.progress": "active",
            "worker.retrying": "retrying",
        }.get(kind, kind.removeprefix("worker."))
    return "running"


def build_parallel_groups(events: Sequence[AgentRuntimeEvent]) -> list[AgentTraceParallelGroup]:
    groups: dict[str, dict[str, Any]] = {}
    member_owners: dict[str, str] = {}
    ordered = sorted(events, key=lambda event: (event.sequence, event.event_id))
    for event in ordered:
        payload = dict(event.payload)
        group_id = _parallel_group_id(payload)
        dispatch_mode = str(payload.get("dispatch_mode") or payload.get("mode") or "parallel").strip().lower()
        if dispatch_mode not in {"serial", "parallel"}:
            raise TraceProjectionError(f"parallel event {event.event_id} has invalid dispatch mode")
        if dispatch_mode == "serial":
            continue
        is_lifecycle = event.kind.startswith(_PARALLEL_EVENT_PREFIXES)
        is_correlated_operation = event.kind.startswith("operation.") and group_id is not None
        if not is_lifecycle and not is_correlated_operation:
            continue
        if group_id is None:
            raise TraceProjectionError(f"parallel event {event.event_id} is missing a group identity")
        group = groups.setdefault(group_id, {
            "group_id": group_id,
            "status": "running",
            "planned": 0,
            "first_sequence": event.sequence,
            "last_sequence": event.sequence,
            "started_at": event.occurred_at,
            "completed_at": None,
            "event_ids": [],
            "members": {},
            "barrier": {"status": "pending"},
            "aggregation": {"status": "pending", "counts": {}},
        })
        group["last_sequence"] = max(int(group["last_sequence"]), event.sequence)
        group["event_ids"].append(event.event_id)
        parent = payload.get("parent_operation_id") or payload.get("parent_id")
        if parent:
            prior_parent = group.get("parent_operation_id")
            if prior_parent and str(prior_parent) != str(parent):
                raise TraceProjectionError(f"parallel group {group_id} has conflicting parent operations")
            group["parent_operation_id"] = str(parent)
        topology_ref = payload.get("topology_ref")
        if isinstance(topology_ref, Mapping):
            group["topology_ref"] = _bounded_value(dict(topology_ref))
        planned = _optional_nonnegative_int(payload.get("planned"), "planned")
        if planned is not None:
            group["planned"] = planned

        if event.kind in {"dispatch.planned", "dispatch.started"}:
            group["started_at"] = group.get("started_at") or event.occurred_at
        if event.kind == "dispatch.cancelled":
            group["status"] = "cancelled"
            group["completed_at"] = event.occurred_at
            group["aggregation"] = {
                "status": "cancelled", "event_id": event.event_id,
                "sequence": event.sequence, "occurred_at": event.occurred_at,
                "counts": {},
            }
        elif event.kind == "dispatch.barrier_reached":
            group["barrier"] = {
                "status": "reached", "event_id": event.event_id,
                "sequence": event.sequence, "occurred_at": event.occurred_at,
                "result_count": _optional_nonnegative_int(payload.get("result_count"), "result_count") or 0,
            }
        elif event.kind.startswith("aggregation."):
            aggregation_status = {
                "aggregation.partial": "partial",
                "aggregation.failed": "failed",
                "aggregation.cancelled": "cancelled",
            }.get(event.kind, str(payload.get("status") or "completed"))
            counts = {
                key: value
                for key in ("planned", "completed", "failed", "timed_out", "cancelled", "skipped", "retried")
                if (value := _optional_nonnegative_int(payload.get(key), key)) is not None
            }
            group["aggregation"] = {
                "status": aggregation_status, "event_id": event.event_id,
                "sequence": event.sequence, "occurred_at": event.occurred_at,
                "counts": counts,
            }
            group["status"] = aggregation_status
            group["completed_at"] = event.occurred_at

        if not event.kind.startswith("worker.") and not is_correlated_operation:
            continue
        member_value = payload.get("work_id") or payload.get("operation_id")
        if member_value is None or not str(member_value).strip():
            raise TraceProjectionError(f"parallel member event {event.event_id} is missing a member identity")
        member_id = str(member_value).strip()
        prior_owner = member_owners.get(member_id)
        if prior_owner and prior_owner != group_id:
            raise TraceProjectionError(f"parallel member {member_id} belongs to conflicting groups")
        member_owners[member_id] = group_id
        members: dict[str, dict[str, Any]] = group["members"]
        member = members.setdefault(member_id, {
            "member_id": member_id,
            "status": "queued",
            "first_sequence": event.sequence,
            "last_sequence": event.sequence,
            "event_ids": [],
            "attempts": {},
        })
        member["last_sequence"] = max(int(member["last_sequence"]), event.sequence)
        member["event_ids"].append(event.event_id)
        for key in ("work_id", "operation_id", "operation_label", "tool_call_id", "tool_name", "subagent_id"):
            if payload.get(key) is not None:
                member[key] = str(payload[key])
        ordinal = _optional_nonnegative_int(payload.get("ordinal", payload.get("work_ordinal")), "ordinal")
        if ordinal is not None:
            member["ordinal"] = ordinal
        attempt_number = _strict_positive_int(payload.get("attempt") if payload.get("attempt") is not None else event.attempt, "attempt")
        attempts: dict[int, dict[str, Any]] = member["attempts"]
        attempt = attempts.setdefault(attempt_number, {
            "attempt": attempt_number,
            "status": "queued",
            "first_sequence": event.sequence,
            "last_sequence": event.sequence,
            "event_ids": [],
            "failure_event_ids": [],
            "caused_by_event_ids": [],
            "related_event_ids": [],
        })
        attempt["last_sequence"] = max(int(attempt["last_sequence"]), event.sequence)
        attempt["event_ids"].append(event.event_id)
        caused_by = payload.get("caused_by_event_id")
        if caused_by and str(caused_by) not in attempt["caused_by_event_ids"]:
            attempt["caused_by_event_ids"].append(str(caused_by))
        for related_id in _string_list(payload.get("related_event_ids")):
            if related_id not in attempt["related_event_ids"]:
                attempt["related_event_ids"].append(related_id)
        status = _parallel_status(event.kind, payload)
        if event.kind in {"worker.started", "operation.started"}:
            attempt["started_at"] = attempt.get("started_at") or event.occurred_at
        if status in _PARALLEL_TERMINAL_MEMBER_STATUSES:
            attempt["status"] = status
            attempt["completed_at"] = event.occurred_at
            duration = payload.get("duration_ms", payload.get("elapsed_ms"))
            if duration is not None:
                try:
                    attempt["duration_ms"] = max(0.0, float(duration))
                except (TypeError, ValueError) as exc:
                    raise TraceProjectionError(f"parallel event {event.event_id} has invalid duration") from exc
            elif attempt.get("started_at"):
                attempt["duration_ms"] = _duration_ms(attempt["started_at"], event.occurred_at)
            if status in {"failed", "timed_out", "cancelled"}:
                attempt["failure_event_ids"].append(event.event_id)
        elif attempt.get("status") not in _PARALLEL_TERMINAL_MEMBER_STATUSES:
            attempt["status"] = status
        latest_attempt = attempts[max(attempts)]
        member["status"] = latest_attempt["status"]

    result: list[AgentTraceParallelGroup] = []
    for group in groups.values():
        member_rows = []
        for member in group.pop("members").values():
            member["attempts"] = sorted(member["attempts"].values(), key=lambda row: row["attempt"])
            member_rows.append(member)
        group["members"] = sorted(member_rows, key=lambda row: (int(row.get("ordinal", 2**31 - 1)), row["member_id"]))
        if not group.get("planned"):
            group["planned"] = len(group["members"])
        if group.get("completed_at") and group.get("started_at"):
            group["duration_ms"] = _duration_ms(group["started_at"], group["completed_at"])
        if group["status"] == "running" and group["members"] and all(member["status"] in _PARALLEL_TERMINAL_MEMBER_STATUSES for member in group["members"]):
            statuses = {member["status"] for member in group["members"]}
            group["status"] = "failed" if statuses & {"failed", "timed_out"} else "cancelled" if statuses == {"cancelled"} else "completed"
        sanitized = _bounded_value({key: value for key, value in group.items() if value is not None})
        # Empty collections are meaningful required fields in the parallel
        # contract. The generic sanitizer removes them, so restore only the
        # structural containers after all values have been bounded/redacted.
        sanitized.setdefault("event_ids", [])
        sanitized.setdefault("members", [])
        sanitized.setdefault("barrier", {"status": "pending"})
        aggregation = sanitized.setdefault("aggregation", {"status": "pending"})
        aggregation.setdefault("counts", {})
        for member in sanitized["members"]:
            member.setdefault("event_ids", [])
            member.setdefault("attempts", [])
            for attempt in member["attempts"]:
                attempt.setdefault("event_ids", [])
                attempt.setdefault("failure_event_ids", [])
                attempt.setdefault("caused_by_event_ids", [])
                attempt.setdefault("related_event_ids", [])
        result.append(sanitized)
    return sorted(result, key=lambda row: (int(row["first_sequence"]), row["group_id"]))


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
                "parallel_group_id": _parallel_group_id(payload),
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


def _graph_visualization(resolved_spec: Mapping[str, Any], operations: Sequence[Mapping[str, Any]]) -> LangGraphVisualization | None:
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
    hermes_events = [
        event for event in events
        if event.source_metadata.get("visualization_id") == TRACE_VISUALIZATION_HERMES
    ]
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
    parallel_groups = build_parallel_groups(ordered)
    diagnostics = build_trace_diagnostics(ordered)
    failures = diagnostics["failures"]
    visualizations: dict[str, Any] = {
        TRACE_VISUALIZATION_GENERIC: {"id": TRACE_VISUALIZATION_GENERIC}
    }
    if parallel_groups:
        visualizations[TRACE_VISUALIZATION_PARALLEL] = {
            "id": TRACE_VISUALIZATION_PARALLEL,
            "group_ids": [group["group_id"] for group in parallel_groups],
        }
    langgraph = _graph_visualization(resolved_spec, operations)
    if langgraph is not None:
        visualizations[TRACE_VISUALIZATION_LANGGRAPH] = langgraph
    hermes = _hermes_visualization(ordered, failures)
    if hermes is not None:
        visualizations[TRACE_VISUALIZATION_HERMES] = hermes
    return {
        "events": [_timeline_event(event, framework) for event in ordered],
        "operations": operations,
        "parallel_groups": parallel_groups,
        "tools": _event_rows(ordered, "tool.", framework),
        "models": _event_rows(ordered, "llm.", framework),
        "approvals": _event_rows(ordered, "approval.", framework),
        "subagents": _event_rows(ordered, "subagent.", framework),
        "artifacts": _event_rows(ordered, "artifact.", framework),
        "diagnostics": diagnostics,
        "visualizations": visualizations,
    }
