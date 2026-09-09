from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

from app.agent_workflows.parallel_projection_contracts import (
    PARALLEL_EVENT_NAMES,
    PARALLEL_PROJECTED_WORKER_STATUSES,
    PARALLEL_TERMINAL_WORKER_STATUSES,
    PARALLEL_WORKER_STATUS_BY_EVENT,
    ParallelEventName,
)


def parallel_span_refs(event_name: str, data: Mapping[str, Any]) -> tuple[str | None, str | None]:
    dispatch_id = str(data.get("dispatch_id") or "")
    if not dispatch_id or event_name not in PARALLEL_EVENT_NAMES:
        return None, None
    dispatch_span_id = f"dispatch:{dispatch_id}"
    if event_name.startswith("worker.") and data.get("work_id"):
        attempt = max(1, int(data.get("attempt") or 1))
        return f"worker:{data['work_id']}:attempt:{attempt}", dispatch_span_id
    run_id = str(data.get("agent_run_id") or data.get("run_id") or "")
    return dispatch_span_id, f"run:{run_id}" if run_id else None


def enrich_parallel_event(event_name: str, data: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(data)
    span_id, parent_span_id = parallel_span_refs(event_name, result)
    if span_id:
        result.setdefault("span_id", span_id)
    if parent_span_id:
        result.setdefault("parent_span_id", parent_span_id)
    return result


def project_parallel_events(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    journal: list[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    attempts: Dict[tuple[str, int], Dict[str, Any]] = {}
    barrier_reached = False
    aggregation_state = "pending"
    for envelope in events:
        event_name = str(envelope.get("event") or "")
        raw_data = envelope.get("data") if isinstance(envelope.get("data"), Mapping) else {}
        data = enrich_parallel_event(event_name, raw_data)
        journal.append({"event": event_name, "data": deepcopy(data)})
        if event_name.startswith(("dispatch.", "aggregation.")):
            summary.update(data)
        if event_name == ParallelEventName.BARRIER_REACHED:
            barrier_reached = True
        if event_name == ParallelEventName.DISPATCH_CANCELLED:
            aggregation_state = "cancelled"
        elif event_name == ParallelEventName.AGGREGATION_PARTIAL and aggregation_state != "cancelled":
            aggregation_state = "partial"
            barrier_reached = True
        elif event_name == ParallelEventName.AGGREGATION_COMPLETED and aggregation_state == "pending":
            aggregation_state = "completed"
            barrier_reached = True
        work_id = str(data.get("work_id") or "")
        status = PARALLEL_WORKER_STATUS_BY_EVENT.get(event_name)
        if status and work_id:
            attempt = max(1, int(data.get("attempt") or 1))
            key = (work_id, attempt)
            previous = attempts.get(key, {})
            previous_status = str(previous.get("status") or "")
            if previous_status in PARALLEL_TERMINAL_WORKER_STATUSES and status not in PARALLEL_TERMINAL_WORKER_STATUSES:
                status = previous_status
            attempts[key] = {**previous, **data, "event": event_name, "status": status, "attempt": attempt}
    latest_by_work: Dict[str, Dict[str, Any]] = {}
    for (work_id, _attempt), item in attempts.items():
        if work_id not in latest_by_work or int(item["attempt"]) > int(latest_by_work[work_id]["attempt"]):
            latest_by_work[work_id] = item
    counts = {status: 0 for status in PARALLEL_PROJECTED_WORKER_STATUSES}
    for item in latest_by_work.values():
        status = str(item.get("status") or "")
        if status in counts:
            counts[status] += 1
    return {
        "journal": journal,
        "summary": {
            **counts,
            **summary,
            "barrier_state": "reached" if barrier_reached else "pending",
            "aggregation_state": aggregation_state,
        },
    }
