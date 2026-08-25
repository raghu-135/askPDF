from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from app.agent_workflows.trace_otel import _artifacts_from_refs
from app.agent_workflows.trace_sanitization import _as_dict, _as_list, _bounded_value, _clean_dict
from app.time_utils import iso_utc_z


DEBUG_PAYLOAD_VERSION = 2


def is_current_debug_payload(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("version") == DEBUG_PAYLOAD_VERSION
        and isinstance(value.get("diagnostics"), dict)
        and isinstance(value.get("events"), list)
        and isinstance(value.get("operations"), list)
    )


INTERRUPT_EVENT_NAMES = {
    "pending": "interrupt.requested",
    "requested": "interrupt.requested",
    "resumed": "interrupt.resumed",
    "rejected": "interrupt.rejected",
    "expired": "interrupt.expired",
}
ROOT_LIFECYCLE_EVENT_NAMES = {
    "checkpoint.created",
    "resume.requested",
    "resume.applied",
    "graph.resumed",
}


def _interrupt_event_name(interrupt: Mapping[str, Any], event_name: Optional[str] = None) -> str:
    if event_name:
        return event_name
    status = str(interrupt.get("status") or "").lower()
    return INTERRUPT_EVENT_NAMES.get(status, "interrupt.requested")


def _interrupt_decision(interrupt: Mapping[str, Any]) -> Dict[str, Any]:
    return _as_dict(interrupt.get("decision"))


def _interrupt_action(interrupt: Mapping[str, Any]) -> Any:
    decision = _interrupt_decision(interrupt)
    return decision.get("action") or decision.get("requested_action") or interrupt.get("action") or interrupt.get("default_action")


def build_interrupt_trace_event(
    interrupt: Mapping[str, Any],
    *,
    event_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a bounded root-span event for a HITL interrupt lifecycle change."""

    data = _as_dict(interrupt)
    decision = _interrupt_decision(data)
    attrs = _clean_dict(
        {
            "askpdf.interrupt.id": data.get("interrupt_id"),
            "askpdf.interrupt.gate_id": data.get("gate_id"),
            "askpdf.node.id": data.get("node_id"),
            "askpdf.interrupt.target_node_id": data.get("target_node_id"),
            "askpdf.interrupt.phase": data.get("phase"),
            "askpdf.interrupt.mode": data.get("mode"),
            "askpdf.interrupt.type": data.get("type"),
            "askpdf.interrupt.status": data.get("status"),
            "askpdf.interrupt.action": _interrupt_action(data),
            "askpdf.interrupt.selected_option_ids": decision.get("selected_option_ids"),
            "askpdf.interrupt.requested_at": data.get("requested_at"),
            "askpdf.interrupt.expires_at": data.get("expires_at"),
            "askpdf.interrupt.resume_version": data.get("resume_version"),
        }
    )
    event = {
        "name": _interrupt_event_name(data, event_name),
        "attributes": attrs,
        "input": _clean_dict(
            {
                "title": _bounded_value(data.get("title")),
                "prompt": _bounded_value(data.get("prompt")),
                "body": _bounded_value(data.get("body")),
                "input_summary": _bounded_value(data.get("input_summary")),
            }
        ),
        "output": _clean_dict(
            {
                "proposed_action": _bounded_value(data.get("proposed_action")),
                "proposed_tool": _bounded_value(data.get("proposed_tool")),
                "proposed_memory": _bounded_value(data.get("proposed_memory")),
                "proposed_final_answer": _bounded_value(data.get("proposed_final_answer")),
                "options": _bounded_value(data.get("options")),
                "decision": _bounded_value(decision),
            }
        ),
    }
    return {key: value for key, value in event.items() if value not in (None, "", [], {})}


def _is_interrupt_event(event: Any) -> bool:
    return isinstance(event, dict) and str(event.get("name") or "").startswith("interrupt.")


def _interrupt_event_key(event: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    attrs = _as_dict(event.get("attributes"))
    return (
        event.get("name"),
        attrs.get("askpdf.interrupt.id"),
        attrs.get("askpdf.interrupt.action"),
    )


def _root_event_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    if _is_interrupt_event(event):
        return (*_interrupt_event_key(event), None)
    attrs = _as_dict(event.get("attributes"))
    output = _as_dict(event.get("output"))
    if attrs.get("event_id"):
        return (event.get("name"), "event_id", attrs.get("event_id"))
    return (
        event.get("name"),
        attrs.get("askpdf.interrupt.id") or output.get("interrupt_id"),
        attrs.get("askpdf.checkpoint.thread_id"),
        attrs.get("askpdf.resume.action") or attrs.get("askpdf.status"),
    )


def build_runtime_trace_event(
    name: str,
    *,
    attributes: Optional[Mapping[str, Any]] = None,
    input_data: Any = None,
    output_data: Any = None,
) -> Dict[str, Any]:
    """Build a bounded root-span lifecycle event."""

    event = {
        "name": name,
        "attributes": _clean_dict(_bounded_value(dict(attributes or {}))),
        "input": _bounded_value(input_data),
        "output": _bounded_value(output_data),
    }
    return {key: value for key, value in event.items() if value not in (None, "", [], {})}


def _find_root_span(spans: List[Any]) -> Optional[Dict[str, Any]]:
    return next(
        (
            span
            for span in spans
            if isinstance(span, dict)
            and (span.get("parent_span_id") is None or str(span.get("span_id") or "").startswith("run:"))
        ),
        spans[0] if spans and isinstance(spans[0], dict) else None,
    )


def _append_root_event(trace: Dict[str, Any], event: Mapping[str, Any]) -> None:
    spans = trace.get("spans") if isinstance(trace.get("spans"), list) else []
    root_span = _find_root_span(spans)
    if root_span is None:
        return
    existing = _as_list(root_span.get("events"))
    event_key = _root_event_key(event)
    if not any(_root_event_key(item) == event_key for item in existing if isinstance(item, dict)):
        root_span["events"] = [*existing, dict(event)]


def _interrupt_events_from_trace(trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        dict(event)
        for span in _as_list(trace.get("spans"))
        for event in _as_list(_as_dict(span).get("events"))
        if _is_interrupt_event(event)
    ]


def _interrupt_summary(trace: Mapping[str, Any]) -> Dict[str, Any]:
    events = _interrupt_events_from_trace(trace)
    compact_events = [
        {
            "name": event.get("name"),
            "attributes": _as_dict(event.get("attributes")),
            "input": _as_dict(event.get("input")),
            "output": _as_dict(event.get("output")),
        }
        for event in events
    ]
    last_attrs = _as_dict(compact_events[-1].get("attributes")) if compact_events else {}
    return {
        "interruptCount": len(compact_events),
        "interrupts": compact_events,
        "lastInterruptStatus": last_attrs.get("askpdf.interrupt.status"),
        "lastInterruptAction": last_attrs.get("askpdf.interrupt.action"),
    }


def append_interrupt_event_to_debug_payload(
    debug_payload: Any,
    interrupt: Mapping[str, Any],
    *,
    event_name: Optional[str] = None,
    run_status: Optional[str] = None,
    completed_at: Any = None,
) -> Any:
    """Append a HITL event to an already-stored canonical debug payload."""

    if not is_current_debug_payload(debug_payload):
        return debug_payload
    trace = debug_payload.get("trace") if isinstance(debug_payload.get("trace"), dict) else None
    if trace is None:
        return debug_payload
    spans = trace.get("spans") if isinstance(trace.get("spans"), list) else []
    if not spans:
        return debug_payload

    event = build_interrupt_trace_event(interrupt, event_name=event_name)
    root_span = _find_root_span(spans)
    if root_span is None:
        return debug_payload

    _append_root_event(trace, event)

    if run_status:
        trace["status"] = run_status
        root_span["status"] = run_status
        root_attrs = dict(root_span.get("attributes") or {})
        root_attrs["askpdf.status"] = run_status
        root_span["attributes"] = root_attrs
        trace_attrs = dict(trace.get("attributes") or {})
        trace_attrs["askpdf.status"] = run_status
        trace["attributes"] = trace_attrs
    if completed_at is not None or run_status == "running":
        value = iso_utc_z(completed_at) if completed_at is not None else None
        trace["completed_at"] = value
        root_span["end_time"] = value

    summary = dict(debug_payload.get("summary") or {})
    if run_status:
        summary["status"] = run_status
    summary.update(_interrupt_summary(trace))
    return {**debug_payload, "trace": trace, "summary": summary}


def append_runtime_event_to_debug_payload(
    debug_payload: Any,
    event_name: str,
    *,
    attributes: Optional[Mapping[str, Any]] = None,
    input_data: Any = None,
    output_data: Any = None,
    run_status: Optional[str] = None,
    completed_at: Any = None,
) -> Any:
    """Append a generic root lifecycle event to a stored trace payload."""

    if not is_current_debug_payload(debug_payload):
        return debug_payload
    trace = debug_payload.get("trace") if isinstance(debug_payload.get("trace"), dict) else None
    if trace is None:
        return debug_payload
    event = build_runtime_trace_event(
        event_name,
        attributes=attributes,
        input_data=input_data,
        output_data=output_data,
    )
    _append_root_event(trace, event)

    spans = trace.get("spans") if isinstance(trace.get("spans"), list) else []
    root_span = _find_root_span(spans)
    if run_status:
        trace["status"] = run_status
        if root_span is not None:
            root_span["status"] = run_status
            root_attrs = dict(root_span.get("attributes") or {})
            root_attrs["askpdf.status"] = run_status
            root_span["attributes"] = root_attrs
        trace_attrs = dict(trace.get("attributes") or {})
        trace_attrs["askpdf.status"] = run_status
        trace["attributes"] = trace_attrs
    if completed_at is not None or run_status == "running":
        value = iso_utc_z(completed_at) if completed_at is not None else None
        trace["completed_at"] = value
        if root_span is not None:
            root_span["end_time"] = value

    summary = dict(debug_payload.get("summary") or {})
    if run_status:
        summary["status"] = run_status
    summary.update(_interrupt_summary(trace))
    return {**debug_payload, "trace": trace, "summary": summary}


def _rebuild_trace_refs(trace: Dict[str, Any]) -> None:
    links: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    for span in _as_list(trace.get("spans")):
        if not isinstance(span, dict):
            continue
        for link in _as_list(span.get("links")):
            if isinstance(link, dict):
                links.append({"span_id": span.get("span_id"), **link})
        artifacts.extend(_artifacts_from_refs(_as_dict(span.get("output")).get("refs"), span_id=str(span.get("span_id") or "")))
    trace["links"] = links
    trace["artifacts"] = artifacts


def _merge_root_span(base_root: Dict[str, Any], incoming_root: Mapping[str, Any]) -> None:
    for event in _as_list(incoming_root.get("events")):
        if isinstance(event, dict):
            existing = _as_list(base_root.get("events"))
            if not any(_root_event_key(item) == _root_event_key(event) for item in existing if isinstance(item, dict)):
                base_root["events"] = [*existing, dict(event)]
    attrs = dict(base_root.get("attributes") or {})
    attrs.update(_as_dict(incoming_root.get("attributes")))
    base_root["attributes"] = attrs
    output = dict(base_root.get("output") or {})
    output.update(_as_dict(incoming_root.get("output")))
    base_root["output"] = output


def merge_debug_payloads(
    base_payload: Any,
    incoming_payload: Any,
    *,
    resolved_spec: Mapping[str, Any],
    run_status: Optional[str] = None,
    completed_at: Any = None,
    chat_turn_id: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Merge a later execution phase into an already-stored debug payload."""

    if not is_current_debug_payload(base_payload):
        return base_payload
    if not is_current_debug_payload(incoming_payload):
        return base_payload
    base_trace = base_payload.get("trace") if isinstance(base_payload.get("trace"), dict) else None
    incoming_trace = incoming_payload.get("trace") if isinstance(incoming_payload.get("trace"), dict) else None
    if base_trace is None or incoming_trace is None:
        return base_payload

    base_spans = base_trace.get("spans") if isinstance(base_trace.get("spans"), list) else []
    incoming_spans = incoming_trace.get("spans") if isinstance(incoming_trace.get("spans"), list) else []
    base_root = _find_root_span(base_spans)
    incoming_root = _find_root_span(incoming_spans)
    if base_root is not None and incoming_root is not None:
        _merge_root_span(base_root, incoming_root)

    def semantic_key(span: Mapping[str, Any]) -> tuple[Any, ...] | None:
        attrs = _as_dict(span.get("attributes"))
        kind = str(span.get("kind") or "")
        node_id = attrs.get("askpdf.node.id")
        visit = attrs.get("askpdf.node.visit_index")
        if node_id and visit is not None:
            return ("node", kind, str(node_id), int(visit), str(span.get("status") or ""))
        tool_name = attrs.get("tool.name") or attrs.get("askpdf.tool.name")
        if tool_name:
            return (
                "tool", str(tool_name), str(attrs.get("askpdf.caller_node") or ""),
                int(attrs.get("askpdf.caller_visit_index") or 1),
                str(attrs.get("askpdf.parallel.work_id") or ""),
                str(attrs.get("askpdf.tool.argument_hash") or ""),
                str(span.get("status") or ""),
            )
        return None

    existing_ids = {str(span.get("span_id")) for span in base_spans if isinstance(span, dict)}
    existing_semantics = {key for span in base_spans if isinstance(span, dict) if (key := semantic_key(span)) is not None}
    id_map: Dict[str, str] = {}
    skipped_ids: set[str] = set()
    appended: List[Dict[str, Any]] = []
    for span in incoming_spans:
        if not isinstance(span, dict):
            continue
        span_id = str(span.get("span_id") or "")
        if not span_id or span is incoming_root:
            continue
        key = semantic_key(span)
        if key is not None and key in existing_semantics:
            skipped_ids.add(span_id)
            continue
        new_id = span_id
        if new_id in existing_ids:
            suffix = 1
            while f"{span_id}:resume:{suffix}" in existing_ids:
                suffix += 1
            new_id = f"{span_id}:resume:{suffix}"
        id_map[span_id] = new_id
        existing_ids.add(new_id)

    for span in incoming_spans:
        if not isinstance(span, dict):
            continue
        span_id = str(span.get("span_id") or "")
        if not span_id or span is incoming_root:
            continue
        if span_id in skipped_ids or str(span.get("parent_span_id") or "") in skipped_ids:
            continue
        merged_span = dict(span)
        merged_span["span_id"] = id_map.get(span_id, span_id)
        parent_id = merged_span.get("parent_span_id")
        if isinstance(parent_id, str) and parent_id in id_map:
            merged_span["parent_span_id"] = id_map[parent_id]
        appended.append(merged_span)
    base_trace["spans"] = [*base_spans, *appended]

    merged_metrics = {
        **_as_dict(base_trace.get("metrics")),
        **_as_dict(incoming_trace.get("metrics")),
        **dict(metrics or {}),
    }
    base_trace["metrics"] = merged_metrics
    if run_status:
        base_trace["status"] = run_status
    if completed_at is not None or run_status == "running":
        base_trace["completed_at"] = iso_utc_z(completed_at) if completed_at is not None else None
    if chat_turn_id is not None:
        base_trace["chat_turn_id"] = chat_turn_id

    if base_root is not None:
        if run_status:
            base_root["status"] = run_status
            root_attrs = dict(base_root.get("attributes") or {})
            root_attrs["askpdf.status"] = run_status
            base_root["attributes"] = root_attrs
            trace_attrs = dict(base_trace.get("attributes") or {})
            trace_attrs["askpdf.status"] = run_status
            base_trace["attributes"] = trace_attrs
        if completed_at is not None or run_status == "running":
            base_root["end_time"] = iso_utc_z(completed_at) if completed_at is not None else None

    _rebuild_trace_refs(base_trace)
    from app.agent_workflows.trace_summary import _build_summary_from_trace
    summary = {
        **_build_summary_from_trace(base_trace, resolved_spec),
        **_as_dict(base_payload.get("summary")),
        **_as_dict(incoming_payload.get("summary")),
    }

    def merge_rows(name: str, key_fields: tuple[str, ...]) -> List[Dict[str, Any]]:
        rows: Dict[tuple[Any, ...], Dict[str, Any]] = {}
        for row in [*_as_list(base_payload.get(name)), *_as_list(incoming_payload.get(name))]:
            if not isinstance(row, dict):
                continue
            key = tuple(row.get(field) for field in key_fields)
            rows[key] = {**rows.get(key, {}), **row}
        return sorted(rows.values(), key=lambda row: (int(row.get("sequence") or 0), str(row.get("event_id") or row.get("operation_id") or "")))

    merged_events = merge_rows("events", ("event_id",))
    from app.agent_workflows.canonical_trace import build_trace_diagnostics
    from app.runtime.contracts import AgentRuntimeEvent
    diagnostic_events = [
        AgentRuntimeEvent(
            event_id=str(row.get("event_id") or f"merged:{index + 1}"),
            run_id=str(base_trace.get("run_id") or "merged"),
            sequence=int(row.get("sequence") or index + 1),
            attempt=int(row.get("attempt") or 1),
            kind=str(row.get("kind") or "runtime.event"),
            payload=_as_dict(row.get("payload")),
            occurred_at=row.get("occurred_at"),
        )
        for index, row in enumerate(merged_events)
    ]
    payload = {
        **base_payload,
        "trace": base_trace,
        "summary": summary,
        "events": merged_events,
        "operations": merge_rows("operations", ("operation_id", "visit_index")),
        "tools": merge_rows("tools", ("event_id",)),
        "approvals": merge_rows("approvals", ("event_id",)),
        "subagents": merge_rows("subagents", ("event_id",)),
        "artifacts": merge_rows("artifacts", ("event_id",)),
        "diagnostics": build_trace_diagnostics(diagnostic_events),
        "visualizations": {
            **_as_dict(base_payload.get("visualizations")),
            **_as_dict(incoming_payload.get("visualizations")),
        },
    }
    summary.pop("nodes", None)
    summary.pop("usedNodeCount", None)
    summary.pop("availableNodeCount", None)
    summary["operations"] = payload["operations"]
    summary["tools"] = payload["tools"]
    summary["usedOperationCount"] = len({
        str(row.get("operation_id"))
        for row in payload["operations"]
        if row.get("operation_id") and row.get("status") != "skipped"
    })
    summary["errorCount"] = int(payload["diagnostics"]["summary"].get("failure_count") or 0)
    summary.pop("errors", None)
    base_trace.update({
        key: payload[key]
        for key in ("events", "operations", "tools", "approvals", "subagents", "artifacts", "diagnostics", "visualizations")
    })
    detail_by_visit: Dict[tuple[str, int], Dict[str, Any]] = {}
    for detail in [*_as_list(base_payload.get("details")), *_as_list(incoming_payload.get("details"))]:
        if not isinstance(detail, dict):
            continue
        try:
            visit_index = max(1, int(detail.get("visit_index") or 1))
        except (TypeError, ValueError):
            visit_index = 1
        detail_by_visit[(str(detail.get("operation_id") or ""), visit_index)] = dict(detail)
    if detail_by_visit:
        from app.agent_workflows.trace_details import TRACE_DETAIL_RUN_LIMIT, trace_detail_size

        retained_details: List[Dict[str, Any]] = []
        retained_size = 0
        omitted_visit_count = 0
        merge_limit_reached = False
        for detail in detail_by_visit.values():
            retained = detail
            detail_size = trace_detail_size(retained)
            if retained_size + detail_size > TRACE_DETAIL_RUN_LIMIT:
                merge_limit_reached = True
                retained = {
                    "operation_id": detail.get("operation_id"),
                    "operation_type": detail.get("operation_type"),
                    "visit_index": detail.get("visit_index"),
                    "status": detail.get("status"),
                    "safety": {
                        "truncated": True,
                        "run_limit_reached": True,
                        "redacted_fields": _as_list(_as_dict(detail.get("safety")).get("redacted_fields")),
                        "truncated_fields": [*_as_list(_as_dict(detail.get("safety")).get("truncated_fields")), "detail"],
                        "omitted_fields": _as_list(_as_dict(detail.get("safety")).get("omitted_fields")),
                    },
                }
                detail_size = trace_detail_size(retained)
            if retained_size + detail_size > TRACE_DETAIL_RUN_LIMIT:
                omitted_visit_count += 1
                continue
            retained_details.append(retained)
            retained_size += detail_size
        payload["details"] = retained_details
    incoming_final = _as_dict(incoming_payload.get("final_output"))
    if incoming_final:
        payload["final_output"] = incoming_final
    elif isinstance(base_payload.get("final_output"), dict):
        payload["final_output"] = base_payload["final_output"]
    detail_safety = {
        **_as_dict(base_payload.get("detail_safety")),
        **_as_dict(incoming_payload.get("detail_safety")),
    }
    if detail_by_visit:
        detail_safety.update({
            "size_bytes": retained_size,
            "run_limit_bytes": TRACE_DETAIL_RUN_LIMIT,
            "truncated": bool(detail_safety.get("truncated") or merge_limit_reached),
            "omitted_visit_count": omitted_visit_count,
        })
    if detail_safety:
        payload["detail_safety"] = detail_safety
    return payload
