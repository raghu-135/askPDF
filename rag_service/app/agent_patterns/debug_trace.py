from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind, Status, StatusCode, set_span_in_context
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from app.agent.tool_registry import get_tool_contract_metadata
from app.agent_patterns.trace import compact_preview
from app.time_utils import iso_utc_z


DEBUG_PAYLOAD_VERSION = 1
TRACE_SCHEMA_VERSION = 1
TRACE_PREVIEW_LIMIT = 900
TRACE_REDACTED_VALUE = "[redacted]"
TRACE_SENSITIVE_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "id_token",
    "password",
    "private_key",
    "refresh_token",
    "resume_token",
    "secret",
    "set_cookie",
    "token",
}
TRACE_NON_SECRET_TOKEN_KEY_PARTS = {
    "cached_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "reasoning_tokens",
    "token_count",
    "token_counts",
    "token_usage",
    "total_tokens",
}

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


class _BufferedSpanExporter(SpanExporter):
    """Local OpenTelemetry exporter used to normalize one agent run."""

    def __init__(self) -> None:
        self.spans: List[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _clean_dict(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _normalized_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_sensitive_key(key: Any) -> bool:
    normalized = _normalized_key(key)
    if any(part in normalized for part in TRACE_NON_SECRET_TOKEN_KEY_PARTS):
        return False
    return any(part in normalized for part in TRACE_SENSITIVE_KEY_PARTS)


def _bounded_value(value: Any, *, key: Any = None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return TRACE_REDACTED_VALUE
    if value in (None, "", [], {}):
        return value
    if isinstance(value, str):
        return compact_preview(value, limit=TRACE_PREVIEW_LIMIT)
    if isinstance(value, list):
        return [_bounded_value(item) for item in value[:50]]
    if isinstance(value, dict):
        return {
            item_key: _bounded_value(item, key=item_key)
            for item_key, item in value.items()
            if item not in (None, "", [], {})
        }
    return value


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str, ensure_ascii=True))
    except Exception:
        return str(value)


def _otel_attr_value(value: Any) -> Any:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list) and all(isinstance(item, (str, bool, int, float)) for item in value):
        return value
    return json.dumps(_jsonable(value), ensure_ascii=True, sort_keys=True)


def _set_attributes(span: Any, attributes: Mapping[str, Any]) -> None:
    for key, value in attributes.items():
        otel_value = _otel_attr_value(value)
        if otel_value is not None:
            span.set_attribute(key, otel_value)


def _parse_time_ns(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _ns_to_iso(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return iso_utc_z(datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc))


def _event_time(value: Any) -> Optional[str]:
    if not value:
        return None
    try:
        return iso_utc_z(value)
    except Exception:
        return None


def _span_status(event: Mapping[str, Any]) -> str:
    status = str(event.get("status") or "").lower()
    if event.get("error") or event.get("ok") is False or status in {"failed", "error"}:
        return "error"
    if status == "skipped" or event.get("skipped") is True:
        return "skipped"
    return str(event.get("status") or "completed")


def _otel_status(status: str) -> Status:
    return Status(StatusCode.ERROR if status == "error" else StatusCode.OK)


def _node_kind(node: str) -> str:
    if node in {"router", "planner", "evidence_evaluator", "replanner"}:
        return OpenInferenceSpanKindValues.AGENT.value
    if node in {"retrieval_worker", "memory_worker", "timeline_worker", "web_worker"}:
        return OpenInferenceSpanKindValues.RETRIEVER.value
    return OpenInferenceSpanKindValues.CHAIN.value


def _node_display_name(node: str) -> str:
    labels = {
        "context_loader": "Context Loader",
        "router": "Router",
        "planner": "Planner",
        "evidence_evaluator": "Evidence Evaluator",
        "replanner": "Replanner",
        "retrieval_worker": "Document Retrieval",
        "memory_worker": "Memory Retrieval",
        "timeline_worker": "Timeline Retrieval",
        "web_worker": "Web Retrieval",
        "web_approval_gate": "Web Approval",
        "direct_answer": "Direct Answer",
        "synthesizer": "Synthesizer",
        "finalizer": "Finalizer",
        "hitl_gate": "HITL Gate",
    }
    return labels.get(node, node.replace("_", " ").title())


def _tool_kind(event: Mapping[str, Any]) -> str:
    return OpenInferenceSpanKindValues.TOOL.value


def enrich_tool_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    tool_name = event.get("tool_name")
    contract = get_tool_contract_metadata(tool_name) if isinstance(tool_name, str) else {}
    if not contract:
        return dict(event)
    return {
        **event,
        "tool_id": contract.get("id"),
        "tool_category": contract.get("category"),
        "tool_display_name": contract.get("display_name"),
        "artifact_keys": contract.get("artifact_keys", []),
        "known_warning_codes": contract.get("warning_codes", []),
    }


def _exception_attributes(error: Any) -> Dict[str, Any]:
    if not error:
        return {}
    if isinstance(error, dict):
        return _clean_dict(
            {
                "exception.type": error.get("type") or error.get("code"),
                "exception.message": error.get("message") or error.get("raw_message"),
                "askpdf.error.code": error.get("code"),
                "askpdf.error.retryable": error.get("retryable"),
            }
        )
    return {"exception.message": str(error)}


def _exception_event(error: Any) -> Optional[Dict[str, Any]]:
    attributes = _exception_attributes(error)
    return {"name": "exception", "attributes": attributes} if attributes else None


def _warning_events(warnings: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for warning in _as_list(warnings):
        if warning:
            result.append({"name": "warning", "attributes": {"warning.code": str(warning)}})
    return result


def _prompt_event(summary: Any) -> Optional[Dict[str, Any]]:
    data = _as_dict(summary)
    if not data:
        return None
    return {
        "name": "prompt.rendered",
        "attributes": _clean_dict(
            {
                "prompt.name": data.get("section"),
                "prompt.chars": data.get("prompt_chars"),
            }
        ),
        "output": _clean_dict(
            {
                "system_message": data.get("system_message"),
                "preview": data.get("preview"),
            }
        ),
    }


def _llm_completed_event(summary: Any) -> Optional[Dict[str, Any]]:
    data = _as_dict(summary)
    llm = _as_dict(data.get("llm"))
    if not llm:
        return None
    token_counts = _as_dict(llm.get("token_counts"))
    return {
        "name": "llm.completed",
        "attributes": _clean_dict(
            {
                SpanAttributes.LLM_MODEL_NAME: llm.get("model_name"),
                "llm.response_chars": llm.get("response_chars"),
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: token_counts.get("prompt"),
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: token_counts.get("completion"),
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: token_counts.get("total"),
                "llm.token_count.reasoning": token_counts.get("reasoning"),
                "llm.token_count.cached": token_counts.get("cached"),
                "llm.reasoning_available": llm.get("reasoning_available"),
                "llm.reasoning_format": llm.get("reasoning_format"),
                "llm.reasoning_chars": llm.get("reasoning_chars"),
            }
        ),
        "output": _clean_dict(
            {
                "reasoning_preview": _bounded_value(llm.get("reasoning_preview")),
            }
        ),
    }


def _llm_retry_events(summary: Any) -> List[Dict[str, Any]]:
    data = _as_dict(summary)
    llm = _as_dict(data.get("llm"))
    result: List[Dict[str, Any]] = []
    for attempt in _as_list(llm.get("retry_attempts")):
        if not isinstance(attempt, dict):
            continue
        result.append(
            {
                "name": "llm.retry",
                "attributes": _clean_dict(
                    {
                        "llm.retry.attempt": attempt.get("attempt"),
                        "llm.retry.delay_ms": attempt.get("delay_ms"),
                        "llm.retry.reason": attempt.get("reason"),
                        "http.status_code": attempt.get("http_status_code"),
                        "exception.type": attempt.get("exception_type"),
                        "exception.message": attempt.get("exception_message"),
                    }
                ),
            }
        )
    return result


def _decision_events(event: Mapping[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    if event.get("route") or event.get("route_reason") or event.get("execution_plan"):
        result.append(
            {
                "name": "decision.made",
                "attributes": _clean_dict(
                    {
                        "askpdf.route": event.get("route"),
                        "askpdf.route_reason": event.get("route_reason"),
                        "askpdf.execution_plan": event.get("execution_plan"),
                    }
                ),
            }
        )
    llm_result = _as_dict(event.get("llm_result_summary"))
    for note in _as_list(llm_result.get("normalization_notes")):
        result.append(
            {
                "name": "normalization.applied",
                "attributes": {"askpdf.normalization.note": str(note)},
            }
        )
    if event.get("evaluator_report"):
        result.append(
            {
                "name": "evaluation.completed",
                "attributes": _clean_dict(
                    {
                        "askpdf.evaluator_route": event.get("evaluator_route"),
                        "askpdf.evaluation_confidence": event.get("evaluation_confidence"),
                        "askpdf.replan_count": event.get("replan_count"),
                        "askpdf.replans": event.get("replans"),
                    }
                ),
                "output": _clean_dict(
                    {
                        "evaluator_report": _bounded_value(event.get("evaluator_report")),
                        "evidence_gaps": _bounded_value(event.get("evidence_gaps")),
                    }
                ),
            }
        )
    event_name = event.get("event_name")
    if isinstance(event_name, str) and event_name in {
        "evaluation.completed",
        "replan.requested",
        "replan.skipped",
        "replan.budget_exhausted",
    } and not (event_name == "evaluation.completed" and event.get("evaluator_report")):
        result.append(
            {
                "name": event_name,
                "attributes": _clean_dict(
                    {
                        "askpdf.evaluator_route": event.get("evaluator_route"),
                        "askpdf.evaluation_confidence": event.get("evaluation_confidence"),
                        "askpdf.replan_count": event.get("replan_count"),
                        "askpdf.replans": event.get("replans"),
                        "askpdf.replan_reason": event.get("replan_reason"),
                    }
                ),
                "output": _clean_dict(
                    {
                        "evaluator_report": _bounded_value(event.get("evaluator_report")),
                        "evidence_gaps": _bounded_value(event.get("evidence_gaps")),
                        "execution_plan": _bounded_value(event.get("execution_plan")),
                    }
                ),
            }
        )
    return result


def _span_links_from_refs(refs: Any) -> List[Dict[str, Any]]:
    data = _as_dict(refs)
    links: List[Dict[str, Any]] = []
    for key, value in data.items():
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    links.append({"type": key, "ref": item, "index": index})
        elif isinstance(value, dict):
            links.append({"type": key, "ref": value})
    return links


def _artifacts_from_refs(refs: Any, *, span_id: str) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    for index, link in enumerate(_span_links_from_refs(refs)):
        artifacts.append(
            {
                "artifact_id": f"{span_id}:{link['type']}:{index}",
                "span_id": span_id,
                "type": link["type"],
                "ref": link["ref"],
            }
        )
    return artifacts


def _first_number(*values: Any) -> int:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _llm_usage_metrics(spans: List[Dict[str, Any]]) -> Dict[str, int]:
    metrics = {
        "llm_span_count": 0,
        "llm_token_count_prompt": 0,
        "llm_token_count_completion": 0,
        "llm_token_count_total": 0,
        "llm_token_count_reasoning": 0,
        "llm_token_count_cached": 0,
        "llm_retry_count": 0,
    }
    for span in spans:
        if span.get("kind") != OpenInferenceSpanKindValues.LLM.value:
            continue
        attributes = _as_dict(span.get("attributes"))
        metrics["llm_span_count"] += 1
        metrics["llm_retry_count"] += _first_number(attributes.get("llm.retry_count"))
        metrics["llm_token_count_prompt"] += _first_number(attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT))
        metrics["llm_token_count_completion"] += _first_number(attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION))
        metrics["llm_token_count_total"] += _first_number(attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL))
        metrics["llm_token_count_reasoning"] += _first_number(attributes.get("llm.token_count.reasoning"))
        metrics["llm_token_count_cached"] += _first_number(attributes.get("llm.token_count.cached"))
    return {key: value for key, value in metrics.items() if value}


def _as_string_list(value: Any) -> List[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


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


def _root_event_key(event: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    if _is_interrupt_event(event):
        return (*_interrupt_event_key(event), None)
    attrs = _as_dict(event.get("attributes"))
    output = _as_dict(event.get("output"))
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
    """Append a HITL event to an already-stored debug payload, preserving v1 shape."""

    if not isinstance(debug_payload, dict) or debug_payload.get("version") != DEBUG_PAYLOAD_VERSION:
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
    """Append a generic root lifecycle event to a stored v1 debug payload."""

    if not isinstance(debug_payload, dict) or debug_payload.get("version") != DEBUG_PAYLOAD_VERSION:
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

    if not isinstance(base_payload, dict) or base_payload.get("version") != DEBUG_PAYLOAD_VERSION:
        return base_payload
    if not isinstance(incoming_payload, dict) or incoming_payload.get("version") != DEBUG_PAYLOAD_VERSION:
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

    existing_ids = {str(span.get("span_id")) for span in base_spans if isinstance(span, dict)}
    id_map: Dict[str, str] = {}
    appended: List[Dict[str, Any]] = []
    for span in incoming_spans:
        if not isinstance(span, dict):
            continue
        span_id = str(span.get("span_id") or "")
        if not span_id or span is incoming_root:
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
    summary = _build_summary_from_trace(base_trace, resolved_spec)
    return {**base_payload, "trace": base_trace, "summary": summary}


class AgentTraceRecorder:
    """Small local OpenTelemetry wrapper for one agent run."""

    def __init__(self, run: Any):
        self.run = run
        self.resolved_spec = _as_dict(getattr(run, "resolved_spec_json", None))
        self.run_span_id = f"run:{run.id}"
        self._provider = TracerProvider()
        self._exporter = _BufferedSpanExporter()
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        self._tracer = self._provider.get_tracer("askpdf.agent_patterns")
        self._spans_by_id: Dict[str, Any] = {}
        self._sidecars: Dict[str, Dict[str, Any]] = {}
        self._node_span_by_node: Dict[str, str] = {}
        self._node_index = 0
        self._tool_index = 0
        self._finalized = False
        self._root_span = self._start_span(
            span_id=self.run_span_id,
            parent_span_id=None,
            name="Agent Run",
            kind=OpenInferenceSpanKindValues.AGENT.value,
            status=str(getattr(run, "status", None) or "running"),
            start_time=getattr(run, "started_at", None),
            attributes=self._run_base_attributes(chat_turn_id=None, metrics={}, route=None, route_reason=None),
            input_data={},
            output_data={},
            events=[],
            links=[],
            raw={},
            order=0,
            end_immediately=False,
        )

    def _run_base_attributes(
        self,
        *,
        chat_turn_id: Optional[str],
        metrics: Mapping[str, Any],
        route: Any,
        route_reason: Any,
    ) -> Dict[str, Any]:
        config = _as_dict(self.resolved_spec.get("config"))
        return _clean_dict(
            {
                SpanAttributes.SESSION_ID: getattr(self.run, "thread_id", None),
                SpanAttributes.USER_ID: getattr(self.run, "user_id", None),
                SpanAttributes.AGENT_NAME: getattr(self.run, "template_id", None),
                "askpdf.run.id": getattr(self.run, "id", None),
                "askpdf.thread.id": getattr(self.run, "thread_id", None),
                "askpdf.chat_turn.id": chat_turn_id,
                "askpdf.template.id": getattr(self.run, "template_id", None),
                "askpdf.template_version.id": getattr(self.run, "template_version_id", None),
                "askpdf.pattern_type": self.resolved_spec.get("pattern_type"),
                "askpdf.route": route or metrics.get("route"),
                "askpdf.route_reason": route_reason,
                "askpdf.use_web_search": config.get("use_web_search"),
                "askpdf.use_reranker": config.get("use_reranker"),
                "askpdf.context_window": config.get("context_window"),
                "askpdf.warning_count": metrics.get("tool_warning_count"),
                "askpdf.error_count": metrics.get("error_count"),
            }
        )

    def _start_span(
        self,
        *,
        span_id: str,
        parent_span_id: Optional[str],
        name: str,
        kind: str,
        status: str,
        start_time: Any,
        attributes: Dict[str, Any],
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        events: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
        raw: Dict[str, Any],
        order: float,
        end_time: Any = None,
        end_immediately: bool = True,
    ) -> Any:
        parent = self._spans_by_id.get(parent_span_id or "")
        context = set_span_in_context(parent) if parent is not None else None
        span = self._tracer.start_span(
            name,
            context=context,
            kind=SpanKind.INTERNAL,
            start_time=_parse_time_ns(start_time),
        )
        output_attributes = _clean_dict(
            {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: kind,
                "askpdf.span_id": span_id,
                "askpdf.parent_span_id": parent_span_id,
                "askpdf.status": status,
                "askpdf.order": order,
                **attributes,
            }
        )
        _set_attributes(span, output_attributes)
        span.set_status(_otel_status(status))
        for event in events:
            attrs = _as_dict(event.get("attributes"))
            span.add_event(event.get("name") or "event", attributes={k: v for k, v in ((_key, _otel_attr_value(_value)) for _key, _value in attrs.items()) if v is not None})
        self._spans_by_id[span_id] = span
        self._sidecars[span_id] = {
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "kind": kind,
            "status": status,
            "attributes": output_attributes,
            "input": input_data,
            "output": output_data,
            "events": events,
            "links": links,
            "raw": raw,
            "order": order,
        }
        if end_immediately:
            span.end(end_time=_parse_time_ns(end_time))
        return span

    def record_node_event(self, event: Mapping[str, Any]) -> None:
        index = self._node_index
        self._node_index += 1
        node = str(event.get("node") or event.get("name") or f"node_{index}")
        node_type = str(event.get("node_type") or node)
        status = _span_status(event)
        span_id = f"node:{node}:{index}"
        llm_summary = _as_dict(_as_dict(event.get("llm_result_summary")).get("llm"))
        token_counts = _as_dict(llm_summary.get("token_counts"))
        prompt = _prompt_event(event.get("prompt_summary"))
        llm_completed = _llm_completed_event(event.get("llm_result_summary"))
        exception = _exception_event(event.get("error"))
        events = [
            *_decision_events(event),
            *([prompt] if prompt else []),
            *_llm_retry_events(event.get("llm_result_summary")),
            *([llm_completed] if llm_completed else []),
            *_warning_events(event.get("warnings")),
            *([exception] if exception else []),
        ]
        if status == "skipped":
            events.append(
                {
                    "name": "skipped",
                    "attributes": _clean_dict({"askpdf.skip_reason": event.get("skip_reason")}),
                }
            )
        attributes = _clean_dict(
            {
                SpanAttributes.GRAPH_NODE_ID: node,
                SpanAttributes.GRAPH_NODE_NAME: _node_display_name(node),
                "askpdf.node.id": node,
                "askpdf.node.type": node_type,
                "askpdf.node.name": _node_display_name(node),
                "askpdf.route": event.get("route"),
                "askpdf.route_reason": event.get("route_reason"),
                "askpdf.evaluator_route": event.get("evaluator_route"),
                "askpdf.evaluation_confidence": event.get("evaluation_confidence"),
                "askpdf.replan_count": event.get("replan_count"),
                "askpdf.replans": event.get("replans"),
                "askpdf.skip_reason": event.get("skip_reason"),
                "askpdf.execution_plan": event.get("execution_plan"),
                "askpdf.evidence_chars": event.get("evidence_chars"),
                "askpdf.answer_chars": event.get("answer_chars"),
                "askpdf.document_source_count": event.get("document_source_count"),
                "askpdf.web_source_count": event.get("web_source_count"),
                "askpdf.used_chat_id_count": event.get("used_chat_id_count"),
                "askpdf.timeline_event_count": event.get("timeline_event_count"),
                SpanAttributes.LLM_MODEL_NAME: llm_summary.get("model_name"),
                "llm.response_chars": llm_summary.get("response_chars"),
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: token_counts.get("prompt"),
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: token_counts.get("completion"),
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: token_counts.get("total"),
                "llm.token_count.reasoning": token_counts.get("reasoning"),
                "llm.token_count.cached": token_counts.get("cached"),
                "llm.retry_count": llm_summary.get("retry_count"),
            }
        )
        input_data = _clean_dict(
            {
                "value": _bounded_value(event.get("input_preview")),
                "refs": _bounded_value(event.get("input_refs")),
                "mime_type": "application/json",
            }
        )
        output_data = _clean_dict(
            {
                "value": _bounded_value(event.get("output_preview")),
                "refs": _bounded_value(event.get("output_refs")),
                "mime_type": "application/json",
            }
        )
        links = _span_links_from_refs(event.get("output_refs"))
        self._start_span(
            span_id=span_id,
            parent_span_id=self.run_span_id,
            name=_node_display_name(node),
            kind=_node_kind(node),
            status=status,
            start_time=event.get("start_time"),
            end_time=event.get("end_time"),
            attributes=_clean_dict(_bounded_value(attributes)),
            input_data=input_data,
            output_data=output_data,
            events=events,
            links=links,
            raw=_bounded_value(dict(event)),
            order=100 + index,
        )
        if node not in self._node_span_by_node:
            self._node_span_by_node[node] = span_id
        if llm_summary:
            self._record_llm_span(node=node, parent_span_id=span_id, index=index, event=event, llm_summary=llm_summary)

    def _record_llm_span(
        self,
        *,
        node: str,
        parent_span_id: str,
        index: int,
        event: Mapping[str, Any],
        llm_summary: Mapping[str, Any],
    ) -> None:
        token_counts = _as_dict(llm_summary.get("token_counts"))
        status = _span_status(event)
        span_id = f"llm:{node}:{index}"
        events = [
            *_llm_retry_events(event.get("llm_result_summary")),
            *([_llm_completed_event(event.get("llm_result_summary"))] if _llm_completed_event(event.get("llm_result_summary")) else []),
            *([_exception_event(event.get("error"))] if _exception_event(event.get("error")) else []),
        ]
        attributes = _clean_dict(
            {
                SpanAttributes.LLM_MODEL_NAME: llm_summary.get("model_name"),
                "llm.response_chars": llm_summary.get("response_chars"),
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: token_counts.get("prompt"),
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: token_counts.get("completion"),
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: token_counts.get("total"),
                "llm.token_count.reasoning": token_counts.get("reasoning"),
                "llm.token_count.cached": token_counts.get("cached"),
                "llm.retry_count": llm_summary.get("retry_count"),
                "llm.reasoning_available": llm_summary.get("reasoning_available"),
                "llm.reasoning_format": llm_summary.get("reasoning_format"),
                "llm.reasoning_chars": llm_summary.get("reasoning_chars"),
                "askpdf.node.id": node,
            }
        )
        self._start_span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=f"{_node_display_name(node)} LLM",
            kind=OpenInferenceSpanKindValues.LLM.value,
            status=status,
            start_time=event.get("start_time"),
            end_time=event.get("end_time"),
            attributes=_clean_dict(_bounded_value(attributes)),
            input_data=_clean_dict({"value": _bounded_value(event.get("prompt_summary")), "mime_type": "application/json"}),
            output_data=_clean_dict({"value": _bounded_value(llm_summary), "mime_type": "application/json"}),
            events=events,
            links=[],
            raw=_bounded_value({"node": node, "llm": dict(llm_summary)}),
            order=100 + index + 0.1,
        )

    def record_tool_event(self, event: Mapping[str, Any]) -> None:
        index = self._tool_index
        self._tool_index += 1
        enriched = enrich_tool_event(event)
        tool_name = str(enriched.get("tool_name") or f"tool_{index}")
        caller_node = enriched.get("caller_node")
        caller_node_type = enriched.get("caller_node_type")
        parent_span_id = self._node_span_by_node.get(str(caller_node)) or self.run_span_id
        status = "error" if enriched.get("ok") is False or enriched.get("error") else "completed"
        span_id = f"tool:{tool_name}:{index}"
        exception = _exception_event(enriched.get("error"))
        events = [
            {
                "name": "tool.called",
                "attributes": _clean_dict(
                    {
                        SpanAttributes.TOOL_NAME: tool_name,
                        SpanAttributes.TOOL_ID: enriched.get("tool_id"),
                        "askpdf.tool.category": enriched.get("tool_category"),
                    }
                ),
            },
            {
                "name": "tool.completed",
                "attributes": _clean_dict(
                    {
                        SpanAttributes.TOOL_NAME: tool_name,
                        "askpdf.result_chars": enriched.get("result_chars"),
                        "askpdf.source_count": enriched.get("source_count"),
                        "askpdf.warning_count": len(_as_list(enriched.get("warnings"))),
                    }
                ),
            },
            *_warning_events(enriched.get("warnings")),
            *([exception] if exception else []),
        ]
        attributes = _clean_dict(
            {
                SpanAttributes.TOOL_NAME: tool_name,
                SpanAttributes.TOOL_ID: enriched.get("tool_id"),
                SpanAttributes.TOOL_DESCRIPTION: enriched.get("tool_display_name"),
                "askpdf.tool.category": enriched.get("tool_category"),
                "askpdf.caller_node": caller_node,
                "askpdf.caller_node_type": caller_node_type,
                "askpdf.result_chars": enriched.get("result_chars"),
                "askpdf.source_count": enriched.get("source_count"),
                "askpdf.artifact_keys": enriched.get("artifact_keys"),
                "askpdf.known_warning_codes": enriched.get("known_warning_codes"),
            }
        )
        output_data = _clean_dict(
            {
                "value": _bounded_value(enriched.get("result_preview")),
                "refs": _bounded_value(enriched.get("artifact_refs")),
                "summary": _bounded_value(enriched.get("artifact_summary")),
                "mime_type": "application/json",
            }
        )
        self._start_span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=str(enriched.get("tool_display_name") or tool_name),
            kind=_tool_kind(enriched),
            status=status,
            start_time=enriched.get("start_time"),
            end_time=enriched.get("end_time"),
            attributes=_clean_dict(_bounded_value(attributes)),
            input_data=_clean_dict({"value": _bounded_value(enriched.get("tool_input")), "mime_type": "application/json"}),
            output_data=output_data,
            events=events,
            links=_span_links_from_refs(enriched.get("artifact_refs")),
            raw=_bounded_value(enriched),
            order=1000 + index,
        )

    def record_interrupt_event(self, interrupt: Mapping[str, Any], *, event_name: Optional[str] = None) -> None:
        event = build_interrupt_trace_event(interrupt, event_name=event_name)
        root_sidecar = self._sidecars.get(self.run_span_id)
        if root_sidecar is None:
            return
        existing = _as_list(root_sidecar.get("events"))
        if not any(_root_event_key(item) == _root_event_key(event) for item in existing if isinstance(item, dict)):
            root_sidecar["events"] = [*existing, event]
            if not self._finalized:
                attrs = _as_dict(event.get("attributes"))
                self._root_span.add_event(
                    event.get("name") or "interrupt",
                    attributes={k: v for k, v in ((_key, _otel_attr_value(_value)) for _key, _value in attrs.items()) if v is not None},
                )

    def record_runtime_event(
        self,
        event_name: str,
        *,
        attributes: Optional[Mapping[str, Any]] = None,
        input_data: Any = None,
        output_data: Any = None,
    ) -> None:
        event = build_runtime_trace_event(
            event_name,
            attributes=attributes,
            input_data=input_data,
            output_data=output_data,
        )
        root_sidecar = self._sidecars.get(self.run_span_id)
        if root_sidecar is None:
            return
        existing = _as_list(root_sidecar.get("events"))
        if not any(_root_event_key(item) == _root_event_key(event) for item in existing if isinstance(item, dict)):
            root_sidecar["events"] = [*existing, event]
            if not self._finalized:
                attrs = _as_dict(event.get("attributes"))
                self._root_span.add_event(
                    event.get("name") or "runtime.event",
                    attributes={k: v for k, v in ((_key, _otel_attr_value(_value)) for _key, _value in attrs.items()) if v is not None},
                )

    def finalize(
        self,
        *,
        run: Any,
        chat_turn_id: Optional[str],
        metrics: Dict[str, Any],
        route: Any = None,
        route_reason: Any = None,
        error: Any = None,
    ) -> Dict[str, Any]:
        if not self._finalized:
            self.run = run
            run_attributes = self._run_base_attributes(
                chat_turn_id=chat_turn_id,
                metrics=metrics,
                route=route,
                route_reason=route_reason,
            )
            _set_attributes(self._root_span, run_attributes)
            root_sidecar = self._sidecars[self.run_span_id]
            root_sidecar["attributes"] = _clean_dict({**root_sidecar["attributes"], **run_attributes})
            root_sidecar["status"] = str(getattr(run, "status", None) or "completed")
            root_sidecar["attributes"]["askpdf.status"] = root_sidecar["status"]
            if error:
                exception = _exception_event(error)
                if exception:
                    self._root_span.add_event("exception", attributes=_exception_attributes(error))
                    root_sidecar["events"].append(exception)
                self._root_span.set_status(_otel_status("error"))
            else:
                self._root_span.set_status(_otel_status(root_sidecar["status"]))
            self._root_span.end(end_time=_parse_time_ns(getattr(run, "completed_at", None)))
            self._finalized = True
        trace = self._build_trace(run=run, chat_turn_id=chat_turn_id, metrics=metrics)
        summary = self._build_summary(trace)
        return {
            "version": DEBUG_PAYLOAD_VERSION,
            "trace": trace,
            "summary": summary,
        }

    def _span_to_dict(self, span: ReadableSpan) -> Dict[str, Any]:
        attrs = dict(span.attributes or {})
        custom_span_id = str(attrs.get("askpdf.span_id") or f"{span.context.span_id:016x}")
        sidecar = self._sidecars.get(custom_span_id, {})
        attributes = dict(sidecar.get("attributes") or attrs)
        for key in ("askpdf.span_id", "askpdf.parent_span_id", "askpdf.order"):
            attributes.pop(key, None)
        return {
            "span_id": custom_span_id,
            "parent_span_id": sidecar.get("parent_span_id"),
            "name": span.name,
            "kind": sidecar.get("kind") or attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) or OpenInferenceSpanKindValues.UNKNOWN.value,
            "status": sidecar.get("status") or attrs.get("askpdf.status") or ("error" if span.status.status_code == StatusCode.ERROR else "completed"),
            "start_time": _ns_to_iso(span.start_time),
            "end_time": _ns_to_iso(span.end_time),
            "duration_ms": round((span.end_time - span.start_time) / 1_000_000, 2) if span.start_time and span.end_time else None,
            "attributes": attributes,
            "input": sidecar.get("input") or {},
            "output": sidecar.get("output") or {},
            "events": sidecar.get("events") or [],
            "links": sidecar.get("links") or [],
            "raw": sidecar.get("raw") or {},
        }

    def _build_trace(self, *, run: Any, chat_turn_id: Optional[str], metrics: Dict[str, Any]) -> Dict[str, Any]:
        span_pairs = []
        for span in self._exporter.spans:
            custom_id = str((span.attributes or {}).get("askpdf.span_id") or f"{span.context.span_id:016x}")
            order = self._sidecars.get(custom_id, {}).get("order", 9999)
            span_pairs.append((order, self._span_to_dict(span)))
        spans = [span for _, span in sorted(span_pairs, key=lambda item: item[0])]
        links: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []
        for span in spans:
            for link in _as_list(span.get("links")):
                if isinstance(link, dict):
                    links.append({"span_id": span.get("span_id"), **link})
            artifacts.extend(_artifacts_from_refs(_as_dict(span.get("output")).get("refs"), span_id=span["span_id"]))
        trace_metrics = {**metrics, **_llm_usage_metrics(spans)}
        root_attributes = spans[0]["attributes"] if spans else {}
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_id": str(getattr(run, "id", "")),
            "run_id": getattr(run, "id", None),
            "thread_id": getattr(run, "thread_id", None),
            "chat_turn_id": chat_turn_id,
            "user_id": getattr(run, "user_id", None),
            "template_id": getattr(run, "template_id", None),
            "template_version_id": getattr(run, "template_version_id", None),
            "pattern_type": self.resolved_spec.get("pattern_type"),
            "status": getattr(run, "status", None),
            "started_at": iso_utc_z(run.started_at) if getattr(run, "started_at", None) else None,
            "completed_at": iso_utc_z(run.completed_at) if getattr(run, "completed_at", None) else None,
            "duration_ms": trace_metrics.get("duration_ms"),
            "attributes": root_attributes,
            "metrics": trace_metrics,
            "spans": spans,
            "links": links,
            "artifacts": artifacts,
        }

    def _build_summary(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        return _build_summary_from_trace(trace, self.resolved_spec)


def _summary_node(span: Mapping[str, Any]) -> Dict[str, Any]:
    attributes = _as_dict(span.get("attributes"))
    raw = _as_dict(span.get("raw"))
    warnings = [
        _as_dict(event.get("attributes")).get("warning.code")
        for event in _as_list(span.get("events"))
        if isinstance(event, dict) and event.get("name") == "warning"
    ]
    error = next(
        (
            _as_dict(event.get("attributes"))
            for event in _as_list(span.get("events"))
            if isinstance(event, dict) and event.get("name") == "exception"
        ),
        None,
    )
    node_id = str(attributes.get("askpdf.node.id") or attributes.get(SpanAttributes.GRAPH_NODE_ID) or span.get("name") or "unknown_node")
    return {
        "id": node_id,
        "type": attributes.get("askpdf.node.type"),
        "status": span.get("status"),
        "skipped": span.get("status") == "skipped",
        "durationMs": span.get("duration_ms"),
        "route": attributes.get("askpdf.route"),
        "routeReason": attributes.get("askpdf.route_reason"),
        "evaluatorRoute": attributes.get("askpdf.evaluator_route"),
        "evaluationConfidence": attributes.get("askpdf.evaluation_confidence"),
        "replanCount": attributes.get("askpdf.replan_count"),
        "executionPlan": _as_string_list(attributes.get("askpdf.execution_plan")),
        "warningCodes": [str(warning) for warning in warnings if warning],
        "error": error,
        "span": dict(span),
        "raw": raw,
    }


def _summary_tool(span: Mapping[str, Any]) -> Dict[str, Any]:
    attributes = _as_dict(span.get("attributes"))
    raw = _as_dict(span.get("raw"))
    warning_events = [
        _as_dict(event.get("attributes")).get("warning.code")
        for event in _as_list(span.get("events"))
        if isinstance(event, dict) and event.get("name") == "warning"
    ]
    return {
        "name": str(attributes.get(SpanAttributes.TOOL_NAME) or span.get("name") or "tool"),
        "id": attributes.get(SpanAttributes.TOOL_ID),
        "category": attributes.get("askpdf.tool.category"),
        "displayName": attributes.get(SpanAttributes.TOOL_DESCRIPTION) or span.get("name"),
        "callerNode": attributes.get("askpdf.caller_node"),
        "callerNodeType": attributes.get("askpdf.caller_node_type"),
        "ok": span.get("status") != "error",
        "durationMs": span.get("duration_ms"),
        "sourceCount": attributes.get("askpdf.source_count"),
        "warningCodes": [str(warning) for warning in warning_events if warning],
        "span": dict(span),
        "raw": raw,
    }


def _build_summary_from_trace(trace: Dict[str, Any], resolved_spec: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = _as_dict(trace.get("metrics"))
    spans = _as_list(trace.get("spans"))
    node_spans = [
        span
        for span in spans
        if _as_dict(_as_dict(span).get("attributes")).get("askpdf.node.id")
        and _as_dict(span).get("kind") != OpenInferenceSpanKindValues.LLM.value
    ]
    tool_spans = [span for span in spans if _as_dict(span).get("kind") == OpenInferenceSpanKindValues.TOOL.value]
    errors = [
        _as_dict(event.get("attributes"))
        for span in spans
        for event in _as_list(_as_dict(span).get("events"))
        if isinstance(event, dict) and event.get("name") == "exception"
    ]
    nodes = [_summary_node(span) for span in node_spans if isinstance(span, dict)]
    tools = [_summary_tool(span) for span in tool_spans if isinstance(span, dict)]
    used_node_count = len({node.get("id") for node in nodes if node.get("id") and not node.get("skipped")})
    warning_count = sum(len(node.get("warningCodes") or []) for node in nodes) + sum(len(tool.get("warningCodes") or []) for tool in tools)
    error_count = int(metrics.get("error_count") or 0)
    if errors:
        error_count = max(error_count, len(errors))
    config = _as_dict(resolved_spec.get("config"))
    evaluator_nodes = [node for node in nodes if node.get("id") == "evidence_evaluator"]
    replanner_nodes = [node for node in nodes if node.get("id") == "replanner"]
    last_evaluator = evaluator_nodes[-1] if evaluator_nodes else {}
    last_evaluator_raw = _as_dict(last_evaluator.get("raw"))
    replan_count = max(
        [_first_number(node.get("replanCount"), _as_dict(node.get("raw")).get("replan_count")) for node in replanner_nodes + evaluator_nodes]
        or [0]
    )
    return {
        "status": trace.get("status"),
        "route": _as_dict(trace.get("attributes")).get("askpdf.route") or metrics.get("route"),
        "routeReason": _as_dict(trace.get("attributes")).get("askpdf.route_reason"),
        "evaluatorRoute": last_evaluator.get("evaluatorRoute") or last_evaluator_raw.get("evaluator_route"),
        "evaluationConfidence": last_evaluator.get("evaluationConfidence") or last_evaluator_raw.get("evaluation_confidence"),
        "evaluatorReport": _bounded_value(last_evaluator_raw.get("evaluator_report")),
        "evidenceGaps": _as_string_list(last_evaluator_raw.get("evidence_gaps")),
        "replanCount": replan_count,
        "durationMs": trace.get("duration_ms"),
        "metrics": metrics,
        "nodes": nodes,
        "tools": tools,
        "usedNodeCount": used_node_count,
        "availableNodeCount": len(_as_list(_as_dict(config.get("graph")).get("nodes"))) or None,
        "usedToolCount": int(metrics.get("tool_event_count") or len(tools)),
        "availableToolCount": len(set(_as_list(config.get("allowed_tool_ids")))) or None,
        "warningCount": int(metrics.get("tool_warning_count") or warning_count),
        "errorCount": error_count,
        "errors": [error for error in errors if error],
        **_interrupt_summary(trace),
    }


def _execution_plan_from_summary(summary: Mapping[str, Any]) -> List[str]:
    for node in _as_list(summary.get("nodes")):
        plan = node.get("executionPlan")
        if isinstance(plan, list) and plan:
            return _as_string_list(plan)
        raw_plan = _as_dict(node.get("raw")).get("execution_plan")
        if isinstance(raw_plan, list) and raw_plan:
            return _as_string_list(raw_plan)
    return []


def _graph_node_status(
    node_id: str,
    summary_node: Mapping[str, Any],
    tool_summaries: List[Dict[str, Any]],
    execution_plan: List[str],
) -> str:
    if summary_node.get("error") or any(not tool.get("ok", True) for tool in tool_summaries):
        return "error"
    if summary_node.get("skipped"):
        return "skipped"
    if summary_node:
        return "active"
    if node_id in execution_plan:
        return "planned"
    return "inactive"


def _has_active_node(node_id: str, nodes_by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    status = _as_dict(nodes_by_id.get(node_id)).get("status")
    return status in {"active", "planned", "skipped", "error"}


def _graph_tool_summary(tool: Mapping[str, Any]) -> Dict[str, Any]:
    raw = _as_dict(tool.get("raw"))
    return {
        "toolName": tool.get("name"),
        "displayName": tool.get("displayName"),
        "callerNode": tool.get("callerNode"),
        "ok": tool.get("ok", True),
        "elapsedMs": tool.get("durationMs"),
        "sourceCount": tool.get("sourceCount"),
        "warnings": tool.get("warningCodes") or [],
        "artifactKeys": _as_list(raw.get("artifact_keys")),
        "toolInput": raw.get("tool_input"),
        "resultPreview": raw.get("result_preview"),
        "artifactRefs": raw.get("artifact_refs"),
        "artifactSummary": raw.get("artifact_summary"),
        "traceSpan": tool.get("span"),
        "raw": raw,
    }


def build_debug_graph(
    *,
    resolved_spec: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the API-only graph view model from canonical stored telemetry."""

    graph_spec = _as_dict(_as_dict(resolved_spec.get("config")).get("graph"))
    spec_nodes = [node for node in _as_list(graph_spec.get("nodes")) if isinstance(node.get("id"), str)]
    spec_edges = [edge for edge in _as_list(graph_spec.get("edges")) if isinstance(edge.get("from"), str)]
    summary_nodes = {node["id"]: node for node in _as_list(summary.get("nodes")) if isinstance(node.get("id"), str)}
    tools_by_node: Dict[str, List[Dict[str, Any]]] = {}
    for tool in _as_list(summary.get("tools")):
        caller = tool.get("callerNode")
        if isinstance(caller, str):
            tools_by_node.setdefault(caller, []).append(tool)

    execution_plan = _execution_plan_from_summary(summary)
    selected_route = summary.get("route")
    nodes = []
    for spec in spec_nodes:
        node_id = spec["id"]
        summary_node = summary_nodes.get(node_id, {})
        tool_summaries = tools_by_node.get(node_id, [])
        status = _graph_node_status(node_id, summary_node, tool_summaries, execution_plan)
        raw_events = [summary_node.get("raw")] if summary_node.get("raw") else []
        raw = _as_dict(summary_node.get("raw"))
        nodes.append(
            {
                "id": node_id,
                "type": str(spec.get("type") or node_id),
                "label": str(spec.get("label") or _node_display_name(node_id)),
                "description": spec.get("description"),
                "status": status,
                "elapsedMs": summary_node.get("durationMs"),
                "route": summary_node.get("route"),
                "routeReason": summary_node.get("routeReason"),
                "evaluatorRoute": summary_node.get("evaluatorRoute"),
                "evaluationConfidence": summary_node.get("evaluationConfidence"),
                "replanCount": summary_node.get("replanCount"),
                "skipped": bool(summary_node.get("skipped")),
                "skipReason": raw.get("skip_reason"),
                "executionPlan": execution_plan if node_id in {"planner", "replanner"} else None,
                "warnings": summary_node.get("warningCodes") or [],
                "inputRefs": raw.get("input_refs"),
                "outputRefs": raw.get("output_refs"),
                "inputPreview": raw.get("input_preview"),
                "outputPreview": raw.get("output_preview"),
                "promptSummary": raw.get("prompt_summary"),
                "llmResultSummary": raw.get("llm_result_summary"),
                "llmSummary": _as_dict(raw.get("llm_result_summary")).get("llm"),
                "toolSummaries": [_graph_tool_summary(tool) for tool in tool_summaries],
                "warningCount": len(summary_node.get("warningCodes") or []) + sum(len(tool.get("warningCodes") or []) for tool in tool_summaries),
                "errorCount": (1 if summary_node.get("error") else 0) + sum(1 for tool in tool_summaries if not tool.get("ok", True)),
                "sourceCount": sum(int(tool.get("sourceCount") or 0) for tool in tool_summaries),
                "artifactCount": sum(len(_as_list(_as_dict(tool.get("raw")).get("artifact_keys"))) for tool in tool_summaries),
                "traceSpans": [summary_node.get("span")] if summary_node.get("span") else [],
                "rawEvents": raw_events,
            }
        )

    nodes_by_id = {node["id"]: node for node in nodes}
    edges = []
    for index, edge in enumerate(spec_edges):
        source = edge.get("from")
        if source == "START" or edge.get("to") == "END":
            continue
        if edge.get("conditional") and isinstance(edge.get("routes"), dict):
            source_node = nodes_by_id.get(str(source), {})
            source_raw = _as_dict(_as_list(source_node.get("rawEvents"))[-1]) if _as_list(source_node.get("rawEvents")) else {}
            for route, target in edge["routes"].items():
                if not isinstance(target, str):
                    continue
                selected = selected_route == route or (
                    source == "evidence_evaluator"
                    and (source_node.get("evaluatorRoute") == route or source_raw.get("evaluator_route") == route)
                )
                edges.append(
                    {
                        "id": f"{source}-{route}-{target}",
                        "source": source,
                        "target": target,
                        "label": route,
                        "route": route,
                        "selected": selected,
                        "active": selected or (_has_active_node(source, nodes_by_id) and _has_active_node(target, nodes_by_id)),
                        "conditional": True,
                        "raw": edge,
                    }
                )
            continue
        target = edge.get("to")
        if not isinstance(target, str):
            continue
        edges.append(
            {
                "id": f"{source}-{target}-{index}",
                "source": source,
                "target": target,
                "selected": False,
                "active": _has_active_node(source, nodes_by_id) and _has_active_node(target, nodes_by_id),
                "conditional": False,
                "raw": edge,
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "executionPlan": execution_plan,
        "selectedRoute": selected_route,
    }


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
