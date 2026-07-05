from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from app.agent.tool_registry import get_tool_contract_metadata
from app.agent_patterns.trace import compact_preview
from app.time_utils import iso_utc_z


TRACE_SCHEMA_VERSION = 1
TRACE_PREVIEW_LIMIT = 900


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _clean_dict(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _bounded_value(value: Any) -> Any:
    if value in (None, "", [], {}):
        return value
    if isinstance(value, str):
        return compact_preview(value, limit=TRACE_PREVIEW_LIMIT)
    if isinstance(value, list):
        return [_bounded_value(item) for item in value[:50]]
    if isinstance(value, dict):
        return {
            key: _bounded_value(item)
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
    return value


def _span_status(event: Mapping[str, Any]) -> str:
    if event.get("error") or event.get("ok") is False:
        return "error"
    if event.get("status") == "skipped" or event.get("skipped") is True:
        return "skipped"
    return str(event.get("status") or "completed")


def _node_kind(node: str) -> str:
    if node in {"router", "planner"}:
        return "AGENT"
    if node in {"retrieval_worker", "memory_worker", "timeline_worker", "web_worker"}:
        return "RETRIEVER"
    return "CHAIN"


def _node_display_name(node: str) -> str:
    labels = {
        "context_loader": "Context Loader",
        "router": "Router",
        "planner": "Planner",
        "retrieval_worker": "Document Retrieval",
        "memory_worker": "Memory Retrieval",
        "timeline_worker": "Timeline Retrieval",
        "web_worker": "Web Retrieval",
        "direct_answer": "Direct Answer",
        "synthesizer": "Synthesizer",
        "finalizer": "Finalizer",
    }
    return labels.get(node, node.replace("_", " ").title())


def _tool_kind(event: Mapping[str, Any]) -> str:
    category = event.get("tool_category")
    if category in {"document", "memory", "timeline", "web"}:
        return "RETRIEVER"
    return "TOOL"


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


def _event_time(value: Any) -> Optional[str]:
    if not value:
        return None
    try:
        return iso_utc_z(value)
    except Exception:
        return None


def _warning_events(warnings: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for warning in _as_list(warnings):
        if not warning:
            continue
        result.append(
            {
                "name": "warning",
                "attributes": {"warning.code": str(warning)},
            }
        )
    return result


def _exception_event(error: Any) -> Optional[Dict[str, Any]]:
    if not error:
        return None
    if isinstance(error, dict):
        attributes = {
            "exception.type": error.get("type") or error.get("code"),
            "exception.message": error.get("message") or error.get("raw_message"),
            "askpdf.error.code": error.get("code"),
            "askpdf.error.retryable": error.get("retryable"),
        }
    else:
        attributes = {"exception.message": str(error)}
    return {"name": "exception", "attributes": _clean_dict(attributes)}


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
                "llm.model_name": llm.get("model_name"),
                "llm.response_chars": llm.get("response_chars"),
                "llm.token_count.prompt": token_counts.get("prompt"),
                "llm.token_count.completion": token_counts.get("completion"),
                "llm.token_count.total": token_counts.get("total"),
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


def _artifacts_from_refs(refs: Any) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    for link in _span_links_from_refs(refs):
        artifact = {
            "artifact_id": f"{link['type']}:{len(artifacts)}",
            "type": link["type"],
            "ref": link["ref"],
        }
        artifacts.append(artifact)
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
        attributes = _as_dict(span.get("attributes"))
        completed_events = [
            event
            for event in _as_list(span.get("events"))
            if event.get("name") == "llm.completed"
        ]
        retry_events = [
            event
            for event in _as_list(span.get("events"))
            if event.get("name") == "llm.retry"
        ]
        if not completed_events and not retry_events and "llm.model_name" not in attributes:
            continue
        metrics["llm_span_count"] += 1
        metrics["llm_retry_count"] += _first_number(attributes.get("llm.retry_count"), len(retry_events))
        for event in completed_events:
            event_attributes = _as_dict(event.get("attributes"))
            metrics["llm_token_count_prompt"] += _first_number(
                event_attributes.get("llm.token_count.prompt"),
                attributes.get("llm.token_count.prompt"),
            )
            metrics["llm_token_count_completion"] += _first_number(
                event_attributes.get("llm.token_count.completion"),
                attributes.get("llm.token_count.completion"),
            )
            metrics["llm_token_count_total"] += _first_number(
                event_attributes.get("llm.token_count.total"),
                attributes.get("llm.token_count.total"),
            )
            metrics["llm_token_count_reasoning"] += _first_number(
                event_attributes.get("llm.token_count.reasoning"),
                attributes.get("llm.token_count.reasoning"),
            )
            metrics["llm_token_count_cached"] += _first_number(
                event_attributes.get("llm.token_count.cached"),
                attributes.get("llm.token_count.cached"),
            )
    return {key: value for key, value in metrics.items() if value}


def _node_span(
    event: Mapping[str, Any],
    *,
    index: int,
    run_span_id: str,
) -> Dict[str, Any]:
    node = str(event.get("node") or event.get("name") or f"node_{index}")
    span_id = f"node:{node}:{index}"
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
    llm_summary = _as_dict(_as_dict(event.get("llm_result_summary")).get("llm"))
    token_counts = _as_dict(llm_summary.get("token_counts"))
    if _span_status(event) == "skipped":
        events.append(
            {
                "name": "skipped",
                "attributes": _clean_dict({"askpdf.skip_reason": event.get("skip_reason")}),
            }
        )

    return {
        "span_id": span_id,
        "parent_span_id": run_span_id,
        "name": _node_display_name(node),
        "kind": _node_kind(node),
        "status": _span_status(event),
        "start_time": _event_time(event.get("start_time")),
        "end_time": _event_time(event.get("end_time")),
        "duration_ms": event.get("elapsed_ms"),
        "attributes": _clean_dict(
            {
                "askpdf.node.id": node,
                "askpdf.node.name": _node_display_name(node),
                "askpdf.route": event.get("route"),
                "askpdf.route_reason": event.get("route_reason"),
                "askpdf.skip_reason": event.get("skip_reason"),
                "askpdf.execution_plan": event.get("execution_plan"),
                "askpdf.evidence_chars": event.get("evidence_chars"),
                "askpdf.answer_chars": event.get("answer_chars"),
                "askpdf.document_source_count": event.get("document_source_count"),
                "askpdf.web_source_count": event.get("web_source_count"),
                "askpdf.used_chat_id_count": event.get("used_chat_id_count"),
                "askpdf.timeline_event_count": event.get("timeline_event_count"),
                "llm.model_name": llm_summary.get("model_name"),
                "llm.response_chars": llm_summary.get("response_chars"),
                "llm.token_count.prompt": token_counts.get("prompt"),
                "llm.token_count.completion": token_counts.get("completion"),
                "llm.token_count.total": token_counts.get("total"),
                "llm.token_count.reasoning": token_counts.get("reasoning"),
                "llm.token_count.cached": token_counts.get("cached"),
                "llm.retry_count": llm_summary.get("retry_count"),
            }
        ),
        "input": _clean_dict(
            {
                "value": _bounded_value(event.get("input_preview")),
                "refs": _bounded_value(event.get("input_refs")),
                "mime_type": "application/json",
            }
        ),
        "output": _clean_dict(
            {
                "value": _bounded_value(event.get("output_preview")),
                "refs": _bounded_value(event.get("output_refs")),
                "mime_type": "application/json",
            }
        ),
        "events": events,
        "links": _span_links_from_refs(event.get("output_refs")),
        "raw": dict(event),
    }


def _tool_span(
    event: Mapping[str, Any],
    *,
    index: int,
    caller_span_by_node: Mapping[str, str],
    run_span_id: str,
) -> Dict[str, Any]:
    tool_name = str(event.get("tool_name") or f"tool_{index}")
    caller_node = event.get("caller_node")
    parent_span_id = caller_span_by_node.get(str(caller_node)) or run_span_id
    span_id = f"tool:{tool_name}:{index}"
    exception = _exception_event(event.get("error"))
    events = [
        {
            "name": "tool.called",
            "attributes": _clean_dict(
                {
                    "tool.name": tool_name,
                    "tool.id": event.get("tool_id"),
                    "askpdf.tool.category": event.get("tool_category"),
                }
            ),
        },
        {
            "name": "tool.completed",
            "attributes": _clean_dict(
                {
                    "tool.name": tool_name,
                    "askpdf.result_chars": event.get("result_chars"),
                    "askpdf.source_count": event.get("source_count"),
                    "askpdf.warning_count": len(_as_list(event.get("warnings"))),
                }
            ),
        },
        *_warning_events(event.get("warnings")),
        *([exception] if exception else []),
    ]
    return {
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": str(event.get("tool_display_name") or tool_name),
        "kind": _tool_kind(event),
        "status": "error" if event.get("ok") is False or event.get("error") else "completed",
        "start_time": _event_time(event.get("start_time")),
        "end_time": _event_time(event.get("end_time")),
        "duration_ms": event.get("elapsed_ms"),
        "attributes": _clean_dict(
            {
                "tool.name": tool_name,
                "tool.id": event.get("tool_id"),
                "tool.description": event.get("tool_display_name"),
                "askpdf.tool.category": event.get("tool_category"),
                "askpdf.caller_node": caller_node,
                "askpdf.result_chars": event.get("result_chars"),
                "askpdf.source_count": event.get("source_count"),
                "askpdf.artifact_keys": event.get("artifact_keys"),
                "askpdf.known_warning_codes": event.get("known_warning_codes"),
            }
        ),
        "input": _clean_dict(
            {
                "value": event.get("tool_input"),
                "mime_type": "application/json",
            }
        ),
        "output": _clean_dict(
            {
                "value": _bounded_value(event.get("result_preview")),
                "refs": _bounded_value(event.get("artifact_refs")),
                "summary": _bounded_value(event.get("artifact_summary")),
                "mime_type": "application/json",
            }
        ),
        "events": events,
        "links": _span_links_from_refs(event.get("artifact_refs")),
        "raw": dict(event),
    }


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
    resolved_spec = _as_dict(getattr(run, "resolved_spec_json", None))
    config = _as_dict(resolved_spec.get("config"))
    run_span_id = f"run:{run.id}"
    trace_id = str(run.id)
    started_at = iso_utc_z(run.started_at) if getattr(run, "started_at", None) else None
    completed_at = iso_utc_z(run.completed_at) if getattr(run, "completed_at", None) else None

    spans: List[Dict[str, Any]] = [
        {
            "span_id": run_span_id,
            "parent_span_id": None,
            "name": "Agent Run",
            "kind": "AGENT",
            "status": getattr(run, "status", None),
            "start_time": started_at,
            "end_time": completed_at,
            "duration_ms": metrics.get("duration_ms"),
            "attributes": _clean_dict(
                {
                    "session.id": getattr(run, "thread_id", None),
                    "user.id": getattr(run, "user_id", None),
                    "askpdf.run.id": getattr(run, "id", None),
                    "askpdf.thread.id": getattr(run, "thread_id", None),
                    "askpdf.chat_turn.id": getattr(chat_turn, "id", None),
                    "askpdf.template.id": getattr(run, "template_id", None),
                    "askpdf.template_version.id": getattr(run, "template_version_id", None),
                    "askpdf.pattern_type": resolved_spec.get("pattern_type"),
                    "askpdf.route": route or metrics.get("route"),
                    "askpdf.route_reason": route_reason,
                    "askpdf.use_web_search": config.get("use_web_search"),
                    "askpdf.use_reranker": config.get("use_reranker"),
                    "askpdf.context_window": config.get("context_window"),
                    "askpdf.warning_count": metrics.get("tool_warning_count"),
                    "askpdf.error_count": metrics.get("error_count"),
                }
            ),
            "input": {},
            "output": {},
            "events": [event for event in [_exception_event(error)] if event],
            "links": [],
            "raw": {},
        }
    ]

    caller_span_by_node: Dict[str, str] = {}
    for index, event in enumerate(node_events):
        span = _node_span(event, index=index, run_span_id=run_span_id)
        spans.append(span)
        node = span["attributes"].get("askpdf.node.id")
        if isinstance(node, str) and node not in caller_span_by_node:
            caller_span_by_node[node] = span["span_id"]

    enriched_tool_events = [
        enrich_tool_event(event)
        for event in tool_events
        if isinstance(event, dict)
    ]
    for index, event in enumerate(enriched_tool_events):
        spans.append(
            _tool_span(
                event,
                index=index,
                caller_span_by_node=caller_span_by_node,
                run_span_id=run_span_id,
            )
        )

    links: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    for span in spans:
        for link in _as_list(span.get("links")):
            if isinstance(link, dict):
                links.append({"span_id": span.get("span_id"), **link})
        artifacts.extend(_artifacts_from_refs(_as_dict(span.get("output")).get("refs")))
    trace_metrics = {**metrics, **_llm_usage_metrics(spans)}

    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": trace_id,
        "run_id": getattr(run, "id", None),
        "thread_id": getattr(run, "thread_id", None),
        "chat_turn_id": getattr(chat_turn, "id", None),
        "user_id": getattr(run, "user_id", None),
        "template_id": getattr(run, "template_id", None),
        "template_version_id": getattr(run, "template_version_id", None),
        "pattern_type": resolved_spec.get("pattern_type"),
        "status": getattr(run, "status", None),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": metrics.get("duration_ms"),
        "attributes": spans[0]["attributes"],
        "metrics": trace_metrics,
        "spans": spans,
        "links": links,
        "artifacts": artifacts,
    }
