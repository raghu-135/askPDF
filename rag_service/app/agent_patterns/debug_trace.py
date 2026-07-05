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
    if node in {"router", "planner"}:
        return OpenInferenceSpanKindValues.AGENT.value
    if node in {"retrieval_worker", "memory_worker", "timeline_worker", "web_worker"}:
        return OpenInferenceSpanKindValues.RETRIEVER.value
    return OpenInferenceSpanKindValues.CHAIN.value


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
            attributes=attributes,
            input_data=input_data,
            output_data=output_data,
            events=events,
            links=links,
            raw=dict(event),
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
            attributes=attributes,
            input_data=_clean_dict({"value": _bounded_value(event.get("prompt_summary")), "mime_type": "application/json"}),
            output_data=_clean_dict({"value": _bounded_value(llm_summary), "mime_type": "application/json"}),
            events=events,
            links=[],
            raw={"node": node, "llm": dict(llm_summary)},
            order=100 + index + 0.1,
        )

    def record_tool_event(self, event: Mapping[str, Any]) -> None:
        index = self._tool_index
        self._tool_index += 1
        enriched = enrich_tool_event(event)
        tool_name = str(enriched.get("tool_name") or f"tool_{index}")
        caller_node = enriched.get("caller_node")
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
            attributes=attributes,
            input_data=_clean_dict({"value": enriched.get("tool_input"), "mime_type": "application/json"}),
            output_data=output_data,
            events=events,
            links=_span_links_from_refs(enriched.get("artifact_refs")),
            raw=enriched,
            order=1000 + index,
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
        metrics = _as_dict(trace.get("metrics"))
        spans = _as_list(trace.get("spans"))
        node_spans = [span for span in spans if _as_dict(span.get("attributes")).get("askpdf.node.id") and span.get("kind") != OpenInferenceSpanKindValues.LLM.value]
        tool_spans = [span for span in spans if span.get("kind") == OpenInferenceSpanKindValues.TOOL.value]
        errors = [
            _as_dict(event.get("attributes"))
            for span in spans
            for event in _as_list(span.get("events"))
            if isinstance(event, dict) and event.get("name") == "exception"
        ]
        nodes = [_summary_node(span) for span in node_spans]
        tools = [_summary_tool(span) for span in tool_spans]
        used_node_count = sum(1 for node in nodes if not node.get("skipped"))
        warning_count = sum(len(node.get("warningCodes") or []) for node in nodes) + sum(len(tool.get("warningCodes") or []) for tool in tools)
        error_count = int(metrics.get("error_count") or 0)
        if errors:
            error_count = max(error_count, len(errors))
        return {
            "status": trace.get("status"),
            "route": _as_dict(trace.get("attributes")).get("askpdf.route") or metrics.get("route"),
            "routeReason": _as_dict(trace.get("attributes")).get("askpdf.route_reason"),
            "durationMs": trace.get("duration_ms"),
            "metrics": metrics,
            "nodes": nodes,
            "tools": tools,
            "usedNodeCount": used_node_count,
            "availableNodeCount": len(_as_list(_as_dict(_as_dict(self.resolved_spec.get("config")).get("graph")).get("nodes"))) or None,
            "usedToolCount": int(metrics.get("tool_event_count") or len(tools)),
            "availableToolCount": len(set(_as_list(_as_dict(self.resolved_spec.get("config")).get("allowed_tool_ids")))) or None,
            "warningCount": int(metrics.get("tool_warning_count") or warning_count),
            "errorCount": error_count,
            "errors": [error for error in errors if error],
        }


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
        "status": span.get("status"),
        "skipped": span.get("status") == "skipped",
        "durationMs": span.get("duration_ms"),
        "route": attributes.get("askpdf.route"),
        "routeReason": attributes.get("askpdf.route_reason"),
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
        "ok": span.get("status") != "error",
        "durationMs": span.get("duration_ms"),
        "sourceCount": attributes.get("askpdf.source_count"),
        "warningCodes": [str(warning) for warning in warning_events if warning],
        "span": dict(span),
        "raw": raw,
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
                "skipped": bool(summary_node.get("skipped")),
                "skipReason": raw.get("skip_reason"),
                "executionPlan": execution_plan if node_id == "planner" else None,
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
            for route, target in edge["routes"].items():
                if not isinstance(target, str):
                    continue
                selected = selected_route == route
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
