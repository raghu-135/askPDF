from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import SpanKind, StatusCode, set_span_in_context
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from app.agent_workflows.enums import NodeEventStatus, TraceStatus
from app.agent_workflows.parallel_contracts import PARALLEL_EVENT_NAMES, PARALLEL_TERMINAL_WORKER_STATUSES, ParallelEventName
from app.agent_workflows.parallel_observability import parallel_span_refs
from app.agent_workflows.trace_otel import (
    _BufferedSpanExporter,
    _artifacts_from_refs,
    _decision_events,
    _exception_attributes,
    _exception_event,
    _llm_completed_event,
    _llm_retry_events,
    _llm_usage_metrics,
    _node_display_name,
    _node_kind,
    _node_metadata,
    _ns_to_iso,
    _observability_metadata,
    _otel_status,
    _parse_time_ns,
    _prompt_event,
    _span_links_from_refs,
    _span_status,
    _tool_kind,
    _warning_events,
    enrich_tool_event,
)
from app.runtime.contracts import AgentRuntimeEvent
from app.agent_workflows.canonical_trace import build_canonical_trace_projection
from app.agent_workflows.trace_payloads import (
    DEBUG_PAYLOAD_VERSION,
    build_interrupt_trace_event,
    build_runtime_trace_event,
    _root_event_key,
)
from app.agent_workflows.trace_sanitization import (
    _as_dict,
    _as_list,
    _bounded_value,
    _clean_dict,
    _otel_attr_value,
    _set_attributes,
)
from app.agent_workflows.trace_summary import _build_summary_from_trace
from app.agent_workflows.trace_details import (
    TRACE_DETAIL_RUN_LIMIT,
    final_output_from_result,
    sanitize_trace_detail,
    state_changes,
    trace_detail_size,
)
from app.agent.reasoning import normalize_ai_response
from app.time_utils import iso_utc_z


TRACE_SCHEMA_VERSION = 2


class AgentTraceRecorder:
    """Small local OpenTelemetry wrapper for one agent run."""

    def __init__(self, run: Any):
        self.run = run
        self.resolved_spec = _as_dict(getattr(run, "resolved_spec_json", None))
        self.run_span_id = f"run:{run.id}"
        self._provider = TracerProvider()
        self._exporter = _BufferedSpanExporter()
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        self._tracer = self._provider.get_tracer("askpdf.agent_workflows")
        self._spans_by_id: Dict[str, Any] = {}
        self._sidecars: Dict[str, Dict[str, Any]] = {}
        self._node_span_by_node: Dict[str, str] = {}
        self._node_span_by_visit: Dict[str, str] = {}
        self._node_spec_by_id = {
            str(node.get("id")): node
            for node in _as_list(_as_dict(_as_dict(self.resolved_spec.get("config")).get("graph")).get("nodes"))
            if isinstance(node.get("id"), str)
        }
        self._node_index = 0
        self._tool_index = 0
        self._node_details: Dict[str, Dict[str, Any]] = {}
        self._tool_details: List[Dict[str, Any]] = []
        self._detail_bytes = 0
        self._detail_limit_reached = False
        self._finalized = False
        self._closed_parallel_span_ids: set[str] = set()
        self._runtime_events: List[AgentRuntimeEvent] = []
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

    @staticmethod
    def _detail_key(node_id: str, visit_index: Any) -> str:
        try:
            visit = max(1, int(visit_index))
        except (TypeError, ValueError):
            visit = 1
        return f"{node_id}:{visit}"

    def _store_detail(self, key: str, detail: Dict[str, Any]) -> Dict[str, Any]:
        previous = self._node_details.get(key)
        previous_safety = _as_dict((previous or {}).get("safety"))
        sanitized, current_safety = sanitize_trace_detail({name: value for name, value in detail.items() if name != "safety"})
        safety = {
            "redacted_fields": list(dict.fromkeys([*_as_list(previous_safety.get("redacted_fields")), *_as_list(current_safety.get("redacted_fields"))])),
            "truncated_fields": list(dict.fromkeys([*_as_list(previous_safety.get("truncated_fields")), *_as_list(current_safety.get("truncated_fields"))])),
            "omitted_fields": list(dict.fromkeys([*_as_list(previous_safety.get("omitted_fields")), *_as_list(current_safety.get("omitted_fields"))])),
            "truncated": bool(previous_safety.get("truncated") or current_safety.get("truncated")),
        }
        if previous_safety.get("run_limit_reached"):
            safety["run_limit_reached"] = True
        payload = {**sanitized, "safety": safety}
        size = trace_detail_size(payload)
        previous_size = trace_detail_size(previous) if previous else 0
        if self._detail_bytes - previous_size + size > TRACE_DETAIL_RUN_LIMIT:
            self._detail_limit_reached = True
            payload = {
                "node_id": detail.get("node_id"),
                "node_type": detail.get("node_type"),
                "visit_index": detail.get("visit_index"),
                "status": detail.get("status"),
                "safety": {
                    "truncated": True,
                    "run_limit_reached": True,
                    "redacted_fields": safety["redacted_fields"],
                    "truncated_fields": list(dict.fromkeys([*safety["truncated_fields"], "detail"])),
                    "omitted_fields": safety["omitted_fields"],
                },
            }
            size = trace_detail_size(payload)
        self._detail_bytes = self._detail_bytes - previous_size + size
        self._node_details[key] = payload
        return payload

    def record_node_started(self, *, node_id: str, node_type: str, visit_index: int, state: Mapping[str, Any]) -> Dict[str, Any]:
        key = self._detail_key(node_id, visit_index)
        existing = self._node_details.get(key) or {}
        return self._store_detail(
            key,
            {
                **existing,
                "node_id": node_id,
                "node_type": node_type,
                "visit_index": visit_index,
                "status": "running",
                "started_at": iso_utc_z(),
                "checkpoint_before": dict(state),
            },
        )

    def record_llm_detail(
        self,
        *,
        node_id: str,
        node_type: str,
        visit_index: int,
        messages: List[Any],
        response: Any,
    ) -> Dict[str, Any]:
        key = self._detail_key(node_id, visit_index)
        existing = self._node_details.get(key) or {
            "node_id": node_id,
            "node_type": node_type,
            "visit_index": visit_index,
        }
        normalized = normalize_ai_response(response)
        prompt = []
        for message in messages:
            prompt.append({
                "role": getattr(message, "type", None) or message.__class__.__name__.replace("Message", "").lower(),
                "content": getattr(message, "content", ""),
            })
        llm = {
            "prompt": prompt,
            "response": normalized.get("answer"),
            "reasoning": normalized.get("reasoning"),
            "reasoning_available": bool(normalized.get("reasoning_available")),
            "reasoning_format": normalized.get("reasoning_format"),
        }
        return self._store_detail(key, {**existing, "llm": llm})

    def record_tool_detail(self, *, payload: Mapping[str, Any], tool_input: Any = None) -> Dict[str, Any]:
        caller = str(payload.get("caller_node") or "")
        visit_index = payload.get("caller_visit_index")
        detail, safety = sanitize_trace_detail({"tool_input": tool_input, "result": dict(payload)})
        row = {**detail, "safety": safety}
        self._tool_details.append(row)
        key = self._detail_key(caller, visit_index)
        existing = self._node_details.get(key)
        if existing is not None:
            self._store_detail(key, {**existing, "tools": [*(existing.get("tools") or []), row]})
        return row

    def record_node_completed(
        self,
        *,
        node_id: str,
        node_type: str,
        visit_index: int,
        state: Mapping[str, Any],
        update: Mapping[str, Any],
        status: str,
        error: Any = None,
        event: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        key = self._detail_key(node_id, visit_index)
        existing = self._node_details.get(key) or {}
        after = {**dict(state), **dict(update)} if error is None else None
        before_safe, _ = sanitize_trace_detail(dict(state))
        after_safe, _ = sanitize_trace_detail(after) if after is not None else (None, {})
        changes = state_changes(before_safe, after_safe) if isinstance(after_safe, dict) else {}
        detail = self._store_detail(
            key,
            {
                **existing,
                "node_id": node_id,
                "node_type": node_type,
                "visit_index": visit_index,
                "status": status,
                "checkpoint_before": dict(state),
                "changes": changes,
                "checkpoint_after": after,
                "output": dict(update),
                "event": dict(event or {}),
                "error": str(error) if error is not None else None,
            },
        )
        if key not in self._node_span_by_visit:
            latest_event = dict(event or {})
            self.record_node_event({
                **latest_event,
                "node": node_id,
                "node_type": node_type,
                "visit_index": visit_index,
                "status": status,
                "start_time": existing.get("started_at"),
                "end_time": iso_utc_z(),
                "route": dict(update).get("route"),
                "route_reason": dict(update).get("route_reason"),
                "output_preview": latest_event.get("output_preview") or {
                    "changed_fields": sorted(str(value) for value in changes.keys())[:40],
                },
                "error": error,
            })
        node_span_id = self._node_span_by_visit.get(key)
        if node_span_id:
            for sidecar in self._sidecars.values():
                attributes = _as_dict(sidecar.get("attributes"))
                if (
                    sidecar.get("kind") == OpenInferenceSpanKindValues.TOOL.value
                    and str(attributes.get("askpdf.caller_node") or "") == node_id
                    and int(attributes.get("askpdf.caller_visit_index") or 1) == visit_index
                ):
                    sidecar["parent_span_id"] = node_span_id
        return detail

    def get_node_detail(self, node_id: str, visit_index: int) -> Optional[Dict[str, Any]]:
        return self._node_details.get(self._detail_key(node_id, visit_index))

    def record_interrupted_snapshot(self, *, interrupt: Mapping[str, Any], state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        node_id = str(interrupt.get("node_id") or interrupt.get("gate_id") or "")
        if not node_id:
            return None
        existing_visits = [
            int(detail.get("visit_index") or 1)
            for detail in self._node_details.values()
            if detail.get("node_id") == node_id
        ]
        visit_index = max(existing_visits) if existing_visits else int((state.get("node_visit_counts") or {}).get(node_id, 0)) + 1
        node_type = str(_as_dict(self._node_spec_by_id.get(node_id)).get("type") or node_id)
        return self.record_node_completed(
            node_id=node_id,
            node_type=node_type,
            visit_index=visit_index,
            state=state,
            update={},
            status=NodeEventStatus.INTERRUPTED.value,
            event={"interrupt": dict(interrupt)},
        )

    def tool_details(self) -> List[Dict[str, Any]]:
        return list(self._tool_details)

    def _node_display_name(self, node: str, node_type: Optional[str] = None) -> str:
        spec = _as_dict(self._node_spec_by_id.get(node))
        label = spec.get("label")
        if isinstance(label, str) and label:
            return label
        if isinstance(node_type, str) and node_type:
            return _node_display_name(node_type)
        return _node_display_name(node)

    def _run_base_attributes(
        self,
        *,
        chat_turn_id: Optional[str],
        metrics: Mapping[str, Any],
        route: Any,
        route_reason: Any,
    ) -> Dict[str, Any]:
        config = _as_dict(self.resolved_spec.get("config"))
        parallel_metrics = {
            f"askpdf.{key.replace('_', '.')}": value
            for key, value in metrics.items()
            if key.startswith("parallel_") and isinstance(value, (bool, int, float, str))
        }
        return _clean_dict(
            {
                SpanAttributes.SESSION_ID: getattr(self.run, "thread_id", None),
                SpanAttributes.USER_ID: getattr(self.run, "user_id", None),
                SpanAttributes.AGENT_NAME: getattr(self.run, "workflow_id", None),
                "askpdf.run.id": getattr(self.run, "id", None),
                "askpdf.thread.id": getattr(self.run, "thread_id", None),
                "askpdf.chat_turn.id": chat_turn_id,
                "askpdf.workflow.id": getattr(self.run, "workflow_id", None),
                "askpdf.workflow_id": self.resolved_spec.get("workflow_id"),
                "askpdf.route": route or metrics.get("route"),
                "askpdf.route_reason": route_reason,
                "askpdf.use_web_search": config.get("use_web_search"),
                "askpdf.use_reranker": config.get("use_reranker"),
                "askpdf.context_window": config.get("context_window"),
                "askpdf.warning_count": metrics.get("tool_warning_count"),
                "askpdf.error_count": metrics.get("error_count"),
                **parallel_metrics,
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
        node_metadata = _node_metadata(node_type)
        observability = _observability_metadata(node_type)
        node_display_name = self._node_display_name(node, node_type)
        status = _span_status(event)
        span_id = f"node:{node}:{index}"
        detail_key = self._detail_key(node, event.get("visit_index"))
        if detail_key in self._node_span_by_visit:
            return
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
        if status == NodeEventStatus.SKIPPED.value:
            events.append(
                {
                    "name": NodeEventStatus.SKIPPED.value,
                    "attributes": _clean_dict({"askpdf.skip_reason": event.get("skip_reason")}),
                }
            )
        attributes = _clean_dict(
            {
                SpanAttributes.GRAPH_NODE_ID: node,
                SpanAttributes.GRAPH_NODE_NAME: node_display_name,
                "askpdf.node.id": node,
                "askpdf.node.type": node_type,
                "askpdf.node.visit_index": event.get("visit_index"),
                "askpdf.node.category": node_metadata.get("category"),
                "askpdf.node.capabilities": node_metadata.get("capabilities"),
                "askpdf.parallel.dispatch_id": event.get("dispatch_id"),
                "askpdf.parallel.work_id": event.get("work_id"),
                "askpdf.parallel.ordinal": event.get("ordinal"),
                "askpdf.parallel.attempt": event.get("attempt"),
                "askpdf.parallel.parent_node_id": event.get("parent_node_id"),
                "askpdf.node.name": node_display_name,
                "askpdf.observability.span_kind": observability.get("span_kind"),
                "askpdf.observability.event_prefix": observability.get("event_prefix"),
                "askpdf.observability.summary_fields": observability.get("summary_fields"),
                "askpdf.observability.raw_payload": observability.get("raw_payload"),
                "askpdf.route": event.get("route"),
                "askpdf.route_reason": event.get("route_reason"),
                "askpdf.evaluator_route": event.get("evaluator_route"),
                "askpdf.evaluation_confidence": event.get("evaluation_confidence"),
                "askpdf.replan_count": event.get("replan_count"),
                "askpdf.corrective.decision": event.get("corrective_decision"),
                "askpdf.corrective.grounding_route": event.get("grounded_answer_route"),
                "askpdf.corrective.budget_exhausted_reason": event.get("budget_exhausted_reason"),
                "askpdf.corrective.citation_violation_count": event.get("citation_violation_count"),
                "askpdf.corrective.contradiction_count": event.get("contradiction_count"),
                "askpdf.replans": event.get("replans"),
                "askpdf.skip_reason": event.get("skip_reason"),
                "askpdf.execution_plan": event.get("execution_plan"),
                "askpdf.evidence_chars": event.get("evidence_chars"),
                "askpdf.answer_chars": event.get("answer_chars"),
                "askpdf.document_source_count": event.get("document_source_count"),
                "askpdf.web_source_count": event.get("web_source_count"),
                "askpdf.used_chat_id_count": event.get("used_chat_id_count"),
                "askpdf.used_memory_id_count": event.get("used_memory_id_count"),
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
        parent_span_id = self._node_span_by_node.get(str(event.get("parent_node_id") or "")) or self.run_span_id
        self._start_span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=node_display_name,
            kind=_node_kind(node, node_type),
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
        self._node_span_by_visit[detail_key] = span_id
        if llm_summary:
            self._record_llm_span(
                node=node,
                node_type=node_type,
                node_display_name=node_display_name,
                parent_span_id=span_id,
                index=index,
                event=event,
                llm_summary=llm_summary,
            )

    def _record_llm_span(
        self,
        *,
        node: str,
        node_type: str,
        node_display_name: str,
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
                "askpdf.node.type": node_type,
                "askpdf.node.visit_index": event.get("visit_index"),
            }
        )
        self._start_span(
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=f"{node_display_name} LLM",
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
        status = TraceStatus.ERROR.value if enriched.get("ok") is False or enriched.get("error") else NodeEventStatus.COMPLETED.value
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
                "askpdf.caller_visit_index": enriched.get("caller_visit_index"),
                "askpdf.parallel.dispatch_id": enriched.get("dispatch_id"),
                "askpdf.parallel.work_id": enriched.get("work_id"),
                "askpdf.parallel.ordinal": enriched.get("ordinal"),
                "askpdf.parallel.attempt": enriched.get("attempt"),
                "askpdf.tool.argument_hash": enriched.get("argument_hash"),
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
        attrs = _as_dict(attributes)
        self._record_parallel_runtime_span(event_name, attrs)
        event = build_runtime_trace_event(
            event_name,
            attributes=attrs,
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

    def record_agent_runtime_event(self, event: AgentRuntimeEvent) -> None:
        """Record one canonical runtime event in the product trace."""

        self._runtime_events.append(event)
        if event.kind != "run.started":
            self.record_runtime_event(event.kind, attributes=dict(event.payload))

    def _record_parallel_runtime_span(self, event_name: str, attributes: Mapping[str, Any]) -> None:
        if event_name not in PARALLEL_EVENT_NAMES:
            return
        attrs = dict(attributes)
        dispatch_id = str(attrs.get("dispatch_id") or "")
        if not dispatch_id:
            return
        dispatch_span_id = f"dispatch:{dispatch_id}"
        if dispatch_span_id not in self._spans_by_id:
            self._start_span(
                span_id=dispatch_span_id,
                parent_span_id=self.run_span_id,
                name="Parallel Dispatch",
                kind=OpenInferenceSpanKindValues.CHAIN.value,
                status="running",
                start_time=attrs.get("occurred_at"),
                attributes=_clean_dict({
                    "askpdf.parallel.dispatch_id": dispatch_id,
                    "askpdf.parallel.planned": attrs.get("planned"),
                }),
                input_data={}, output_data={}, events=[], links=[], raw={}, order=80,
                end_immediately=False,
            )
        dispatch_span = self._spans_by_id[dispatch_span_id]
        event_attrs = {key: _otel_attr_value(value) for key, value in attrs.items() if _otel_attr_value(value) is not None}
        if dispatch_span_id not in self._closed_parallel_span_ids:
            dispatch_span.add_event(event_name, attributes=event_attrs)
        if event_name.startswith("worker.") and attrs.get("work_id"):
            worker_span_id, _ = parallel_span_refs(event_name, attrs)
            if worker_span_id and worker_span_id not in self._spans_by_id and event_name != ParallelEventName.WORKER_QUEUED:
                self._start_span(
                    span_id=worker_span_id,
                    parent_span_id=dispatch_span_id,
                    name=f"Parallel Worker: {attrs.get('worker_node_id') or attrs.get('worker_type') or 'worker'}",
                    kind=OpenInferenceSpanKindValues.RETRIEVER.value,
                    status="running",
                    start_time=attrs.get("occurred_at"),
                    attributes=_clean_dict({
                        "askpdf.parallel.dispatch_id": dispatch_id,
                        "askpdf.parallel.work_id": attrs.get("work_id"),
                        "askpdf.parallel.ordinal": attrs.get("ordinal"),
                        "askpdf.parallel.worker_node_id": attrs.get("worker_node_id"),
                        "askpdf.parallel.worker_type": attrs.get("worker_type"),
                        "askpdf.parallel.attempt": attrs.get("attempt") or 1,
                    }),
                    input_data={}, output_data={}, events=[], links=[], raw={}, order=81 + float(attrs.get("ordinal") or 0) / 100,
                    end_immediately=False,
                )
            worker_span = self._spans_by_id.get(worker_span_id or "")
            if worker_span is not None and worker_span_id not in self._closed_parallel_span_ids:
                worker_span.add_event(event_name, attributes=event_attrs)
                worker_status = event_name.removeprefix("worker.")
                if worker_status in PARALLEL_TERMINAL_WORKER_STATUSES:
                    status = TraceStatus.ERROR.value if worker_status in {"failed", "timed_out"} else worker_status
                    worker_span.set_status(_otel_status(status))
                    sidecar = self._sidecars[worker_span_id]
                    sidecar["status"] = status
                    sidecar["attributes"] = _clean_dict({**sidecar["attributes"], **attrs})
                    sidecar["output"] = {"value": _bounded_value(attrs), "mime_type": "application/json"}
                    worker_span.end(end_time=_parse_time_ns(attrs.get("occurred_at")))
                    self._closed_parallel_span_ids.add(worker_span_id)
        if event_name in {ParallelEventName.AGGREGATION_COMPLETED, ParallelEventName.AGGREGATION_PARTIAL, ParallelEventName.DISPATCH_CANCELLED} and dispatch_span_id not in self._closed_parallel_span_ids:
            dispatch_status = "cancelled" if event_name == ParallelEventName.DISPATCH_CANCELLED else NodeEventStatus.COMPLETED.value
            dispatch_span.set_status(_otel_status(dispatch_status))
            sidecar = self._sidecars[dispatch_span_id]
            sidecar["status"] = dispatch_status
            sidecar["attributes"] = _clean_dict({**sidecar["attributes"], **attrs})
            sidecar["output"] = {"value": _bounded_value(attrs), "mime_type": "application/json"}
            dispatch_span.end(end_time=_parse_time_ns(attrs.get("occurred_at")))
            self._closed_parallel_span_ids.add(dispatch_span_id)

    def finalize(
        self,
        *,
        run: Any,
        chat_turn_id: Optional[str],
        metrics: Dict[str, Any],
        route: Any = None,
        route_reason: Any = None,
        error: Any = None,
        result: Any = None,
    ) -> Dict[str, Any]:
        if not self._finalized:
            self.run = run
            for span_id, span in list(self._spans_by_id.items()):
                if span_id == self.run_span_id or span_id in self._closed_parallel_span_ids:
                    continue
                if not span_id.startswith(("dispatch:", "worker:")):
                    continue
                status = TraceStatus.ERROR.value if error else NodeEventStatus.COMPLETED.value
                span.set_status(_otel_status(status))
                self._sidecars[span_id]["status"] = status
                span.end(end_time=_parse_time_ns(getattr(run, "completed_at", None)))
                self._closed_parallel_span_ids.add(span_id)
            run_attributes = self._run_base_attributes(
                chat_turn_id=chat_turn_id,
                metrics=metrics,
                route=route,
                route_reason=route_reason,
            )
            _set_attributes(self._root_span, run_attributes)
            root_sidecar = self._sidecars[self.run_span_id]
            root_sidecar["attributes"] = _clean_dict({**root_sidecar["attributes"], **run_attributes})
            root_sidecar["status"] = str(getattr(run, "status", None) or NodeEventStatus.COMPLETED.value)
            root_sidecar["attributes"]["askpdf.status"] = root_sidecar["status"]
            if error:
                exception = _exception_event(error)
                if exception:
                    self._root_span.add_event("exception", attributes=_exception_attributes(error))
                    root_sidecar["events"].append(exception)
                self._root_span.set_status(_otel_status(TraceStatus.ERROR.value))
            else:
                self._root_span.set_status(_otel_status(root_sidecar["status"]))
            self._root_span.end(end_time=_parse_time_ns(getattr(run, "completed_at", None)))
            self._finalized = True
        trace = self._build_trace(run=run, chat_turn_id=chat_turn_id, metrics=metrics)
        summary = self._build_summary(trace)
        framework = str(getattr(run, "framework", None) or "")
        canonical = build_canonical_trace_projection(
            events=self._runtime_events,
            resolved_spec=self.resolved_spec,
            framework=framework,
        )
        summary = {
            **summary,
            "operations": canonical["operations"],
            "tools": canonical["tools"],
            "approvalCount": len(canonical["approvals"]),
            "subagentCount": len(canonical["subagents"]),
            "artifactCount": len(canonical["artifacts"]),
            "errorCount": int(canonical["diagnostics"]["summary"].get("failure_count") or 0),
            "usedOperationCount": len({row["operation_id"] for row in canonical["operations"] if row.get("status") != "skipped"}),
        }
        summary.pop("nodes", None)
        summary.pop("usedNodeCount", None)
        summary.pop("availableNodeCount", None)
        summary.pop("errors", None)
        final_output = final_output_from_result(result)
        details = []
        for row in canonical["operations"]:
            stored = self._node_details.get(self._detail_key(str(row["operation_id"]), row.get("visit_index")))
            detail = dict(stored or {})
            detail.pop("node_id", None)
            detail.pop("node_type", None)
            details.append({
                **detail,
                "operation_id": row["operation_id"],
                "operation_type": row.get("operation_type"),
                "visit_index": row.get("visit_index"),
                "status": row.get("status"),
            })
        payload = {
            "version": DEBUG_PAYLOAD_VERSION,
            "trace": {**trace, **canonical, "framework": framework},
            "summary": summary,
            "events": canonical["events"],
            "operations": canonical["operations"],
            "tools": canonical["tools"],
            "approvals": canonical["approvals"],
            "subagents": canonical["subagents"],
            "artifacts": canonical["artifacts"],
            "diagnostics": canonical["diagnostics"],
            "visualizations": canonical["visualizations"],
            "details": details,
            "detail_safety": {
                "size_bytes": self._detail_bytes,
                "run_limit_bytes": TRACE_DETAIL_RUN_LIMIT,
                "truncated": self._detail_limit_reached,
            },
        }
        if final_output:
            payload["final_output"] = final_output
        return payload

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
            "status": sidecar.get("status")
            or attrs.get("askpdf.status")
            or (TraceStatus.ERROR.value if span.status.status_code == StatusCode.ERROR else NodeEventStatus.COMPLETED.value),
            "start_time": _ns_to_iso(span.start_time),
            "end_time": _ns_to_iso(span.end_time),
            "duration_ms": max(0.0, round((span.end_time - span.start_time) / 1_000_000, 2)) if span.start_time and span.end_time else None,
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
            "workflow_id": self.resolved_spec.get("workflow_id") or getattr(run, "workflow_id", None),
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
