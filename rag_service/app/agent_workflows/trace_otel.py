from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from app.agent.tool_registry import get_tool_contract_metadata
from app.agent_workflows.corrective_contracts import CorrectiveEventName
from app.agent_workflows.enums import NodeEventStatus, TraceSpanKind, TraceStatus
from app.agent_workflows.trace_sanitization import (
    _as_dict,
    _as_list,
    _bounded_value,
    _clean_dict,
)
from app.time_utils import iso_utc_z


class _BufferedSpanExporter(SpanExporter):
    """Local OpenTelemetry exporter used to normalize one agent run."""

    def __init__(self) -> None:
        self.spans: List[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


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
    if event.get("error") or event.get("ok") is False or status in {NodeEventStatus.FAILED.value, TraceStatus.ERROR.value}:
        return TraceStatus.ERROR.value
    if status == NodeEventStatus.SKIPPED.value or event.get("skipped") is True:
        return NodeEventStatus.SKIPPED.value
    return str(event.get("status") or NodeEventStatus.COMPLETED.value)


def _otel_status(status: str) -> Status:
    return Status(StatusCode.ERROR if status == TraceStatus.ERROR.value else StatusCode.OK)


def _node_metadata(node_type: Optional[str]) -> Dict[str, Any]:
    # Framework runtimes emit observability attributes on canonical events.
    # The product trace layer does not load a framework node catalog.
    return {}


def _observability_metadata(node_type: Optional[str]) -> Dict[str, Any]:
    metadata = _node_metadata(node_type)
    observability = metadata.get("observability")
    return observability if isinstance(observability, dict) else {}


def _catalog_display_name(node: str) -> str:
    metadata = _node_metadata(node)
    display_name = metadata.get("display_name")
    if isinstance(display_name, str) and display_name:
        return display_name
    return node.replace("_", " ").title()


def _node_kind(node: str, node_type: Optional[str] = None) -> str:
    observability = _observability_metadata(node_type or node)
    span_kind = str(observability.get("span_kind") or "")
    if span_kind in {TraceSpanKind.CONTROL.value, TraceSpanKind.HUMAN_REVIEW.value}:
        return OpenInferenceSpanKindValues.AGENT.value
    if span_kind == TraceSpanKind.TOOL_WORKER.value:
        return OpenInferenceSpanKindValues.RETRIEVER.value
    return OpenInferenceSpanKindValues.CHAIN.value


def _node_display_name(node: str) -> str:
    if node == "web_approval_gate":
        return "Web Approval"
    return _catalog_display_name(node)


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
                        "askpdf.corrective.decision": event.get("corrective_decision"),
                        "askpdf.corrective.grounding_route": event.get("grounded_answer_route"),
                        "askpdf.corrective.budget_exhausted_reason": event.get("budget_exhausted_reason"),
                        "askpdf.corrective.citation_violation_count": event.get("citation_violation_count"),
                        "askpdf.corrective.contradiction_count": event.get("contradiction_count"),
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
    retrieval_report = _as_dict(event.get("retrieval_quality_report"))
    if retrieval_report:
        result.append({
            "name": CorrectiveEventName.DECISION,
            "attributes": _clean_dict({"askpdf.corrective.decision": event.get("corrective_decision"), "askpdf.corrective.confidence": retrieval_report.get("confidence")}),
            "output": _bounded_value(retrieval_report),
        })
        for contradiction in _as_list(retrieval_report.get("material_contradictions")):
            result.append({"name": CorrectiveEventName.CONTRADICTION, "output": _bounded_value(contradiction)})
        for gap in _as_list(retrieval_report.get("missing_requirements")):
            result.append({"name": CorrectiveEventName.UNRESOLVED_GAP, "attributes": {"askpdf.corrective.gap": str(gap)}})
    grounding_report = _as_dict(event.get("grounding_report"))
    if grounding_report:
        result.append({"name": CorrectiveEventName.SUPPORT_VERIFIED, "attributes": _clean_dict({"askpdf.corrective.grounding_route": event.get("grounded_answer_route"), "askpdf.corrective.support_ratio": grounding_report.get("supported_claim_ratio")}), "output": _bounded_value(grounding_report)})
        for violation in _as_list(grounding_report.get("citation_violations")):
            result.append({"name": CorrectiveEventName.CITATION_VIOLATION, "attributes": {"askpdf.corrective.violation": str(violation)}})
        for contradiction in _as_list(grounding_report.get("contradictions")):
            result.append({"name": CorrectiveEventName.CONTRADICTION, "output": _bounded_value(contradiction)})
        for gap in _as_list(grounding_report.get("unresolved_gaps")):
            result.append({"name": CorrectiveEventName.UNRESOLVED_GAP, "attributes": {"askpdf.corrective.gap": str(gap)}})
    for proposal in _as_list(event.get("work_item_proposals")):
        result.append({
            "name": CorrectiveEventName.QUERY_REWRITE,
            "attributes": _clean_dict({"askpdf.corrective.worker": proposal.get("worker_node_id"), "askpdf.corrective.file_hash": proposal.get("file_hash")}),
            "output": _bounded_value({"query": proposal.get("query"), "reason": proposal.get("reason")}),
        })
    if event.get("budget_exhausted_reason"):
        result.append({"name": CorrectiveEventName.BUDGET_EXHAUSTED, "attributes": {"askpdf.corrective.budget": str(event.get("budget_exhausted_reason"))}})
    event_name = event.get("event_name")
    if isinstance(event_name, str) and event_name in {
        "evaluation.completed",
        "replan.requested",
        "replan.skipped",
        "replan.budget_exhausted",
        CorrectiveEventName.RETRIEVAL_GRADED,
        CorrectiveEventName.SUPPORT_VERIFIED,
        "corrective.finalized_cautiously",
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
                        "askpdf.corrective.decision": event.get("corrective_decision"),
                        "askpdf.corrective.grounding_route": event.get("grounded_answer_route"),
                        "askpdf.corrective.budget_exhausted_reason": event.get("budget_exhausted_reason"),
                    }
                ),
                "output": _clean_dict(
                    {
                        "evaluator_report": _bounded_value(event.get("evaluator_report")),
                        "evidence_gaps": _bounded_value(event.get("evidence_gaps")),
                        "execution_plan": _bounded_value(event.get("execution_plan")),
                        "retrieval_quality_report": _bounded_value(event.get("retrieval_quality_report")),
                        "grounding_report": _bounded_value(event.get("grounding_report")),
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
