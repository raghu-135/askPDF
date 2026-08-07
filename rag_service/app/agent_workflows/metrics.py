from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from app.agent_workflows.parallel_contracts import (
    PARALLEL_EVENT_JOURNAL_LIMIT,
    PARALLEL_SUMMARY_COUNT_FIELDS,
    PARALLEL_SUMMARY_METRIC_FIELDS,
    ParallelEventName,
)
from app.agent_workflows.parallel_observability import project_parallel_events


def _dict_events(events: Any) -> List[Dict[str, Any]]:
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _elapsed_ms(event: Mapping[str, Any]) -> float:
    try:
        return float(event.get("elapsed_ms") or 0)
    except (TypeError, ValueError):
        return 0.0


def _sum_elapsed(events: Iterable[Mapping[str, Any]]) -> float:
    return round(sum(_elapsed_ms(event) for event in events), 2)


def build_run_metrics(result: Mapping[str, Any], *, duration_ms: float) -> Dict[str, Any]:
    """Build persisted run-level observability metrics from graph output."""

    node_events = _dict_events(result.get("node_events"))
    tool_events = _dict_events(result.get("tool_events"))
    errors = _dict_events(result.get("errors"))

    node_elapsed_ms: Dict[str, float] = {}
    replan_count = 0
    evaluation_confidence = None
    for event in node_events:
        node = event.get("node") or event.get("name")
        if not isinstance(node, str) or not node:
            continue
        node_elapsed_ms[node] = round(node_elapsed_ms.get(node, 0.0) + _elapsed_ms(event), 2)
        try:
            replan_count = max(replan_count, int(event.get("replan_count") or 0))
        except (TypeError, ValueError):
            pass
        if event.get("evaluation_confidence") is not None:
            evaluation_confidence = event.get("evaluation_confidence")

    metrics = {
        "duration_ms": round(float(duration_ms), 2),
        "route": result.get("route"),
        "node_event_count": len(node_events),
        "node_elapsed_ms": node_elapsed_ms,
        "node_total_elapsed_ms": _sum_elapsed(node_events),
        "tool_event_count": len(tool_events),
        "tool_warning_count": sum(len(event.get("warnings") or []) for event in tool_events),
        "tool_error_count": sum(1 for event in tool_events if not event.get("ok", True)),
        "tool_elapsed_ms": _sum_elapsed(tool_events),
        "error_count": len(errors) if errors else (1 if result.get("agent_error") else 0),
        "document_source_count": len(result.get("document_sources") or []),
        "web_source_count": len(result.get("web_sources") or []),
        "used_chat_id_count": len(result.get("used_chat_ids") or []),
        "clarification": bool(result.get("clarification_options")),
        "replan_count": replan_count,
        "evaluation_confidence": evaluation_confidence,
    }
    parallel = result.get("parallel_summary") if isinstance(result.get("parallel_summary"), dict) else None
    if parallel:
        metrics["parallel_summary"] = dict(parallel)
        attempts = result.get("parallel_attempts")
        if not isinstance(attempts, list):
            attempts = [
                {"event": f"worker.{item.get('status') or 'unknown'}", "data": dict(item)}
                for item in (result.get("_parallel_attempt_records") or result.get("parallel_attempt_records") or [])
                if isinstance(item, dict)
            ]
        if isinstance(attempts, list):
            projected = project_parallel_events([item for item in attempts if isinstance(item, dict)])
            metrics["parallel_attempts"] = projected["journal"][-PARALLEL_EVENT_JOURNAL_LIMIT:]
        for key in PARALLEL_SUMMARY_COUNT_FIELDS:
            metrics[f"parallel_worker_{key}"] = int(parallel.get(key) or 0)
        for key in PARALLEL_SUMMARY_METRIC_FIELDS:
            metrics[f"parallel_{key}"] = int(parallel.get(key) or 0)
        worker_latencies = [
            _elapsed_ms((item.get("data") or {}))
            for item in metrics.get("parallel_attempts", [])
            if item.get("event") in {
                ParallelEventName.WORKER_COMPLETED,
                ParallelEventName.WORKER_SKIPPED,
                ParallelEventName.WORKER_FAILED,
                ParallelEventName.WORKER_TIMED_OUT,
                ParallelEventName.WORKER_CANCELLED,
            }
        ]
        metrics["parallel_dispatch_latency_ms"] = float(parallel.get("elapsed_ms") or 0)
        metrics["parallel_worker_latency_ms_total"] = round(sum(worker_latencies), 2)
        metrics["parallel_worker_latency_ms_average"] = round(sum(worker_latencies) / len(worker_latencies), 2) if worker_latencies else 0.0
        metrics["parallel_worker_latency_ms_max"] = round(max(worker_latencies), 2) if worker_latencies else 0.0
        metrics["parallel_partial_evidence"] = bool(parallel.get("partial_evidence"))
    if result.get("workflow_id") == "corrective_self_rag_agent" or result.get("retrieval_quality_report") or result.get("grounding_report"):
        retrieval = result.get("retrieval_quality_report") if isinstance(result.get("retrieval_quality_report"), dict) else {}
        grounding = result.get("grounding_report") if isinstance(result.get("grounding_report"), dict) else {}
        assessments = retrieval.get("packet_assessments") if isinstance(retrieval.get("packet_assessments"), list) else []
        attempts = [item for item in result.get("parallel_attempt_records") or [] if isinstance(item, dict)]
        work_ids = {
            str(item.get("work_id")) for item in result.get("worker_result_packets") or []
            if isinstance(item, dict) and item.get("work_id")
        }
        metrics["corrective"] = {
            "waves": max(0, int(result.get("corrective_wave") or 0)),
            "distinct_work_items": len(work_ids),
            "tool_attempts": len(attempts),
            "tool_retries": sum(1 for item in attempts if int(item.get("attempt") or 1) > 1),
            "accepted_packets": sum(1 for item in assessments if isinstance(item, dict) and item.get("relevant") and item.get("provenance_complete") and not item.get("instruction_injection_risk")),
            "rejected_packets": sum(1 for item in assessments if isinstance(item, dict) and (not item.get("relevant") or not item.get("provenance_complete") or item.get("instruction_injection_risk"))),
            "support_ratio": float(grounding.get("supported_claim_ratio") or 0.0),
            "unsupported_claims": sum(1 for item in grounding.get("claims") or [] if isinstance(item, dict) and item.get("support") != "full"),
            "citation_violations": len(grounding.get("citation_violations") or []),
            "contradictions": len(grounding.get("contradictions") or retrieval.get("material_contradictions") or []),
            "unresolved_gaps": len(grounding.get("unresolved_gaps") or retrieval.get("missing_requirements") or []),
            "partial_wave": bool((result.get("parallel_summary") or {}).get("partial_evidence")),
            "corrective_latency_ms": round(sum(_elapsed_ms(event) for event in node_events if (event.get("node") or event.get("name")) in {"retrieval_quality_grader", "replanner", "grounded_answer_verifier"}), 2),
            "exhausted_budget_type": result.get("corrective_budget_exhausted_reason") or None,
            "history": [dict(item) for item in result.get("corrective_history") or [] if isinstance(item, dict)][-8:],
        }
        metrics["retrieval_quality_report"] = retrieval
        metrics["grounding_report"] = grounding
    return metrics
