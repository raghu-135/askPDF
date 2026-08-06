from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


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
            metrics["parallel_attempts"] = [dict(item) for item in attempts[-256:] if isinstance(item, dict)]
        for key in ("planned", "completed", "skipped", "failed", "timed_out", "cancelled", "retried"):
            metrics[f"parallel_worker_{key}"] = int(parallel.get(key) or 0)
        metrics["parallel_partial_evidence"] = bool(parallel.get("partial_evidence"))
    return metrics
