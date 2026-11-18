"""Emit missing parallel lifecycle events when a run completes after HITL pauses."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.product_orchestration.parallel_projection_contracts import ParallelEventName


def _completed_worker_ids(events: Sequence[Any]) -> set[str]:
    completed: set[str] = set()
    for event in events:
        kind = str(getattr(event, "kind", "") or "")
        if not kind.startswith("worker."):
            continue
        status = kind.removeprefix("worker.")
        if status not in {"completed", "skipped", "failed", "timed_out", "cancelled"}:
            continue
        payload = getattr(event, "payload", None)
        if isinstance(payload, dict) and payload.get("work_id"):
            completed.add(str(payload["work_id"]))
    return completed


def _worker_packets(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    packets = result.get("worker_result_packets") or result.get("_worker_result_packets")
    if isinstance(packets, list) and packets:
        return [dict(item) for item in packets if isinstance(item, dict)]
    records = result.get("parallel_attempt_records") or result.get("_parallel_attempt_records") or []
    if not isinstance(records, list):
        return []
    latest: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, Mapping):
            continue
        work_id = str(item.get("work_id") or "")
        if not work_id:
            continue
        prior = latest.get(work_id)
        if prior is None or int(item.get("attempt") or 0) >= int(prior.get("attempt") or 0):
            latest[work_id] = dict(item)
    return list(latest.values())


async def reconcile_parallel_terminal_events(
    sink: Any,
    result: Mapping[str, Any],
) -> None:
    """Backfill worker and aggregation terminal events omitted during checkpoint pauses."""

    if sink is None or not hasattr(sink, "emit"):
        return
    summary = result.get("parallel_summary") or result.get("dispatch_summary")
    if not isinstance(summary, dict):
        return
    dispatch_id = str(summary.get("dispatch_id") or result.get("dispatch_id") or "")
    if not dispatch_id:
        return
    canonical = getattr(sink, "canonical_events", lambda: [])()
    kinds = {str(getattr(event, "kind", "") or "") for event in canonical}
    completed_workers = _completed_worker_ids(canonical)
    parent_operation = str(
        summary.get("parent_operation_id")
        or result.get("dispatch_node_id")
        or "parallel_dispatch"
    )
    base = {
        "agent_run_id": result.get("agent_run_id"),
        "dispatch_id": dispatch_id,
        "parent_operation_id": parent_operation,
    }
    terminal_statuses = {"completed", "skipped", "failed", "timed_out", "cancelled"}
    for packet in _worker_packets(result):
        work_id = str(packet.get("work_id") or packet.get("worker_node_id") or "")
        status = str(packet.get("status") or "completed")
        if status not in terminal_statuses or not work_id or work_id in completed_workers:
            continue
        event_kind = f"worker.{status}"
        await sink.emit(
            event_kind,
            {
                **base,
                "work_id": work_id,
                "operation_id": packet.get("operation_id") or packet.get("worker_node_id") or work_id,
                "worker_node_id": packet.get("worker_node_id"),
                "worker_type": packet.get("worker_type"),
                "ordinal": packet.get("ordinal"),
                "attempt": packet.get("attempt") or 1,
                "elapsed_ms": packet.get("elapsed_ms"),
                "status": status,
            },
        )
        completed_workers.add(work_id)
        kinds.add(event_kind)
    barrier = str(summary.get("barrier_state") or "")
    aggregation = str(summary.get("aggregation_state") or summary.get("status") or "")
    if barrier == "reached" and ParallelEventName.BARRIER_REACHED not in kinds:
        await sink.emit(
            ParallelEventName.BARRIER_REACHED,
            {
                **base,
                "result_count": int(summary.get("completed") or len(_worker_packets(result)) or 0),
            },
        )
        kinds.add(ParallelEventName.BARRIER_REACHED)
    if aggregation in {"completed", "partial"}:
        agg_kind = (
            ParallelEventName.AGGREGATION_PARTIAL
            if aggregation == "partial"
            else ParallelEventName.AGGREGATION_COMPLETED
        )
        if agg_kind not in kinds:
            await sink.emit(
                agg_kind,
                {
                    **base,
                    **{key: int(summary.get(key) or 0) for key in ("planned", "completed", "skipped", "failed", "timed_out", "cancelled")},
                    "partial_evidence": bool(summary.get("partial_evidence")),
                    "status": aggregation,
                },
            )