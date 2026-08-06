from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from langgraph.types import Send, TimeoutPolicy
from langgraph.errors import NodeTimeoutError

from app.agent_workflows.enums import WorkflowNodeType
from app.agent_workflows.evidence import combine_evidence
from app.agent_workflows.parallel_contracts import (
    DEFAULT_PARALLEL_POLICY,
    PARALLEL_EVENT_NAMES,
    PARALLEL_FEATURE_ENV,
    PARALLEL_REFERENCE_WORKFLOW_ID,
    PARALLEL_TERMINAL_WORKER_STATUSES,
    PARALLEL_WORKER_EVIDENCE_KINDS,
    ParallelEventName,
    normalized_parallel_policy,
    parallel_timeout_watchdog_seconds,
)
from app.models.retry import is_retryable_model_error


WORKER_EVIDENCE_KIND = PARALLEL_WORKER_EVIDENCE_KINDS
TERMINAL_WORKER_STATUSES = PARALLEL_TERMINAL_WORKER_STATUSES


def parallel_feature_enabled() -> bool:
    return os.getenv(PARALLEL_FEATURE_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


class ParallelDispatchDeadlineExceeded(TimeoutError):
    pass


class ParallelWorkerError(Exception):
    def __init__(self, error: BaseException, *, attempt: int, status: str = "failed"):
        super().__init__(str(error))
        self.error = error
        self.attempt = max(1, int(attempt))
        self.status = status


def parallel_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, ParallelWorkerError):
        exc = exc.error
    if isinstance(exc, ParallelDispatchDeadlineExceeded):
        return False
    if isinstance(exc, (ValueError, TypeError, PermissionError, asyncio.CancelledError)):
        return False
    if isinstance(exc, NodeTimeoutError):
        return True
    if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
        return True
    retryable, _ = is_retryable_model_error(str(exc))
    return bool(retryable)


def parallel_runtime_authorized(state: Mapping[str, Any]) -> bool:
    if state.get("parallel_runtime_override") is True:
        return True
    return bool(
        parallel_feature_enabled()
        and state.get("workflow_id") == PARALLEL_REFERENCE_WORKFLOW_ID
    )


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _normalized_query(value: Any, *, fallback: str) -> str:
    text = " ".join(str(value or fallback or "").split()).strip()
    return text[:2_000]


def _proposal_worker_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("worker_node_id") or value.get("node") or value.get("worker") or value.get("id") or "")
    return ""


def normalize_work_items(
    proposals: Any,
    *,
    state: Mapping[str, Any],
    dispatch_node_id: str,
    dispatch_visit: int,
) -> List[Dict[str, Any]]:
    policy = normalized_parallel_policy(state.get("parallel_policy"))
    available = state.get("available_worker_nodes") if isinstance(state.get("available_worker_nodes"), list) else []
    by_id = {
        str(item.get("id")): item
        for item in available
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and str(item.get("type")) in WORKER_EVIDENCE_KIND
    }
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for item in by_id.values():
        by_type.setdefault(str(item.get("type")), []).append(item)

    raw_items = proposals if isinstance(proposals, list) else []
    dispatch_id = _stable_hash({
        "run": state.get("agent_run_id"),
        "node": dispatch_node_id,
        "visit": dispatch_visit,
    })
    normalized: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_items:
        worker_id = _proposal_worker_id(raw)
        worker = by_id.get(worker_id)
        if worker is None and worker_id in by_type and len(by_type[worker_id]) == 1:
            worker = by_type[worker_id][0]
            worker_id = str(worker["id"])
        if worker is None:
            continue
        worker_type = str(worker.get("type"))
        query = _normalized_query(raw.get("query") if isinstance(raw, dict) else None, fallback=str(state.get("question") or ""))
        key = (worker_type, query.casefold())
        if not query or key in seen:
            continue
        if worker_type == WorkflowNodeType.WEB_WORKER.value and not state.get("use_web_search", False):
            continue
        seen.add(key)
        ordinal = len(normalized)
        dedupe_key = _stable_hash({"worker_type": worker_type, "query": query.casefold()})
        work_id = _stable_hash({
            "dispatch_id": dispatch_id,
            "ordinal": ordinal,
            "worker_node_id": worker_id,
            "query": query.casefold(),
        })
        timeout_ms = (
            policy["web_worker_timeout_ms"]
            if worker_type == WorkflowNodeType.WEB_WORKER.value
            else policy["default_worker_timeout_ms"]
        )
        normalized.append({
            "dispatch_id": dispatch_id,
            "dispatch_visit": dispatch_visit,
            "work_id": work_id,
            "ordinal": ordinal,
            "worker_node_id": worker_id,
            "worker_type": worker_type,
            "query": query,
            "reason": str(raw.get("reason") or "")[:500] if isinstance(raw, dict) else "",
            "evidence_kind": WORKER_EVIDENCE_KIND[worker_type],
            "dedupe_key": dedupe_key,
            "attempt": 1,
            "timeout_ms": timeout_ms,
        })
        if len(normalized) >= policy["max_work_items"]:
            break
    return normalized


def work_item_proposals(parsed: Mapping[str, Any], execution_plan: Sequence[str], question: str) -> List[Dict[str, Any]]:
    proposals = parsed.get("work_items")
    if isinstance(proposals, list):
        return [item for item in proposals if isinstance(item, (dict, str))]
    return [{"worker_node_id": worker_id, "query": question, "reason": "planner execution plan"} for worker_id in execution_plan]


def dispatch_sends(state: Mapping[str, Any]) -> List[Send]:
    policy = normalized_parallel_policy(state.get("parallel_policy"))
    terminal_ids = {
        str(packet.get("work_id"))
        for packet in state.get("worker_result_packets", [])
        if isinstance(packet, dict) and packet.get("status") in TERMINAL_WORKER_STATUSES
    }
    pending = [
        item for item in state.get("work_items", [])
        if isinstance(item, dict) and str(item.get("work_id")) not in terminal_ids
    ]
    if not pending:
        aggregator = str(state.get("parallel_aggregator_id") or "")
        return [Send(aggregator, dict(state))] if aggregator else []
    common = dict(state)
    return [
        Send(
            str(item["worker_node_id"]),
            {
                **common,
                "question": str(item.get("query") or state.get("question") or ""),
                "work_item": dict(item),
            },
            timeout=TimeoutPolicy(
                run_timeout=parallel_timeout_watchdog_seconds(
                    int(item.get("timeout_ms") or policy["default_worker_timeout_ms"]),
                    policy["max_attempts"],
                )
            ),
        )
        for item in pending
    ]


def _sort_key(packet: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(packet.get("dispatch_visit") or 1),
        int(packet.get("ordinal") or 0),
        str(packet.get("worker_node_id") or ""),
        str(packet.get("work_id") or ""),
        int(packet.get("attempt") or 1),
    )


def _normalized_url(value: Any) -> str:
    try:
        parts = urlsplit(str(value or "").strip())
    except ValueError:
        return ""
    if not parts.netloc:
        return ""
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    return urlunsplit((parts.scheme.lower() or "https", parts.netloc.lower(), path.rstrip("/") or "/", parts.query, ""))


def _content_hash(value: Any) -> str:
    normalized = " ".join(str(value or "").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _document_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    identity = item.get("document_id") or item.get("file_id") or item.get("file_hash")
    page = item.get("page_number") or item.get("page") or item.get("page_start")
    locator = item.get("section_id") or item.get("chunk_id") or item.get("section")
    if identity:
        return (str(identity), page, str(locator or ""))
    return (str(item.get("title") or item.get("file_name") or "").casefold(), page, _content_hash(item.get("content") or item.get("text")))


def _web_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    url = _normalized_url(item.get("url") or item.get("source_url") or item.get("link"))
    if url:
        return (url,)
    return (str(item.get("title") or "").casefold(), _content_hash(item.get("content") or item.get("text") or item.get("snippet")))


def _merge_unique_dicts(items: Iterable[Any], key_fn) -> List[Dict[str, Any]]:
    merged: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        key = key_fn(raw)
        if key not in merged:
            merged[key] = dict(raw)
            order.append(key)
            continue
        for field, value in raw.items():
            if field not in merged[key] or merged[key][field] in (None, "", [], {}):
                merged[key][field] = value
    return [merged[key] for key in order]


def _dedupe_evidence_first(items: Iterable[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        refs = item.get("refs") if isinstance(item.get("refs"), dict) else {}
        primary_ref = (
            item.get("primary_reference_key")
            or item.get("source_id")
            or item.get("document_id")
            or item.get("memory_id")
            or item.get("message_id")
            or item.get("url")
            or refs.get("document_id")
            or refs.get("memory_id")
            or refs.get("message_id")
            or refs.get("url")
            or ""
        )
        key = (
            str(item.get("kind") or ""),
            str(item.get("content_hash") or _content_hash(item.get("content"))),
            str(primary_ref),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result


def _dedupe_dicts(items: Iterable[Any], key_fn) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: set[Any] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result


def _unique_strings(items: Iterable[Any]) -> List[str]:
    values: List[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "")
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def worker_terminal_delta(
    item: Mapping[str, Any],
    *,
    status: str,
    attempt: int,
    output: Mapping[str, Any] | None = None,
    lifecycle_events: Sequence[Mapping[str, Any]] = (),
    errors: Sequence[Mapping[str, Any]] = (),
    started_at: str = "",
    completed_at: str = "",
    elapsed_ms: float = 0.0,
) -> Dict[str, Any]:
    """Build one immutable terminal packet and its reducer-safe artifact deltas."""

    local = dict(output or {})
    work_id = str(item.get("work_id") or "")
    dispatch_id = str(item.get("dispatch_id") or "")
    worker_node_id = str(item.get("worker_node_id") or "")
    worker_type = str(item.get("worker_type") or "")
    node_events = [
        {
            **dict(event),
            "parent_node_id": item.get("dispatch_node_id") or WorkflowNodeType.PARALLEL_DISPATCH.value,
            "dispatch_id": dispatch_id,
            "work_id": work_id,
            "ordinal": item.get("ordinal"),
            "attempt": attempt,
        }
        for event in [*lifecycle_events, *(local.get("node_events") or [])]
        if isinstance(event, Mapping)
    ]
    tool_events = [
        {
            **dict(event),
            "parent_node_id": item.get("dispatch_node_id") or WorkflowNodeType.PARALLEL_DISPATCH.value,
            "dispatch_id": dispatch_id,
            "work_id": work_id,
            "ordinal": item.get("ordinal"),
            "attempt": attempt,
            "invocation_sequence": index,
        }
        for index, event in enumerate(local.get("tool_events") or [], start=1)
        if isinstance(event, Mapping)
    ]
    timeline_refs = [
        dict(ref)
        for event in node_events
        for ref in (
            (event.get("output_refs") or {}).get("timeline_events", [])
            if isinstance(event.get("output_refs"), dict)
            else []
        )
        if isinstance(ref, Mapping)
    ]
    normalized_errors = [
        {
            **dict(error),
            "dispatch_id": dispatch_id,
            "work_id": work_id,
            "attempt": attempt,
        }
        for error in errors
        if isinstance(error, Mapping)
    ]
    memory_refs = [
        {"memory_id": value, "dispatch_id": dispatch_id, "work_id": work_id}
        for value in local.get("used_memory_ids") or []
        if value
    ]
    visits = [
        {
            "node_id": worker_node_id,
            "node": worker_node_id,
            "node_type": worker_type,
            "dispatch_id": dispatch_id or "serial",
            "work_id": work_id,
            "attempt": visit_attempt,
            "visit_index": int(item.get("ordinal") or 0) + 1,
        }
        for visit_attempt in range(1, max(1, attempt) + 1)
    ]
    attempts = [
        {
            "dispatch_id": dispatch_id,
            "work_id": work_id,
            "ordinal": item.get("ordinal"),
            "worker_node_id": worker_node_id,
            "worker_type": worker_type,
            "attempt": visit_attempt,
            "status": status if visit_attempt == attempt else "retried",
            "started_at": started_at if visit_attempt == attempt else "",
            "completed_at": completed_at if visit_attempt == attempt else "",
            "elapsed_ms": elapsed_ms if visit_attempt == attempt else 0.0,
            "parent_node_id": item.get("dispatch_node_id") or WorkflowNodeType.PARALLEL_DISPATCH.value,
        }
        for visit_attempt in range(1, max(1, attempt) + 1)
    ]
    packet = {
        **dict(item),
        "attempt": attempt,
        "status": status,
        "evidence_packets": list(local.get("evidence_packets") or []),
        "document_sources": list(local.get("document_sources") or []),
        "web_sources": list(local.get("web_sources") or []),
        "chat_ids": list(local.get("used_chat_ids") or []),
        "memory_refs": memory_refs,
        "node_events": node_events,
        "tool_events": tool_events,
        "errors": normalized_errors,
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_ms": elapsed_ms,
    }
    return {
        "worker_result_packets": [packet],
        "parallel_evidence_deltas": packet["evidence_packets"],
        "parallel_document_source_deltas": packet["document_sources"],
        "parallel_web_source_deltas": packet["web_sources"],
        "parallel_chat_id_deltas": packet["chat_ids"],
        "parallel_memory_ref_deltas": memory_refs,
        "parallel_timeline_ref_deltas": timeline_refs,
        "parallel_node_event_deltas": node_events,
        "parallel_tool_event_deltas": tool_events,
        "parallel_error_deltas": normalized_errors,
        "parallel_skipped_work_deltas": [
            {"work_id": work_id, "worker_node_id": worker_node_id, "reason": "worker_skipped"}
        ] if status == "skipped" else [],
        "parallel_visit_records": visits,
        "parallel_attempt_records": attempts,
    }


def cancelled_parallel_dispatch(
    state: Mapping[str, Any],
    event_envelopes: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Materialize terminal cancellation data for every unfinished dispatched item."""

    work_items = [dict(item) for item in state.get("work_items", []) if isinstance(item, dict)]
    if not work_items:
        queued: Dict[str, Dict[str, Any]] = {}
        for envelope in event_envelopes:
            if envelope.get("event") != ParallelEventName.WORKER_QUEUED:
                continue
            data = envelope.get("data") if isinstance(envelope.get("data"), dict) else {}
            if data.get("work_id"):
                queued[str(data["work_id"])] = dict(data)
        work_items = list(queued.values())
    terminal = {
        str(packet.get("work_id")): dict(packet)
        for packet in state.get("worker_result_packets", [])
        if isinstance(packet, dict) and packet.get("status") in TERMINAL_WORKER_STATUSES
    }
    attempts: Dict[str, int] = {}
    active: set[str] = set()
    for envelope in event_envelopes:
        event = str(envelope.get("event") or "")
        data = envelope.get("data") if isinstance(envelope.get("data"), dict) else {}
        work_id = str(data.get("work_id") or "")
        if not work_id:
            continue
        attempts[work_id] = max(attempts.get(work_id, 1), int(data.get("attempt") or 1))
        if event == ParallelEventName.WORKER_STARTED:
            active.add(work_id)
        elif event in {
            ParallelEventName.WORKER_COMPLETED,
            ParallelEventName.WORKER_FAILED,
            ParallelEventName.WORKER_TIMED_OUT,
            ParallelEventName.WORKER_SKIPPED,
        }:
            active.discard(work_id)
    combined: Dict[str, Any] = {}
    cancelled_count = 0
    for item in work_items:
        work_id = str(item.get("work_id") or "")
        if not work_id or work_id in terminal:
            continue
        cancelled_count += 1
        delta = worker_terminal_delta(
            item,
            status="cancelled",
            attempt=attempts.get(work_id, 1),
            lifecycle_events=[{
                "name": ParallelEventName.WORKER_CANCELLED,
                "status": "cancelled",
                "reason": "active_cancelled" if work_id in active else "queued_cancelled",
            }],
            completed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        for key, values in delta.items():
            combined.setdefault(key, []).extend(values)
    existing_statuses = [packet.get("status") for packet in terminal.values()]
    planned = len(work_items)
    summary = {
        "agent_run_id": state.get("agent_run_id"),
        "dispatch_id": state.get("dispatch_id") or (work_items[0].get("dispatch_id") if work_items else None),
        "planned": planned,
        "completed": existing_statuses.count("completed"),
        "skipped": existing_statuses.count("skipped"),
        "failed": existing_statuses.count("failed"),
        "timed_out": existing_statuses.count("timed_out"),
        "cancelled": cancelled_count + existing_statuses.count("cancelled"),
        "retried": sum(max(0, value - 1) for value in attempts.values()),
        "partial_evidence": False,
        "status": "cancelled",
        "barrier_state": "cancelled",
        "aggregation_state": "cancelled",
    }
    return {**combined, "parallel_summary": summary}


def _peak_concurrency(packets: Sequence[Mapping[str, Any]], *, maximum: int) -> int:
    boundaries: List[tuple[float, int]] = []
    for packet in packets:
        try:
            started = datetime.fromisoformat(str(packet.get("started_at") or "").replace("Z", "+00:00")).timestamp()
            completed = datetime.fromisoformat(str(packet.get("completed_at") or "").replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            continue
        boundaries.extend(((started, 1), (max(started, completed), -1)))
    active = peak = 0
    for _, delta in sorted(boundaries, key=lambda item: (item[0], item[1])):
        active = max(0, active + delta)
        peak = max(peak, active)
    return min(maximum, peak)


def aggregate_parallel_results(state: Mapping[str, Any]) -> Dict[str, Any]:
    packets = sorted(
        [dict(item) for item in state.get("worker_result_packets", []) if isinstance(item, dict)],
        key=_sort_key,
    )
    final_by_work: Dict[str, Dict[str, Any]] = {}
    attempts_by_work: Dict[str, int] = {}
    for packet in packets:
        work_id = str(packet.get("work_id") or "")
        attempts_by_work[work_id] = max(attempts_by_work.get(work_id, 0), int(packet.get("attempt") or 1))
        final_by_work[work_id] = packet
    terminal = sorted(final_by_work.values(), key=_sort_key)
    successful = [item for item in terminal if item.get("status") == "completed"]
    skipped = [item for item in terminal if item.get("status") == "skipped"]
    failed = [item for item in terminal if item.get("status") == "failed"]
    timed_out = [item for item in terminal if item.get("status") == "timed_out"]
    cancelled = [item for item in terminal if item.get("status") == "cancelled"]

    worker_evidence_input = state.get("parallel_evidence_deltas") or [
        evidence for packet in successful for evidence in packet.get("evidence_packets", [])
    ]
    worker_evidence = _dedupe_evidence_first(worker_evidence_input)
    existing_evidence = [item for item in state.get("evidence_packets", []) if isinstance(item, dict)]
    all_evidence = _dedupe_evidence_first([*existing_evidence, *worker_evidence])
    evidence_text = str(state.get("evidence") or "")
    for packet in worker_evidence:
        evidence_text = combine_evidence(
            evidence_text,
            packet.get("content"),
            label=str(packet.get("kind") or "Worker evidence").replace("_", " ").title(),
        )

    document_sources = _merge_unique_dicts(
        [*state.get("document_sources", []), *(state.get("parallel_document_source_deltas") or (item for packet in successful for item in packet.get("document_sources", [])))],
        _document_key,
    )
    web_sources = _merge_unique_dicts(
        [*state.get("web_sources", []), *(state.get("parallel_web_source_deltas") or (item for packet in successful for item in packet.get("web_sources", [])))],
        _web_key,
    )
    chat_ids = _unique_strings([
        *state.get("used_chat_ids", []),
        *(state.get("parallel_chat_id_deltas") or (item for packet in successful for item in packet.get("chat_ids", []))),
    ])
    memory_refs = _merge_unique_dicts(
        state.get("parallel_memory_ref_deltas") or [item for packet in successful for item in packet.get("memory_refs", [])],
        lambda item: str(item.get("memory_id") or item.get("id") or _stable_hash(item)),
    )
    memory_ids = _unique_strings([
        *state.get("used_memory_ids", []),
        *(item.get("memory_id") or item.get("id") for item in memory_refs),
    ])
    node_events = [*state.get("node_events", []), *state.get("parallel_node_event_deltas", [])]
    tool_events = [*state.get("tool_events", []), *state.get("parallel_tool_event_deltas", [])]
    errors = [*state.get("errors", []), *state.get("parallel_error_deltas", [])]
    skipped_nodes = [*state.get("skipped_nodes", [])]
    visits = [*state.get("node_visit_sequence", [])]
    if not state.get("parallel_node_event_deltas"):
        node_events.extend(item for packet in terminal for item in packet.get("node_events", []) if isinstance(item, dict))
    if not state.get("parallel_tool_event_deltas"):
        tool_events.extend(item for packet in terminal for item in packet.get("tool_events", []) if isinstance(item, dict))
    if not state.get("parallel_error_deltas"):
        errors.extend(item for packet in terminal for item in packet.get("errors", []) if isinstance(item, dict))
    for packet in terminal:
        if packet.get("status") == "skipped":
            skipped_nodes.append(str(packet.get("worker_node_id") or ""))
    visits.extend(item for item in state.get("parallel_visit_records", []) if isinstance(item, dict))
    if not state.get("parallel_visit_records"):
        for packet in terminal:
            visits.extend({
                "node_id": packet.get("worker_node_id"),
                "node": packet.get("worker_node_id"),
                "node_type": packet.get("worker_type"),
                "visit_index": int(packet.get("ordinal") or 0) + 1,
                "dispatch_id": packet.get("dispatch_id"),
                "work_id": packet.get("work_id"),
                "attempt": attempt,
            } for attempt in range(1, int(packet.get("attempt") or 1) + 1))
    node_events = _dedupe_dicts(
        node_events,
        lambda item: (
            str(item.get("work_id") or "serial"),
            int(item.get("attempt") or 1),
            str(item.get("name") or item.get("event") or item.get("event_name") or item.get("status") or ""),
            int(item.get("visit_index") or item.get("node_visit_index") or 1),
        ),
    )
    tool_events = _dedupe_dicts(
        tool_events,
        lambda item: (
            str(item.get("work_id") or "serial"),
            int(item.get("attempt") or 1),
            str(item.get("tool") or item.get("tool_name") or ""),
            int(item.get("invocation_sequence") or item.get("sequence") or 1),
        ),
    )
    errors = _dedupe_dicts(
        errors,
        lambda item: (
            str(item.get("work_id") or "serial"),
            int(item.get("attempt") or 1),
            str(item.get("code") or item.get("error_code") or item.get("type") or "unknown").strip().casefold(),
        ),
    )
    visits = sorted(
        [item for item in visits if isinstance(item, dict)],
        key=lambda item: (
            str(item.get("dispatch_id") or "serial"),
            str(item.get("work_id") or ""),
            int(item.get("attempt") or 1),
            str(item.get("node") or ""),
        ),
    )
    counts: Dict[str, int] = dict(state.get("node_visit_counts") or {})
    for packet in terminal:
        node = str(packet.get("worker_node_id") or "")
        if node:
            counts[node] = counts.get(node, 0) + int(packet.get("attempt") or 1)

    planned = len([item for item in state.get("work_items", []) if isinstance(item, dict)])
    retried = sum(max(0, count - 1) for count in attempts_by_work.values())
    policy = normalized_parallel_policy(state.get("parallel_policy"))
    summary = {
        "dispatch_id": state.get("dispatch_id"),
        "planned": planned,
        "completed": len(successful),
        "skipped": len(skipped),
        "failed": len(failed),
        "timed_out": len(timed_out),
        "cancelled": len(cancelled),
        "retried": retried,
        "partial_evidence": bool(successful and (failed or timed_out or cancelled)),
        "elapsed_ms": round(
            max(0, int(time.time() * 1000) - int(state["dispatch_started_epoch_ms"]))
            if state.get("dispatch_started_epoch_ms")
            else max((float(item.get("elapsed_ms") or 0) for item in terminal), default=0.0),
            2,
        ),
        "no_workers_dispatched": planned == 0,
        "fan_out_width": planned,
        "peak_concurrency": _peak_concurrency(terminal, maximum=policy["max_concurrency"]),
        "evidence_packets_before_dedupe": len(existing_evidence) + len(list(worker_evidence_input)),
        "evidence_packets_after_dedupe": len(all_evidence),
        "document_sources_before_dedupe": len(state.get("document_sources", [])) + len(state.get("parallel_document_source_deltas") or [item for packet in successful for item in packet.get("document_sources", [])]),
        "document_sources_after_dedupe": len(document_sources),
        "web_sources_before_dedupe": len(state.get("web_sources", [])) + len(state.get("parallel_web_source_deltas") or [item for packet in successful for item in packet.get("web_sources", [])]),
        "web_sources_after_dedupe": len(web_sources),
    }
    all_skipped = planned > 0 and len(skipped) == planned
    if successful and (failed or timed_out or cancelled) and not policy["continue_on_partial_failure"]:
        raise RuntimeError("parallel_dispatch_partial_failure")
    if planned and len(successful) < policy["minimum_successes"] and not all_skipped:
        raise RuntimeError("parallel_dispatch_no_usable_results")
    return {
        "evidence": evidence_text,
        "evidence_packets": all_evidence,
        "document_sources": document_sources,
        "web_sources": web_sources,
        "used_chat_ids": chat_ids,
        "used_memory_ids": memory_ids,
        "node_events": node_events,
        "tool_events": tool_events,
        "errors": errors,
        "skipped_nodes": _unique_strings(skipped_nodes),
        "node_visit_counts": counts,
        "node_visit_sequence": visits,
        "parallel_summary": summary,
        "parallel_attempt_records": list(state.get("parallel_attempt_records") or []),
    }
