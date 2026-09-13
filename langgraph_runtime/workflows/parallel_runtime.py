from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from langgraph.types import Send, TimeoutPolicy
from langgraph.errors import NodeTimeoutError

from langgraph_runtime.workflows.enums import WorkflowNodeType
from langgraph_runtime.workflows.corrective_contracts import (
    CORRECTIVE_SOURCE_STRATEGIES,
    CORRECTIVE_SOURCE_STRATEGY_RANK,
    CORRECTIVE_WORKFLOW_ID,
    corrective_memory_recall_allowed,
    corrective_source_strategy,
    normalized_corrective_policy,
    stable_corrective_identity,
)
from langgraph_runtime.workflows.evidence import (
    canonical_source_id,
    combine_evidence,
    normalized_canonical_source_id,
    normalized_source_url,
    packet_source_ids,
)
from langgraph_runtime.workflows.parallel_contracts import (
    DEFAULT_PARALLEL_POLICY,
    PARALLEL_EVENT_NAMES,
    PARALLEL_AUTHORIZED_WORKFLOW_IDS,
    PARALLEL_TERMINAL_WORKER_STATUSES,
    PARALLEL_WORKER_EVIDENCE_KINDS,
    ParallelEventName,
    normalized_parallel_policy,
    parallel_timeout_watchdog_seconds,
)
from langgraph_runtime.models.retry import is_retryable_model_error


WORKER_EVIDENCE_KIND = PARALLEL_WORKER_EVIDENCE_KINDS
TERMINAL_WORKER_STATUSES = PARALLEL_TERMINAL_WORKER_STATUSES
CORRECTIVE_PROVENANCE_FIELDS = (
    "dispatch_id", "work_id", "query_id", "work_ordinal", "wave_id",
    "retrieval_query", "source_strategy", "source_scope", "source_expansion",
)


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
    return state.get("workflow_id") in PARALLEL_AUTHORIZED_WORKFLOW_IDS


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _normalized_query(value: Any, *, fallback: str) -> str:
    text = " ".join(str(value or fallback or "").split()).strip()
    return text[:2_000]


def dispatch_started_epoch_ms(state: Mapping[str, Any], dispatch_id: Any) -> int | None:
    """Recover the persisted dispatch start without inventing a zero-latency start."""

    candidates: List[int] = []
    direct = state.get("dispatch_started_epoch_ms")
    if isinstance(direct, (int, float)) and int(direct) > 0:
        candidates.append(int(direct))
    wanted = str(dispatch_id or "")
    for record in state.get("corrective_wave_records") or []:
        if not isinstance(record, Mapping) or wanted and str(record.get("dispatch_id") or "") != wanted:
            continue
        try:
            value = datetime.fromisoformat(str(record.get("started_at") or "").replace("Z", "+00:00"))
            candidates.append(int(value.timestamp() * 1000))
        except (TypeError, ValueError):
            pass
    for collection in (state.get("work_items") or [], state.get("worker_result_packets") or []):
        for item in collection:
            if not isinstance(item, Mapping) or wanted and str(item.get("dispatch_id") or "") != wanted:
                continue
            raw = item.get("dispatch_started_epoch_ms")
            if isinstance(raw, (int, float)) and int(raw) > 0:
                candidates.append(int(raw))
            try:
                value = datetime.fromisoformat(str(item.get("started_at") or "").replace("Z", "+00:00"))
                candidates.append(int(value.timestamp() * 1000))
            except (TypeError, ValueError):
                pass
    return min(candidates) if candidates else None


def _proposal_worker_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("worker_node_id") or value.get("node") or value.get("worker") or value.get("id") or "")
    return ""


def policy_filtered_memory_proposals(proposals: Any, state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if state.get("workflow_id") != CORRECTIVE_WORKFLOW_ID or corrective_memory_recall_allowed(state):
        return []
    available = state.get("available_worker_nodes") if isinstance(state.get("available_worker_nodes"), list) else []
    memory_ids = {
        str(item.get("id"))
        for item in available
        if isinstance(item, Mapping) and item.get("type") == WorkflowNodeType.DURABLE_MEMORY_WORKER.value
    }
    memory_ids.add(WorkflowNodeType.DURABLE_MEMORY_WORKER.value)
    result: List[Dict[str, Any]] = []
    for ordinal, raw in enumerate(proposals if isinstance(proposals, list) else []):
        if not isinstance(raw, Mapping) or _proposal_worker_id(raw) not in memory_ids:
            continue
        result.append({
            "proposal_id": stable_corrective_identity(
                "memory-policy-filter",
                run=state.get("agent_run_id"),
                wave=state.get("corrective_wave", 0),
                ordinal=ordinal,
                query=_normalized_query(raw.get("query"), fallback=str(state.get("question") or "")).casefold(),
            ),
            "wave_id": max(0, int(state.get("corrective_wave") or 0)),
            "worker_node_id": _proposal_worker_id(raw),
            "reason": "no_policy_readable_memory_scope",
        })
    return result


def normalize_work_items(
    proposals: Any,
    *,
    state: Mapping[str, Any],
    dispatch_node_id: str,
    dispatch_visit: int,
) -> List[Dict[str, Any]]:
    policy = normalized_parallel_policy(state.get("parallel_policy"))
    corrective_policy = normalized_corrective_policy(state.get("corrective_policy"))
    is_corrective = state.get("workflow_id") == CORRECTIVE_WORKFLOW_ID
    prior_work_ids = {
        str(item.get("work_id")) for item in state.get("worker_result_packets", [])
        if isinstance(item, Mapping) and item.get("work_id")
    }
    remaining_work = max(0, corrective_policy["max_total_work_items"] - len(prior_work_ids)) if is_corrective else policy["max_work_items"]
    prior_attempts = sum(
        1 for item in state.get("parallel_attempt_records", [])
        if isinstance(item, Mapping)
    )
    remaining_attempts = max(0, corrective_policy["max_total_tool_attempts"] - prior_attempts) if is_corrective else policy["max_work_items"] * policy["max_attempts"]
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
    allowed_file_hashes = {
        str(item.get("file_hash"))
        for item in ((state.get("pre_fetch_bundle") or {}).get("documents") or [])
        if isinstance(item, Mapping) and item.get("file_hash")
    }
    wave = max(0, int(state.get("corrective_wave") or 0))
    dispatch_fields = {
        "run": state.get("agent_run_id"),
        "node": dispatch_node_id,
        "visit": dispatch_visit,
        "wave": wave,
    }
    dispatch_id = (
        stable_corrective_identity("dispatch", **dispatch_fields)
        if is_corrective else _stable_hash(dispatch_fields)
    )
    candidates: List[Dict[str, Any]] = []
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
        file_hash = str(raw.get("file_hash") or "")[:256] if isinstance(raw, dict) else ""
        if is_corrective and file_hash and any(token in file_hash for token in ("/", "\\", "://")):
            continue
        if file_hash and (worker_type != WorkflowNodeType.RETRIEVAL_WORKER.value or file_hash not in allowed_file_hashes):
            continue
        key = (worker_type, f"{file_hash}|{query.casefold()}")
        if not query or key in seen:
            continue
        if worker_type == WorkflowNodeType.WEB_WORKER.value and not state.get("use_web_search", False):
            continue
        if is_corrective and worker_type == WorkflowNodeType.WEB_WORKER.value and not corrective_policy["allow_web_fallback"]:
            continue
        if (
            is_corrective
            and worker_type == WorkflowNodeType.DURABLE_MEMORY_WORKER.value
            and not corrective_memory_recall_allowed(state)
        ):
            continue
        source_strategy = corrective_source_strategy(worker_type, file_hash=file_hash)
        proposed_strategy = str(raw.get("strategy") or "") if isinstance(raw, dict) else ""
        if is_corrective and (
            not source_strategy
            or proposed_strategy and (
                proposed_strategy not in CORRECTIVE_SOURCE_STRATEGIES
                or proposed_strategy != source_strategy
            )
        ):
            continue
        seen.add(key)
        candidates.append({
            "worker_node_id": worker_id,
            "worker_type": worker_type,
            "operation_label": str(worker.get("label") or worker_id),
            "query": query,
            "tool_name": str(raw.get("tool_name") or "").strip() if isinstance(raw, dict) else "",
            "file_hash": file_hash or None,
            "source_strategy": source_strategy if is_corrective else proposed_strategy,
            "source_scope": file_hash or source_strategy or worker_type,
            "reason": str(raw.get("reason") or "")[:500] if isinstance(raw, dict) else "",
        })
    if is_corrective:
        candidates.sort(key=lambda item: (
            CORRECTIVE_SOURCE_STRATEGY_RANK[item["source_strategy"]],
            str(item.get("source_scope") or ""),
            str(item.get("worker_node_id") or ""),
            str(item.get("query") or "").casefold(),
        ))

    prior_strategy_ranks = [
        CORRECTIVE_SOURCE_STRATEGY_RANK[str(item.get("source_strategy"))]
        for item in state.get("worker_result_packets", [])
        if isinstance(item, Mapping) and str(item.get("source_strategy")) in CORRECTIVE_SOURCE_STRATEGY_RANK
    ]
    prior_max_rank = max(prior_strategy_ranks, default=-1)
    normalized: List[Dict[str, Any]] = []
    allocated_attempts = 0
    for candidate in candidates:
        ordinal = len(normalized)
        query = str(candidate["query"])
        worker_id = str(candidate["worker_node_id"])
        worker_type = str(candidate["worker_type"])
        source_scope = str(candidate["source_scope"])
        identity_fields = {
            "run": state.get("agent_run_id"),
            "node": dispatch_node_id,
            "wave": wave,
            "ordinal": ordinal,
            "worker": worker_id,
            "source_scope": source_scope,
            "query": query.casefold(),
        }
        query_id = stable_corrective_identity("query", **identity_fields) if is_corrective else _stable_hash(identity_fields)
        dedupe_key = query_id
        work_id = stable_corrective_identity("work", **identity_fields) if is_corrective else _stable_hash(identity_fields)
        timeout_ms = (
            policy["web_worker_timeout_ms"]
            if worker_type == WorkflowNodeType.WEB_WORKER.value
            else policy["default_worker_timeout_ms"]
        )
        item_max_attempts = min(policy["max_attempts"], max(0, remaining_attempts - allocated_attempts))
        if item_max_attempts < 1:
            break
        normalized.append({
            "dispatch_id": dispatch_id,
            "dispatch_visit": dispatch_visit,
            "parent_operation_id": dispatch_node_id,
            "work_id": work_id,
            "query_id": query_id,
            "ordinal": ordinal,
            "worker_node_id": worker_id,
            "worker_type": worker_type,
            "operation_id": worker_id,
            "operation_label": str(candidate.get("operation_label") or worker_id),
            "query": query,
            "tool_name": candidate.get("tool_name"),
            "file_hash": candidate.get("file_hash"),
            "wave_id": wave,
            "corrective_provenance": is_corrective,
            "reason": candidate["reason"],
            "source_strategy": candidate["source_strategy"],
            "source_scope": source_scope,
            "source_expansion": bool(
                is_corrective
                and wave > 0
                and CORRECTIVE_SOURCE_STRATEGY_RANK[candidate["source_strategy"]] > prior_max_rank
            ),
            "evidence_kind": WORKER_EVIDENCE_KIND[worker_type],
            "dedupe_key": dedupe_key,
            "attempt": 1,
            "max_attempts": item_max_attempts,
            "timeout_ms": timeout_ms,
        })
        allocated_attempts += item_max_attempts
        if len(normalized) >= min(policy["max_work_items"], remaining_work, remaining_attempts):
            break
    return normalized


def work_item_proposals(parsed: Mapping[str, Any], execution_plan: Sequence[str], question: str) -> List[Dict[str, Any]]:
    decisions = parsed.get("worker_decisions")
    if isinstance(decisions, list):
        proposals = [
            {
                "worker_node_id": item.get("worker_node_id"),
                "query": item.get("query") or question,
                "tool_name": item.get("tool_name"),
                "reason": item.get("reason") or "planner worker decision",
                "file_hash": item.get("file_hash"),
                "strategy": item.get("strategy"),
            }
            for item in decisions
            if isinstance(item, dict) and item.get("selected") is True
        ]
        proposed_worker_ids = {_proposal_worker_id(item) for item in proposals}
        proposals.extend(
            {"worker_node_id": worker_id, "query": question, "reason": "planner execution plan"}
            for worker_id in execution_plan
            if worker_id not in proposed_worker_ids
        )
        return proposals
    proposals = parsed.get("work_items")
    if isinstance(proposals, list):
        normalized = [item for item in proposals if isinstance(item, (dict, str))]
        proposed_worker_ids = {_proposal_worker_id(item) for item in normalized}
        normalized.extend(
            {"worker_node_id": worker_id, "query": question, "reason": "planner execution plan"}
            for worker_id in execution_plan
            if worker_id not in proposed_worker_ids
        )
        return normalized
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


def serial_dispatch_next(state: Mapping[str, Any]) -> Send | str:
    """Dispatch the next unfinished work item, preserving deterministic planner order."""

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
        aggregator = str(state.get("dispatch_aggregator_id") or state.get("parallel_aggregator_id") or "")
        return aggregator
    item = min(pending, key=_sort_key)
    return Send(
        str(item["worker_node_id"]),
        {
            **dict(state),
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


def _sort_key(packet: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(packet.get("dispatch_visit") or 1),
        int(packet.get("ordinal") or 0),
        str(packet.get("worker_node_id") or ""),
        str(packet.get("work_id") or ""),
        int(packet.get("attempt") or 1),
    )


def _content_hash(value: Any) -> str:
    normalized = " ".join(str(value or "").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sanitize_corrective_urls(value: Any) -> Any:
    if isinstance(value, list):
        return [_sanitize_corrective_urls(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _sanitize_corrective_urls(item) for key, item in value.items()}
    for key in ("url", "source_url", "link", "display_url"):
        if key in result:
            safe_url = normalized_source_url(result[key])
            result[key] = safe_url or ""
            if safe_url:
                result.setdefault("display_url", safe_url)
    return result


def _document_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    identity = item.get("document_id") or item.get("file_id") or item.get("file_hash")
    page = next((item.get(key) for key in ("page_number", "page", "page_start") if item.get(key) not in (None, "")), None)
    locator = next((item.get(key) for key in ("section_id", "chunk_id", "section") if item.get(key) not in (None, "")), None)
    if identity:
        return (str(identity), page, str(locator or ""))
    return (str(item.get("title") or item.get("file_name") or "").casefold(), page, _content_hash(item.get("content") or item.get("text")))


def _web_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    url = normalized_source_url(item.get("url") or item.get("source_url") or item.get("link"))
    if url:
        return (url,)
    return (str(item.get("title") or "").casefold(), _content_hash(item.get("content") or item.get("text") or item.get("snippet")))


def _merge_unique_dicts(items: Iterable[Any], key_fn, *, enrich_provenance: bool = False) -> List[Dict[str, Any]]:
    merged: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        key = key_fn(raw)
        if key not in merged:
            if not enrich_provenance:
                merged[key] = dict(raw)
                order.append(key)
                continue
            safe_raw = _sanitize_corrective_urls(dict(raw))
            source_id = canonical_source_id(safe_raw)
            merged[key] = {
                **safe_raw,
                **({"source_id": source_id} if source_id else {}),
                "provenance": [{
                    key: raw.get(key) for key in CORRECTIVE_PROVENANCE_FIELDS
                    if raw.get(key) not in (None, "")
                }],
            }
            order.append(key)
            continue
        for field, value in raw.items():
            if field not in merged[key] or merged[key][field] in (None, "", [], {}):
                merged[key][field] = value
        if not enrich_provenance:
            continue
        provenance = {
            field: raw.get(field) for field in CORRECTIVE_PROVENANCE_FIELDS
            if raw.get(field) not in (None, "")
        }
        if provenance and provenance not in merged[key]["provenance"]:
            merged[key]["provenance"].append(provenance)
    if not enrich_provenance:
        return [merged[key] for key in order]
    return sorted(
        [merged[key] for key in order],
        key=lambda item: (int(item.get("wave_id") or 0), int(item.get("work_ordinal") or 0), str(item.get("source_id") or "")),
    )


def _dedupe_evidence_first(items: Iterable[Any], *, enrich_provenance: bool = False) -> List[Dict[str, Any]]:
    if not enrich_provenance:
        result: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            refs = item.get("refs") if isinstance(item.get("refs"), dict) else {}
            primary_ref = (
                item.get("primary_reference_key")
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
    merged: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for item in sorted(
        [dict(value) for value in items if isinstance(value, dict)],
        key=lambda value: (int(value.get("wave_id") or 0), int(value.get("work_ordinal") or 0), str((value.get("source_ids") or [""])[0]), str(value.get("id") or "")),
    ):
        if not isinstance(item, dict):
            continue
        source_ids = sorted({
            normalized
            for source_id in (item.get("source_ids") or packet_source_ids(item))
            if (normalized := normalized_canonical_source_id(source_id))
        })
        item["source_ids"] = source_ids
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
        provenance = {
            field: item.get(field) for field in CORRECTIVE_PROVENANCE_FIELDS
            if item.get(field) not in (None, "")
        }
        if key in merged:
            merged[key]["source_ids"] = sorted(set(merged[key].get("source_ids") or []) | set(source_ids))
            if provenance and provenance not in merged[key]["provenance"]:
                merged[key]["provenance"].append(provenance)
            continue
        merged[key] = {**item, "provenance": [provenance] if provenance else []}
    return list(merged.values())


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
    enrich_provenance = bool(item.get("corrective_provenance"))
    chat_refs = (
        [
            {"message_id": value, "dispatch_id": dispatch_id, "work_id": work_id, "query_id": item.get("query_id"), "work_ordinal": item.get("ordinal"), "wave_id": item.get("wave_id", 0), "source_strategy": item.get("source_strategy")}
            for value in local.get("used_chat_ids") or []
            if value
        ]
        if enrich_provenance
        else list(local.get("used_chat_ids") or [])
    )
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
    def _provenance(values: Iterable[Any]) -> List[Dict[str, Any]]:
        if not enrich_provenance:
            return [dict(value) for value in values if isinstance(value, Mapping)]
        return [
            {
                **_sanitize_corrective_urls(dict(value)),
                "dispatch_id": dispatch_id,
                "work_id": work_id,
                "work_ordinal": item.get("ordinal"),
                "wave_id": item.get("wave_id", 0),
                "retrieval_query": item.get("query"),
                "query_id": item.get("query_id"),
                "source_strategy": item.get("source_strategy"),
                "source_scope": item.get("source_scope"),
                "source_expansion": item.get("source_expansion", False),
            }
            for value in values
            if isinstance(value, Mapping)
        ]

    evidence_packet_values = list(local.get("evidence_packets") or [])
    document_source_values = list(local.get("document_sources") or [])
    web_source_values = list(local.get("web_sources") or [])
    if enrich_provenance:
        evidence_packet_values = [
            value for value in evidence_packet_values
            if isinstance(value, Mapping)
            and len(value.get("source_ids") or []) == 1
        ]
    packet = {
        **dict(item),
        "attempt": attempt,
        "status": status,
        "evidence_packets": _provenance(evidence_packet_values),
        "document_sources": _provenance(document_source_values),
        "web_sources": _provenance(web_source_values),
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
        "parallel_chat_id_deltas": chat_refs,
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
    result = {**combined, "parallel_summary": summary}
    if state.get("workflow_id") == CORRECTIVE_WORKFLOW_ID:
        now = datetime.now(timezone.utc)
        started_ms = dispatch_started_epoch_ms(state, summary.get("dispatch_id"))
        record_id = stable_corrective_identity(
            "wave", run=state.get("agent_run_id"), dispatch=summary.get("dispatch_id"), wave=state.get("corrective_wave", 0)
        )
        result["corrective_wave_records"] = [{
            "record_id": record_id,
            "dispatch_id": summary.get("dispatch_id"),
            "wave_id": max(0, int(state.get("corrective_wave") or 0)),
            "started_at": (
                datetime.fromtimestamp(started_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                if started_ms is not None else None
            ),
            "completed_at": now.isoformat().replace("+00:00", "Z"),
            "elapsed_ms": max(0, int(now.timestamp() * 1000) - started_ms) if started_ms is not None else None,
            "latency_unavailable": started_ms is None,
            "outcome": "cancelled",
            "partial": False,
            **{key: summary.get(key, 0) for key in ("planned", "completed", "skipped", "failed", "timed_out", "cancelled")},
        }]
    return result


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
    current_dispatch_id = str(state.get("dispatch_id") or "")
    packets = sorted(
        [
            dict(item) for item in state.get("worker_result_packets", [])
            if isinstance(item, dict)
            and (not current_dispatch_id or str(item.get("dispatch_id") or "") == current_dispatch_id)
        ],
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

    def _current(values: Iterable[Any]) -> List[Any]:
        return [
            value for value in values
            if not isinstance(value, Mapping)
            or not current_dispatch_id
            or str(value.get("dispatch_id") or "") == current_dispatch_id
        ]

    worker_evidence_input = _current(state.get("parallel_evidence_deltas") or []) or [
        evidence for packet in successful for evidence in packet.get("evidence_packets", [])
    ]
    enrich_provenance = state.get("workflow_id") == CORRECTIVE_WORKFLOW_ID
    worker_evidence = _dedupe_evidence_first(worker_evidence_input, enrich_provenance=enrich_provenance)
    existing_evidence = [item for item in state.get("evidence_packets", []) if isinstance(item, dict)]
    all_evidence = _dedupe_evidence_first(
        [*existing_evidence, *worker_evidence],
        enrich_provenance=enrich_provenance,
    )
    evidence_text = str(state.get("evidence") or "")
    for packet in worker_evidence:
        evidence_text = combine_evidence(
            evidence_text,
            packet.get("content"),
            label=str(packet.get("kind") or "Worker evidence").replace("_", " ").title(),
        )

    document_sources = _merge_unique_dicts(
        [*state.get("document_sources", []), *(_current(state.get("parallel_document_source_deltas") or []) or (item for packet in successful for item in packet.get("document_sources", [])))],
        _document_key,
        enrich_provenance=enrich_provenance,
    )
    web_sources = _merge_unique_dicts(
        [*state.get("web_sources", []), *(_current(state.get("parallel_web_source_deltas") or []) or (item for packet in successful for item in packet.get("web_sources", [])))],
        _web_key,
        enrich_provenance=enrich_provenance,
    )
    current_chat_deltas = _current(state.get("parallel_chat_id_deltas") or [])
    chat_ids = _unique_strings([
        *state.get("used_chat_ids", []),
        *((item.get("message_id") if isinstance(item, Mapping) else item) for item in current_chat_deltas),
        *((item for packet in successful for item in packet.get("chat_ids", [])) if not current_chat_deltas else []),
    ])
    memory_refs = _merge_unique_dicts(
        _current(state.get("parallel_memory_ref_deltas") or []) or [item for packet in successful for item in packet.get("memory_refs", [])],
        lambda item: str(item.get("memory_id") or item.get("id") or _stable_hash(item)),
        enrich_provenance=enrich_provenance,
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
            str(item.get("node_id") or item.get("node") or item.get("worker_node_id") or ""),
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
    if planned and len(successful) < policy["minimum_successes"] and not all_skipped and not policy["continue_on_insufficient_successes"]:
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
        "dispatch_summary": {**summary, "mode": str(state.get("dispatch_mode") or "parallel")},
        "parallel_attempt_records": list(state.get("parallel_attempt_records") or []),
    }
