from __future__ import annotations

import base64
import json
from typing import Any, Dict, Mapping, Tuple

from app.agent_workflows.trace_sanitization import TRACE_REDACTED_VALUE, _is_sensitive_key


TRACE_DETAIL_SCALAR_LIMIT = 256 * 1024
TRACE_DETAIL_RUN_LIMIT = 5 * 1024 * 1024
TRACE_DETAIL_OMITTED_VALUE = "[omitted from trace]"
TRACE_DETAIL_STATE_OMIT_KEYS = {
    "agent_run_id",
    "checkpoint_thread_id",
    "node_events",
    "tool_events",
}


def _looks_binary(value: str) -> bool:
    if len(value) < 4096:
        return False
    compact = "".join(value.split())
    if len(compact) < 4096 or len(compact) % 4:
        return False
    try:
        decoded = base64.b64decode(compact[:8192], validate=True)
    except Exception:
        return False
    return bool(decoded) and sum(byte < 9 or 13 < byte < 32 for byte in decoded) > len(decoded) // 8


def sanitize_trace_detail(value: Any) -> Tuple[Any, Dict[str, Any]]:
    redacted: list[str] = []
    truncated: list[str] = []
    omitted: list[str] = []

    def walk(item: Any, path: str, key: Any = None) -> Any:
        if key is not None and _is_sensitive_key(key):
            redacted.append(path)
            return TRACE_REDACTED_VALUE
        if isinstance(item, bytes):
            omitted.append(path)
            return TRACE_DETAIL_OMITTED_VALUE
        if isinstance(item, str):
            if _looks_binary(item):
                omitted.append(path)
                return TRACE_DETAIL_OMITTED_VALUE
            if len(item) > TRACE_DETAIL_SCALAR_LIMIT:
                truncated.append(path)
                return item[:TRACE_DETAIL_SCALAR_LIMIT] + "\n[truncated]"
            return item
        if isinstance(item, Mapping):
            result: Dict[str, Any] = {}
            for raw_key, child in item.items():
                child_key = str(raw_key)
                child_path = f"{path}.{child_key}" if path else child_key
                if child_key in TRACE_DETAIL_STATE_OMIT_KEYS:
                    omitted.append(child_path)
                    continue
                result[child_key] = walk(child, child_path, child_key)
            return result
        if isinstance(item, (list, tuple)):
            return [walk(child, f"{path}[{index}]") for index, child in enumerate(item)]
        if item is None or isinstance(item, (bool, int, float)):
            return item
        try:
            return json.loads(json.dumps(item, default=str))
        except Exception:
            return str(item)

    sanitized = walk(value, "")
    metadata = {
        "redacted_fields": redacted,
        "truncated_fields": truncated,
        "omitted_fields": omitted,
        "truncated": bool(truncated),
    }
    return sanitized, metadata


def trace_detail_size(value: Any) -> int:
    return len(json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def state_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    added = {key: after[key] for key in after.keys() - before.keys()}
    removed = {key: before[key] for key in before.keys() - after.keys()}
    changed = {
        key: {"before": before[key], "after": after[key]}
        for key in before.keys() & after.keys()
        if before[key] != after[key]
    }
    return {"added": added, "changed": changed, "removed": removed}


def final_output_from_result(result: Any) -> Dict[str, Any]:
    source = result if isinstance(result, Mapping) else {}
    payload = {
        "answer": source.get("final_answer") or source.get("answer"),
        "clarification_options": source.get("clarification_options"),
        "route": source.get("route"),
        "route_reason": source.get("route_reason"),
        "reasoning": source.get("reasoning"),
        "reasoning_available": source.get("reasoning_available"),
        "reasoning_format": source.get("reasoning_format"),
        "memory_candidate_ids": source.get("memory_candidate_ids"),
        "memory_candidates": source.get("memory_candidates"),
    }
    compact = {key: value for key, value in payload.items() if value not in (None, "", [], {})}
    if not compact:
        return {}
    sanitized, metadata = sanitize_trace_detail(compact)
    return {**sanitized, "safety": metadata}


def detail_manifest(details: Any) -> list[Dict[str, Any]]:
    rows = details if isinstance(details, list) else []
    manifest = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        manifest.append(
            {
                "node_id": row.get("node_id"),
                "node_type": row.get("node_type"),
                "visit_index": row.get("visit_index"),
                "status": row.get("status"),
                "available": True,
                "size_bytes": trace_detail_size(row),
                "truncated": bool((row.get("safety") or {}).get("truncated")),
            }
        )
    return manifest
