"""Shared deduplication helpers for document/web source rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping

from langgraph_runtime.workflows.corrective_contracts import CORRECTIVE_PROVENANCE_FIELDS
from langgraph_runtime.workflows.evidence import (
    canonical_source_id,
    normalized_source_url,
)


def _content_hash(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        payload = value
    else:
        try:
            payload = json.dumps(value, sort_keys=True, default=str)
        except Exception:
            payload = str(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def document_source_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    identity = item.get("document_id") or item.get("file_id") or item.get("file_hash")
    page = next(
        (item.get(key) for key in ("page_number", "page", "page_start") if item.get(key) not in (None, "")),
        None,
    )
    locator = next(
        (item.get(key) for key in ("section_id", "chunk_id", "section") if item.get(key) not in (None, "")),
        None,
    )
    if identity:
        return (str(identity), page, str(locator or ""))
    return (
        str(item.get("title") or item.get("file_name") or "").casefold(),
        page,
        _content_hash(item.get("content") or item.get("text")),
    )


def web_source_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    url = normalized_source_url(item.get("url") or item.get("source_url") or item.get("link"))
    if url:
        return (url,)
    return (
        str(item.get("title") or "").casefold(),
        _content_hash(item.get("content") or item.get("text") or item.get("snippet")),
    )


def merge_unique_source_dicts(
    items: Iterable[Any],
    key_fn,
    *,
    enrich_provenance: bool = False,
) -> List[Dict[str, Any]]:
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
            safe_raw = dict(raw)
            for field_key in ("url", "source_url", "link"):
                safe_url = normalized_source_url(safe_raw.get(field_key))
                if safe_url:
                    safe_raw[field_key] = safe_url
            source_id = canonical_source_id(safe_raw)
            merged[key] = {
                **safe_raw,
                **({"source_id": source_id} if source_id else {}),
                "provenance": [
                    {
                        field: raw.get(field)
                        for field in CORRECTIVE_PROVENANCE_FIELDS
                        if raw.get(field) not in (None, "")
                    }
                ],
            }
            order.append(key)
            continue
        for field, value in raw.items():
            if field not in merged[key] or merged[key][field] in (None, "", [], {}):
                merged[key][field] = value
        if not enrich_provenance:
            continue
        provenance = {
            field: raw.get(field)
            for field in CORRECTIVE_PROVENANCE_FIELDS
            if raw.get(field) not in (None, "")
        }
        if provenance and provenance not in merged[key]["provenance"]:
            merged[key]["provenance"].append(provenance)
    if not enrich_provenance:
        return [merged[key] for key in order]
    return sorted(
        [merged[key] for key in order],
        key=lambda item: (
            int(item.get("wave_id") or 0),
            int(item.get("work_ordinal") or 0),
            str(item.get("source_id") or ""),
        ),
    )


def unique_strings(items: Iterable[Any]) -> List[str]:
    values: List[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, Mapping):
            value = str(item.get("message_id") or item.get("id") or item.get("chat_id") or "")
        else:
            value = str(item or "")
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values