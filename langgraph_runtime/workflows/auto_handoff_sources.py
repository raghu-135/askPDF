"""Merge Auto prefetch sources with delegated child workflow results."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from langgraph_runtime.workflows.source_dedupe import (
    document_source_key,
    merge_unique_source_dicts,
    unique_strings,
    web_source_key,
)


def extract_delegate_child_result(delegate_payload: Any) -> Dict[str, Any]:
    if not isinstance(delegate_payload, Mapping):
        return {}
    nested = delegate_payload.get("result")
    if isinstance(nested, Mapping):
        return dict(nested)
    return dict(delegate_payload)


def _list_dict(container: Any) -> list[Dict[str, Any]]:
    if not isinstance(container, list):
        return []
    return [item for item in container if isinstance(item, dict)]


def _document_identity(item: Mapping[str, Any]) -> str:
    for key in ("file_hash", "file_id", "document_id"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _has_page_number(item: Mapping[str, Any]) -> bool:
    for key in ("page_number", "page", "page_start"):
        value = item.get(key)
        if value not in (None, ""):
            return True
    return False


def _filter_parent_documents_for_child(
    parent_rows: list[Dict[str, Any]],
    child_rows: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    child_identities = {_document_identity(item) for item in child_rows if _document_identity(item)}
    if not child_identities:
        return parent_rows
    filtered: list[Dict[str, Any]] = []
    for item in parent_rows:
        identity = _document_identity(item)
        if identity and identity in child_identities and not _has_page_number(item):
            continue
        filtered.append(item)
    return filtered


def merge_auto_handoff_sources(
    state: Mapping[str, Any],
    *,
    child_payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return merged document/web sources and used chat ids for Auto handoff."""

    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), dict) else {}
    child = dict(child_payload or extract_delegate_child_result(state.get("delegate_child_output")))

    child_document_rows = _list_dict(child.get("document_sources"))
    parent_document_rows = _filter_parent_documents_for_child(
        [
            *_list_dict(state.get("document_sources")),
            *_list_dict(bundle.get("document_sources")),
            *_list_dict(bundle.get("documents")),
        ],
        child_document_rows,
    )
    document_rows = merge_unique_source_dicts(
        [
            *parent_document_rows,
            *child_document_rows,
        ],
        document_source_key,
        prefer_last=True,
    )
    web_rows = merge_unique_source_dicts(
        [
            *_list_dict(state.get("web_sources")),
            *_list_dict(bundle.get("web_sources")),
            *_list_dict(child.get("web_sources")),
        ],
        web_source_key,
        prefer_last=True,
    )
    used_chat_ids = unique_strings(
        [
            *(state.get("used_chat_ids") or []),
            *(bundle.get("used_chat_ids") or []),
            *(child.get("used_chat_ids") or []),
        ]
    )
    return {
        "document_sources": document_rows,
        "web_sources": web_rows,
        "used_chat_ids": used_chat_ids,
    }