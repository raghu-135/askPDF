from __future__ import annotations

from typing import Any, Dict, Iterable, List


PREVIEW_LIMIT = 900
PROMPT_PREVIEW_LIMIT = 4000


def compact_preview(value: Any, *, limit: int = PREVIEW_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def normalize_warnings(*items: Any) -> List[str]:
    warnings: List[str] = []
    for item in items:
        if not item:
            continue
        values = item if isinstance(item, list) else [item]
        for value in values:
            text = str(value or "").strip()
            if text and text not in warnings:
                warnings.append(text)
    return warnings


def prompt_summary(section: str, system_message: str, prompt: str) -> Dict[str, Any]:
    return {
        "section": section,
        "system_message": compact_preview(system_message, limit=1200),
        "prompt_chars": len(prompt or ""),
        "preview": compact_preview(prompt, limit=PROMPT_PREVIEW_LIMIT),
    }


def refs_from_documents(items: Any) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return refs
    for item in items:
        if not isinstance(item, dict):
            continue
        ref = {
            key: item.get(key)
            for key in (
                "file_hash",
                "file_name",
                "chunk_id",
                "page_start",
                "page_end",
                "pages",
                "score",
                "rerank_score",
                "document_available_in_thread_at",
                "timeline_event_at",
                "timeline_event_type",
            )
            if item.get(key) not in (None, "")
        }
        preview = item.get("preview") or item.get("text") or item.get("excerpt")
        if preview:
            ref["preview"] = compact_preview(preview)
        if ref:
            refs.append(ref)
    return refs


def refs_from_messages(items: Any) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return refs
    for item in items:
        if isinstance(item, str):
            refs.append({"message_id": item})
            continue
        if not isinstance(item, dict):
            continue
        ref = {
            key: item.get(key)
            for key in (
                "message_id",
                "memory_message_id",
                "turn_id",
                "role",
                "created_at",
                "message_created_at",
                "score",
                "rerank_score",
            )
            if item.get(key) not in (None, "")
        }
        preview = item.get("preview") or item.get("text") or item.get("excerpt")
        if preview:
            ref["preview"] = compact_preview(preview)
        if ref:
            refs.append(ref)
    return refs


def refs_from_timeline(items: Any) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return refs
    for item in items:
        if not isinstance(item, dict):
            continue
        ref = {
            key: item.get(key)
            for key in (
                "source_type",
                "timeline_event_at",
                "timeline_event_type",
                "message_id",
                "file_hash",
                "file_name",
                "url",
                "title",
                "search_query",
                "score",
            )
            if item.get(key) not in (None, "")
        }
        preview = item.get("preview") or item.get("excerpt") or item.get("text")
        if preview:
            ref["preview"] = compact_preview(preview)
        if ref:
            refs.append(ref)
    return refs


def refs_from_web(items: Any) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return refs
    for item in items:
        if not isinstance(item, dict):
            continue
        ref = {
            key: item.get(key)
            for key in (
                "url",
                "title",
                "score",
                "search_query",
                "web_search_performed_at",
                "timeline_event_at",
                "timeline_event_type",
            )
            if item.get(key) not in (None, "")
        }
        preview = item.get("preview") or item.get("text") or item.get("excerpt")
        if preview:
            ref["preview"] = compact_preview(preview)
        if ref:
            refs.append(ref)
    return refs


def available_document_refs(items: Any) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return refs
    for item in items:
        if not isinstance(item, dict):
            continue
        ref = {
            key: item.get(key)
            for key in (
                "index",
                "file_hash",
                "file_name",
                "source_type",
                "document_available_in_thread_at",
                "chunk_count",
                "page_count",
                "word_count",
                "sentence_count",
                "filetype",
            )
            if item.get(key) not in (None, "")
        }
        if ref:
            refs.append(ref)
    return refs


def refs_from_artifacts(artifacts: Any) -> Dict[str, Any]:
    data = artifacts if isinstance(artifacts, dict) else {}
    refs: Dict[str, Any] = {}
    documents = refs_from_documents(data.get("document_sources"))
    web = refs_from_web(data.get("web_sources"))
    messages = refs_from_messages(data.get("used_chat_ids"))
    timeline = refs_from_timeline(data.get("timeline_events"))
    if documents:
        refs["document_matches"] = documents
    if web:
        refs["web_sources"] = web
    if messages:
        refs["messages"] = messages
    if timeline:
        refs["timeline_events"] = timeline
    return refs


def artifact_summary(artifacts: Any) -> Dict[str, int]:
    data = artifacts if isinstance(artifacts, dict) else {}
    return {
        key: len(value)
        for key, value in {
            "document_sources": data.get("document_sources"),
            "web_sources": data.get("web_sources"),
            "used_chat_ids": data.get("used_chat_ids"),
            "timeline_events": data.get("timeline_events"),
        }.items()
        if isinstance(value, list)
    }


def compact_refs(refs: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in refs.items() if value}


def selected_and_skipped_workers(plan: Iterable[str], all_workers: Iterable[str]) -> Dict[str, List[str]]:
    selected = [item for item in all_workers if item in set(plan)]
    skipped = [item for item in all_workers if item not in set(plan)]
    return {"selected_workers": selected, "skipped_workers": skipped}
