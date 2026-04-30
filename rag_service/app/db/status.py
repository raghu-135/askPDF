"""
status.py - File status normalization and management helpers.

This module provides utilities for normalizing and manipulating file processing status,
including parsing and indexing status across multiple embedding models and threads.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.db.models_sqlmodel import ProcessStatus


def _parse_settings(raw: Optional[str]) -> Dict[str, Any]:
    """Parse JSON settings string, returning empty dict on failure."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _parse_json_list(raw: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Deserialize a JSON-encoded list from a database text column."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return None
    except Exception:
        return None


def _default_process_section(status: str = ProcessStatus.UNKNOWN.value) -> Dict[str, Any]:
    """Return the default shape for a process-status section."""
    return {"status": status}


def _copy_process_section(section: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a shallow copy of a process-status section with a default status."""
    copied = dict(section or {})
    copied.setdefault("status", ProcessStatus.UNKNOWN.value)
    return copied


def _collapse_process_sections(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collapse many process-status sections into a single summary.

    Priority reflects the most actionable state for the UI:
    failed > running > pending > completed > unknown
    """
    if not sections:
        return _default_process_section()

    copied = [_copy_process_section(section) for section in sections]
    statuses = [section.get("status", ProcessStatus.UNKNOWN.value) for section in copied]

    if any(status == ProcessStatus.FAILED.value for status in statuses):
        status = ProcessStatus.FAILED.value
    elif any(status == ProcessStatus.RUNNING.value for status in statuses):
        status = ProcessStatus.RUNNING.value
    elif any(status == ProcessStatus.PENDING.value for status in statuses):
        status = ProcessStatus.PENDING.value
    elif all(status == ProcessStatus.COMPLETED.value for status in statuses):
        status = ProcessStatus.COMPLETED.value
    else:
        status = ProcessStatus.UNKNOWN.value

    summary: Dict[str, Any] = {"status": status}

    started_candidates = [
        section.get("started_at")
        for section in copied
        if section.get("started_at")
    ]
    finished_candidates = [
        section.get("finished_at")
        for section in copied
        if section.get("finished_at")
    ]
    if started_candidates:
        summary["started_at"] = min(started_candidates)
    if finished_candidates and all(
        section.get("status") == ProcessStatus.COMPLETED.value for section in copied
    ):
        summary["finished_at"] = max(finished_candidates)

    errors = [section.get("error") for section in copied if section.get("error")]
    if errors:
        summary["error"] = errors[-1]

    chunk_counts = [
        int(section.get("chunk_count", 0) or 0)
        for section in copied
        if section.get("chunk_count") is not None
    ]
    total_chars = [
        int(section.get("total_chars", 0) or 0)
        for section in copied
        if section.get("total_chars") is not None
    ]
    if chunk_counts:
        summary["chunk_count"] = max(chunk_counts)
    if total_chars:
        summary["total_chars"] = max(total_chars)

    return summary


def _normalize_file_status(status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize legacy and current file_status payloads into a single shape."""
    raw = dict(status or {})

    parsing = _copy_process_section(
        raw.get("parsing_status") if isinstance(raw.get("parsing_status"), dict) else raw.get("parsing")
    )

    raw_indexing_status = raw.get("indexing_status")
    if isinstance(raw_indexing_status, dict):
        raw_models = raw_indexing_status.get("models")
        raw_summary = raw_indexing_status.get("summary")
    else:
        raw_models = None
        raw_summary = None

    models: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_models, dict):
        for model_name, model_entry in raw_models.items():
            if not isinstance(model_entry, dict):
                continue
            normalized_model = _copy_process_section(model_entry)
            threads = model_entry.get("threads", {})
            normalized_threads: Dict[str, Dict[str, Any]] = {}
            if isinstance(threads, dict):
                for thread_id, thread_entry in threads.items():
                    if isinstance(thread_entry, dict):
                        normalized_threads[thread_id] = _copy_process_section(thread_entry)
            normalized_model["threads"] = normalized_threads
            models[model_name] = normalized_model

    summary = _copy_process_section(raw_summary if isinstance(raw_summary, dict) else raw.get("indexing"))
    if models:
        summary = _collapse_process_sections(list(models.values()))

    return {
        **raw,
        "parsing": parsing,
        "parsing_status": parsing,
        "indexing": summary,
        "indexing_status": {
            "summary": summary,
            "models": models,
        },
        "updated_at": raw.get("updated_at") or datetime.utcnow().isoformat(),
    }


def get_scoped_indexing_status(
    status: Optional[Dict[str, Any]],
    embedding_model: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Select the summary, model, or thread-specific indexing section."""
    normalized = _normalize_file_status(status)
    indexing_status = normalized.get("indexing_status", {})
    models = indexing_status.get("models", {})

    if embedding_model:
        model_status = _copy_process_section(models.get(embedding_model))
        if thread_id:
            return _copy_process_section(model_status.get("threads", {}).get(thread_id))
        return model_status

    if thread_id:
        thread_sections = []
        for model_status in models.values():
            if not isinstance(model_status, dict):
                continue
            threads = model_status.get("threads", {})
            if thread_id in threads and isinstance(threads[thread_id], dict):
                thread_sections.append(threads[thread_id])
        return _collapse_process_sections(thread_sections)

    return _copy_process_section(indexing_status.get("summary"))
