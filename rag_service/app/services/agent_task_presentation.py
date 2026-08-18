from __future__ import annotations

import hashlib
from typing import Any, Optional
from urllib.parse import urlparse


def timeline_sources(
    artifacts: list[Any],
    *,
    attempts_by_run: Optional[dict[str, int]] = None,
    selected_run_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Project authorized artifact provenance into stable, origin-aware sources."""
    attempts = attempts_by_run or {}
    ordered = sorted(
        artifacts,
        key=lambda artifact: (
            0 if selected_run_id and str(getattr(artifact, "agent_run_id", "")) == selected_run_id else 1,
            -attempts.get(str(getattr(artifact, "agent_run_id", "")), 0),
            str(getattr(artifact, "created_at", "")),
            str(getattr(artifact, "id", "")),
        ),
    )
    projected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for artifact in ordered:
        refs = dict(artifact.source_refs_json or {})
        candidates = [value for value in refs.get("sources") or [] if isinstance(value, dict)]
        for tool in refs.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            candidates.extend(value for value in tool.get("sources") or [] if isinstance(value, dict))
            tool_artifacts = tool.get("artifacts") if isinstance(tool.get("artifacts"), dict) else {}
            for key in ("web_sources", "document_sources", "memory_sources", "thread_sources"):
                candidates.extend(value for value in tool_artifacts.get(key) or [] if isinstance(value, dict))

        origin_run_id = str(getattr(artifact, "agent_run_id", None) or "")
        origin_attempt = attempts.get(origin_run_id, 0)
        plan_revision = int((getattr(artifact, "provenance_json", None) or {}).get("plan_revision") or 0)
        artifact_id = str(getattr(artifact, "id", "") or "")
        origin = {
            "run_id": origin_run_id,
            "attempt": origin_attempt,
            "artifact_id": artifact_id,
            "plan_revision": plan_revision,
            "inherited": bool(selected_run_id and origin_run_id != selected_run_id),
        }
        for value in candidates:
            url = str(value.get("url") or "").strip()[:4_000]
            if url and urlparse(url).scheme.lower() not in {"http", "https"}:
                url = ""
            file_hash = str(value.get("file_hash") or value.get("document_id") or "").strip()
            memory_id = str(value.get("memory_id") or "").strip()
            thread_ref = str(value.get("chat_id") or value.get("timeline_event_id") or "").strip()
            kind = "web" if url else "document" if file_hash else "memory" if memory_id else "thread" if thread_ref else ""
            if not kind:
                continue
            page = value.get("page", value.get("page_number"))
            reference = url or file_hash or memory_id or thread_ref
            identity = (kind, reference, str(page) if page is not None else "")
            existing = projected.get(identity)
            if existing is not None:
                if origin not in existing["origins"]:
                    existing["origins"].append(origin)
                continue
            if len(projected) >= limit:
                continue
            title = str(value.get("title") or value.get("filename") or value.get("file_name") or "").strip()[:1_000]
            snippet = str(value.get("snippet") or value.get("text") or value.get("content") or "").strip()[:4_000]
            source_id = hashlib.sha256("\x1f".join(identity).encode("utf-8")).hexdigest()[:24]
            projected[identity] = {
                "id": source_id,
                "kind": kind,
                **({"title": title} if title else {}),
                **({"url": url} if url else {}),
                **({"snippet": snippet} if snippet else {}),
                **({"file_hash": file_hash} if file_hash else {}),
                **({"page": page} if page is not None else {}),
                **({"memory_id": memory_id} if memory_id else {}),
                "artifact_id": artifact_id,
                "origin_run_id": origin_run_id,
                "origin_attempt": origin_attempt,
                "plan_revision": plan_revision,
                "inherited": origin["inherited"],
                "origins": [origin],
            }
    return list(projected.values())


def plan_diff(previous: dict[str, Any], current: dict[str, Any], *, reason: str) -> dict[str, Any]:
    old = {str(value.get("id")): value for value in previous.get("todos") or [] if isinstance(value, dict) and value.get("id")}
    new = {str(value.get("id")): value for value in current.get("todos") or [] if isinstance(value, dict) and value.get("id")}
    fields = ("title", "description", "completion_criteria", "dependency_ids", "priority", "required", "profile_id")
    changed = [
        {"id": todo_id, "fields": names, "title": new[todo_id].get("title")}
        for todo_id in sorted(old.keys() & new.keys())
        if (names := [field for field in fields if old[todo_id].get(field) != new[todo_id].get(field)])
    ]
    old_order = [str(value.get("id")) for value in previous.get("todos") or [] if isinstance(value, dict)]
    new_order = [str(value.get("id")) for value in current.get("todos") or [] if isinstance(value, dict)]
    return {
        "reason": reason,
        "added": [{"id": key, "title": new[key].get("title")} for key in sorted(new.keys() - old.keys())],
        "removed": [{"id": key, "title": old[key].get("title")} for key in sorted(old.keys() - new.keys())],
        "changed": changed,
        "reordered": old_order != new_order,
    }
