from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from app.services.agent_grounding_evaluator import MANIFEST_EVIDENCE_TOOLS

_RUNTIME_REPORT_SECTION_PATTERN = re.compile(r"\n##\s+Unresolved required work[\s\S]*$", re.IGNORECASE)
_CANVAS_PUBLICATION_CLAIM_PATTERN = re.compile(
    r"(?im)^.*\b("
    r"canvas draft|draft (?:research )?canvas|canvas was drafted|"
    r"published (?:a |the )?(?:research )?canvas|"
    r"research canvas (?:was|has been) (?:drafted|published|created)"
    r").*$\n?",
)


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


def timeline_run_error(run: Any) -> dict[str, Any]:
    stored = dict(getattr(run, "error_json", None) or {})
    if stored.get("code") or stored.get("safe_message") or stored.get("message"):
        return stored
    debug = getattr(run, "debug_trace_json", None)
    summary = ((debug or {}).get("diagnostics") or {}).get("summary") if isinstance(debug, dict) else {}
    if isinstance(summary, dict) and (summary.get("code") or summary.get("message")):
        return {
            "code": summary.get("code"),
            "safe_message": summary.get("message"),
            "retryable": summary.get("retryable"),
            "details": dict(summary.get("location") or {}),
        }
    return stored


_TIMELINE_TYPE_ORDER = {
    "objective": 0,
    "research_guidance": 10,
    "plan": 20,
    "replan": 25,
    "todo_result": 30,
    "todo_failure": 31,
    "approval": 35,
    "research_report": 40,
    "research_canvas": 50,
    "run_failure": 60,
}


def timeline_sort_key(item: Mapping[str, Any]) -> tuple[str, int, str]:
    timestamp = str(item.get("timestamp") or "")
    item_type = str(item.get("type") or "")
    order = int(item.get("sort_order") or _TIMELINE_TYPE_ORDER.get(item_type, 45))
    return (timestamp, order, str(item.get("id") or ""))


def _strip_runtime_report_appendix(markdown: str) -> str:
    return _RUNTIME_REPORT_SECTION_PATTERN.sub("", str(markdown or "")).rstrip()


def _honest_canvas_narration(markdown: str, *, has_valid_canvas: bool) -> str:
    text = str(markdown or "").strip()
    if not text or has_valid_canvas:
        return text
    if not _CANVAS_PUBLICATION_CLAIM_PATTERN.search(text):
        return text
    text = _CANVAS_PUBLICATION_CLAIM_PATTERN.sub("", text).rstrip()
    note = "_No research canvas was published for this run._"
    if note.casefold() not in text.casefold():
        text = f"{text}\n\n{note}" if text else note
    return text.strip()


def followup_guidance_entries(task_config: Mapping[str, Any] | None, run: Any) -> list[dict[str, str]]:
    config = dict(task_config or {})
    run_id = str(getattr(run, "id", "") or "")
    parent_run_id = str(getattr(run, "parent_run_id", "") or "")
    run_metadata = dict(getattr(run, "run_metadata_json", None) or {})
    metadata_source_run_id = str(run_metadata.get("result_review_source_run_id") or "")
    metadata_idempotency_key = str(run_metadata.get("result_review_idempotency_key") or "")
    entries: list[dict[str, str]] = []
    for entry in config.get("result_review_context") or []:
        if not isinstance(entry, Mapping):
            continue
        source_run_id = str(entry.get("source_run_id") or "")
        followup = str(entry.get("followup_input") or "").strip()
        if not followup:
            continue
        entry_key = str(entry.get("idempotency_key") or source_run_id)
        matched = False
        if parent_run_id and source_run_id == parent_run_id:
            matched = True
        elif metadata_source_run_id and source_run_id == metadata_source_run_id:
            matched = True
        elif metadata_idempotency_key and entry_key == metadata_idempotency_key:
            matched = True
        elif not parent_run_id and source_run_id == run_id:
            matched = True
        if matched:
            entries.append({"followup": followup, "key": entry_key})
    return entries


async def timeline_report_markdown(
    run: Any,
    artifacts: list[Any],
    *,
    research_canvas_kind: str = "research_canvas",
) -> str | None:
    from app.models.canvas import RESEARCH_CANVAS_ARTIFACT_KIND

    run_id = str(getattr(run, "id", "") or "")
    has_valid_canvas = any(
        str(getattr(artifact, "agent_run_id", "")) == run_id
        and str(getattr(artifact, "kind", "")) in {research_canvas_kind, RESEARCH_CANVAS_ARTIFACT_KIND}
        and str(getattr(artifact, "validity", "")) == "valid"
        for artifact in artifacts
    )
    pending = dict(getattr(run, "pending_interrupt_json", None) or {})
    provisional = _strip_runtime_report_appendix(str(pending.get("provisional_answer") or "").strip())
    final_report = next(
        (
            artifact for artifact in artifacts
            if str(getattr(artifact, "kind", "")) == "final_report"
            and str(getattr(artifact, "validity", "")) == "valid"
        ),
        None,
    )
    final_body = ""
    if final_report is not None:
        object_key = str(getattr(final_report, "object_key", "") or "").strip()
        if object_key:
            from app.services.content_store import get_content_store

            try:
                final_body = (await get_content_store().read(object_key)).decode("utf-8", errors="replace").strip()
            except FileNotFoundError:
                final_body = ""
        if not final_body:
            summary = dict(getattr(final_report, "summary_json", None) or {})
            final_body = str(summary.get("text") or summary.get("excerpt") or "").strip()
    chosen = _strip_runtime_report_appendix(final_body) or provisional
    if not chosen:
        return None
    return _honest_canvas_narration(chosen, has_valid_canvas=has_valid_canvas) or None


def _manifest_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _gap_lines(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _manifest_from_retrieval_sources(runtime_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build display manifest entries from neutral runtime retrieval source lists."""

    entries: list[dict[str, Any]] = []
    for origin, key in (("document", "document_sources"), ("web", "web_sources")):
        for source in runtime_result.get(key) or []:
            if not isinstance(source, Mapping):
                continue
            source_id = str(
                source.get("source_id")
                or source.get("manifest_id")
                or source.get("chunk_id")
                or ""
            ).strip()
            if not source_id:
                continue
            entries.append({
                "id": source_id,
                "kind": "retrieval_source",
                "source_id": source_id,
                "file_hash": source.get("file_hash"),
                "file_name": source.get("file_name") or source.get("title"),
                "provenance": {"origin": origin, "pages": source.get("pages"), "score": source.get("score")},
            })
    return entries


def _manifest_from_tool_events(
    tool_events: list[Any] | None,
    *,
    run_events: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Synthesize manifest entries from persisted tool telemetry when artifacts were not staged."""

    entries: list[dict[str, Any]] = []
    candidates: list[Mapping[str, Any]] = []
    for event in tool_events or []:
        if isinstance(event, Mapping):
            candidates.append(event)
    for event in run_events or []:
        if str(getattr(event, "kind", "") or "") != "tool.completed":
            continue
        payload = dict(getattr(event, "payload_json", None) or {})
        if payload:
            candidates.append(payload)
    for payload in candidates:
        tool_name = str(payload.get("tool_name") or "").strip()
        if not tool_name or tool_name not in MANIFEST_EVIDENCE_TOOLS:
            continue
        if payload.get("ok") is not True:
            continue
        result_count = int(payload.get("result_count") or payload.get("source_count") or 0)
        if result_count <= 0:
            continue
        entry_id = str(payload.get("tool_call_id") or payload.get("mcp_request_id") or "").strip()
        if not entry_id:
            continue
        entries.append({
            "id": entry_id,
            "kind": "tool_output",
            "tool_name": tool_name,
            "result_count": result_count,
            "evidence_type": payload.get("evidence_type"),
            "provenance": {"origin": "tool_telemetry"},
        })
    return entries


def _manifest_from_run_artifacts(artifacts: list[Any], run_id: str) -> list[dict[str, Any]]:
    """Build a display manifest from persisted product artifacts for one run."""

    if not run_id:
        return []
    entries: list[dict[str, Any]] = []
    for artifact in artifacts:
        if str(getattr(artifact, "agent_run_id", "")) != run_id:
            continue
        if str(getattr(artifact, "validity", "")) != "valid":
            continue
        kind = str(getattr(artifact, "kind", ""))
        if kind not in {"tool_output", "intermediate_report", "context_summary"}:
            continue
        artifact_id = str(getattr(artifact, "id", "") or "")
        if not artifact_id:
            continue
        provenance = dict(getattr(artifact, "provenance_json", None) or {})
        source_refs = dict(getattr(artifact, "source_refs_json", None) or {})
        entries.append({
            "id": artifact_id,
            "artifact_id": artifact_id,
            "kind": kind,
            "sha256": getattr(artifact, "sha256", None),
            "byte_size": getattr(artifact, "byte_size", None),
            "todo_id": getattr(artifact, "todo_id", None),
            "media_type": getattr(artifact, "media_type", None),
            **({"source_refs": source_refs} if source_refs else {}),
            **({"provenance": provenance} if provenance else {}),
        })
    return entries


def resolve_run_evidence_manifest(
    run: Any,
    artifacts: list[Any] | None = None,
    *,
    run_events: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve display evidence from runtime projection, metadata, or legacy artifacts."""

    metadata = dict(getattr(run, "run_metadata_json", None) or {})
    projection = dict(metadata.get("projection") or {})
    runtime_result = dict(projection.get("runtime_result") or {})
    orchestration_result = dict(metadata.get("orchestration_result") or {})
    task_result = dict(orchestration_result.get("task_result") or runtime_result.get("runtime_task_result") or {})

    manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in (
        metadata.get("task_evidence_manifest"),
        metadata.get("evidence_manifest"),
        runtime_result.get("task_evidence_manifest"),
        runtime_result.get("evidence_manifest"),
        runtime_result.get("task_artifact_manifest"),
        orchestration_result.get("task_evidence_manifest"),
        task_result.get("task_evidence_manifest"),
        task_result.get("evidence_manifest"),
        task_result.get("artifacts"),
    ):
        for entry in _manifest_entries(source):
            entry_id = str(entry.get("id") or entry.get("artifact_id") or entry.get("source_id") or "")
            dedupe_key = entry_id or str(hashlib.sha256(repr(sorted(entry.items())).encode()).hexdigest())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            manifest.append(entry)

    run_id = str(getattr(run, "id", "") or "")
    if not manifest and artifacts:
        for entry in _manifest_from_run_artifacts(artifacts, run_id):
            entry_id = str(entry.get("id") or entry.get("artifact_id") or "")
            if entry_id in seen:
                continue
            seen.add(entry_id)
            manifest.append(entry)

    final_report = None
    if artifacts:
        final_report = next(
            (
                artifact
                for artifact in artifacts
                if str(getattr(artifact, "agent_run_id", "")) == run_id
                and str(getattr(artifact, "kind", "")) == "final_report"
                and str(getattr(artifact, "validity", "")) == "valid"
            ),
            None,
        )
    if final_report is not None:
        provenance = dict(getattr(final_report, "provenance_json", None) or {})
        for entry in _manifest_entries(provenance.get("evidence_manifest")):
            entry_id = str(entry.get("id") or entry.get("artifact_id") or "")
            dedupe_key = entry_id or str(hashlib.sha256(repr(sorted(entry.items())).encode()).hexdigest())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            manifest.append(entry)

    if not manifest:
        for entry in (
            *_manifest_from_retrieval_sources(runtime_result),
            *_manifest_from_tool_events(runtime_result.get("tool_events"), run_events=run_events),
        ):
            entry_id = str(entry.get("id") or entry.get("artifact_id") or entry.get("source_id") or "")
            dedupe_key = entry_id or str(hashlib.sha256(repr(sorted(entry.items())).encode()).hexdigest())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            manifest.append(entry)

    gaps = list(dict.fromkeys(
        _gap_lines(metadata.get("evidence_gaps"))
        + _gap_lines(metadata.get("task_evidence_gaps"))
        + _gap_lines(runtime_result.get("task_evidence_gaps"))
        + _gap_lines(runtime_result.get("task_incomplete_reasons"))
        + _gap_lines(task_result.get("gaps"))
        + _gap_lines((dict(getattr(final_report, "provenance_json", None) or {})).get("evidence_gaps"))
    ))
    return manifest, gaps
