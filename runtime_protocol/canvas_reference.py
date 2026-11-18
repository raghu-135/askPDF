"""Detect research-canvas ids and shareable links in user text."""

from __future__ import annotations

import re
from typing import Callable, Iterable
from urllib.parse import parse_qs, urlparse

CANVAS_ARTIFACT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_FILE_HASH_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)

# Stable share params (aligned with frontend workspace-deep-link.ts / PR #105).
CANVAS_URL_QUERY_PARAM = "canvas"
THREAD_URL_QUERY_PARAM = "thread"


def is_canvas_artifact_id(value: str) -> bool:
    return bool(CANVAS_ARTIFACT_ID_RE.fullmatch(str(value or "").strip()))


def _canvas_id_from_url(url: str) -> str | None:
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme and not parsed.path.startswith("/"):
        return None
    query = parse_qs(parsed.query)
    for key in (CANVAS_URL_QUERY_PARAM, "canvas_id", "canvasId", "research_canvas_id"):
        values = query.get(key) or []
        for item in values:
            candidate = str(item or "").strip()
            if is_canvas_artifact_id(candidate):
                return candidate.lower()
    path_parts = [part for part in parsed.path.split("/") if part]
    for index, part in enumerate(path_parts):
        lowered = part.lower()
        if lowered in {"canvases", "canvas"} and index + 1 < len(path_parts):
            candidate = path_parts[index + 1].strip()
            if is_canvas_artifact_id(candidate):
                return candidate.lower()
    return None


def extract_canvas_references(text: str) -> list[str]:
    """Return canvas artifact ids mentioned in plain text or URLs, in order."""

    if not str(text or "").strip():
        return []
    seen: set[str] = set()
    ordered: list[str] = []

    def add(candidate: str | None) -> None:
        if not candidate:
            return
        normalized = str(candidate).strip().lower()
        if not is_canvas_artifact_id(normalized) or _FILE_HASH_RE.fullmatch(normalized):
            return
        if normalized in seen:
            return
        seen.add(normalized)
        ordered.append(normalized)

    for token in re.findall(r"https?://\S+", str(text)):
        trailing = ""
        while token and token[-1] in ".,);]>\"'":
            trailing = token[-1] + trailing
            token = token[:-1]
        add(_canvas_id_from_url(token))

    for match in CANVAS_ARTIFACT_ID_RE.finditer(str(text)):
        add(match.group(0))

    return ordered


def pick_quoted_canvas_supersedes_target(
    references: Iterable[str],
    *,
    current_canvas_ids: set[str],
    lineage_head_resolver: Callable[[str], str | None] | None = None,
) -> tuple[str | None, list[str]]:
    """Choose the last quoted id that names a current canvas (or its lineage head)."""

    warnings: list[str] = []
    refs = [item for item in references if is_canvas_artifact_id(item)]
    if not refs:
        return None, warnings
    resolve_head = lineage_head_resolver or (lambda canvas_id: canvas_id)
    for canvas_id in reversed(refs):
        head = str(resolve_head(canvas_id) or "").strip().lower()
        if head in current_canvas_ids:
            if head != canvas_id:
                warnings.append("quoted_canvas_resolved_to_current_head")
            return head, warnings
    warnings.append("quoted_canvas_not_current")
    return None, warnings