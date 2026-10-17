"""Shared deep-research gap filtering for control plane and runtime workers."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

_OBJECTIVE_WEB_EVIDENCE_PATTERN = re.compile(
    r"\b(web\s+search|live\s+web|industry\s+(?:benchmark|comparison)|external\s+sources?|web\s+benchmarks?)\b",
    re.IGNORECASE,
)
_WAIVE_WEB_GUIDANCE_PATTERN = re.compile(
    r"\b(skip|omit|ignore|waive|without|no\s+need\s+for)\b[\s\S]{0,40}\b(web|benchmarks?|industry\s+comparison)",
    re.IGNORECASE,
)

_INDEX_READINESS_GAP_MARKERS = (
    "document index is not ready",
    "document conversion is not ready",
    "document context is being repaired",
)


def text_cites_document_evidence(*texts: str) -> bool:
    joined = "\n".join(str(text or "") for text in texts)
    if re.search(r"`doc:", joined):
        return True
    if re.search(r"\[Source:\s*PDF:", joined, re.IGNORECASE):
        return True
    if re.search(r"\([^)]+\.pdf,\s*page\s+\d+\)", joined, re.IGNORECASE):
        return True
    if re.search(r"\.pdf\b", joined, re.IGNORECASE) and re.search(r"\bpage\s+\d+\b", joined, re.IGNORECASE):
        return True
    return False


def gap_mentions_index_not_ready(gap: str) -> bool:
    lowered = str(gap or "").casefold().strip()
    if not lowered:
        return False
    if any(marker in lowered for marker in _INDEX_READINESS_GAP_MARKERS):
        return True
    if lowered in {"document index not ready", "index not ready"}:
        return True
    return bool(re.search(r"document\s+index\s+(?:is\s+)?not\s+(\w+\s+)*ready", lowered))


def limitation_topic_keys(text: str) -> set[str]:
    lowered = str(text or "").casefold()
    topics: set[str] = set()
    if gap_mentions_index_not_ready(lowered) or re.search(
        r"\bindex(?:ing)?\s+(?:is\s+)?not\s+ready\b", lowered
    ):
        topics.add("index_readiness")
    if re.search(
        r"\bno\s+staff\b|staff\s+information\s+not|without\s+staff|does\s+not\s+provide\s+(?:any\s+)?staff",
        lowered,
    ):
        topics.add("staff_absence")
    if "web search is off" in lowered or "web benchmarks" in lowered:
        topics.add("web_off")
    return topics


def manifest_has_document_evidence(manifest: Any) -> bool:
    if not isinstance(manifest, list):
        return False
    for entry in manifest:
        if not isinstance(entry, Mapping):
            continue
        for key in ("source_id", "id", "canonical_source_id"):
            raw = str(entry.get(key) or "").strip()
            if raw.startswith("doc:") or raw.startswith("src_"):
                return True
        if entry.get("file_hash") or entry.get("document_id"):
            return True
    return False


def filter_stale_index_readiness_gaps(
    gaps: list[str],
    *,
    manifest: Any = None,
    evidence_texts: Iterable[str] = (),
) -> list[str]:
    """Drop transient indexing gaps when manifest or synthesis already cites document evidence."""
    if not gaps:
        return gaps
    has_doc_evidence = manifest_has_document_evidence(manifest) or text_cites_document_evidence(*evidence_texts)
    if not has_doc_evidence:
        return gaps
    kept: list[str] = []
    for gap in gaps:
        if gap_mentions_index_not_ready(gap):
            continue
        kept.append(gap)
    return kept


def normalize_limitation_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", "", str(text or "").casefold())).strip()


def gap_display_line(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, Mapping):
        for key in ("message", "summary", "text", "description"):
            raw = entry.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        code = str(entry.get("code") or "").strip()
        if code:
            return code.replace("_", " ")
    return str(entry or "").strip()


def guidance_waives_web_benchmarks(*texts: str) -> bool:
    return any(_WAIVE_WEB_GUIDANCE_PATTERN.search(str(text or "")) for text in texts)


def objective_requests_live_web(objective: str) -> bool:
    return bool(_OBJECTIVE_WEB_EVIDENCE_PATTERN.search(str(objective or "")))


def web_evidence_plan_gaps(
    objective: str,
    web_search_mode: str,
    *guidance_texts: str,
) -> list[str]:
    if guidance_waives_web_benchmarks(*guidance_texts):
        return []
    if not objective_requests_live_web(objective):
        return []
    if web_search_mode == "off":
        return [
            "Web benchmarks or live web evidence were requested in the objective, but web search is off for this task.",
        ]
    return []


_NEGATIVE_LIMITATION_PATTERN = re.compile(
    r"\b(?:no|not|missing|without|lack(?:s|ing)?|absent|never|did\s+not|does\s+not|didn't|doesn't)\b",
    re.IGNORECASE,
)


def _limitation_denies_canvas_presence(text: str) -> bool:
    lowered = str(text or "").casefold()
    if "canvas" not in lowered:
        return False
    return bool(_NEGATIVE_LIMITATION_PATTERN.search(lowered))


def _limitation_denies_document_citations(text: str) -> bool:
    lowered = str(text or "").casefold()
    if "citation" not in lowered:
        return False
    return bool(_NEGATIVE_LIMITATION_PATTERN.search(lowered))


def _limitation_denies_publish_canvas_tool(text: str) -> bool:
    lowered = str(text or "").casefold()
    if "canvas" not in lowered or "publish" not in lowered:
        return False
    return bool(
        _NEGATIVE_LIMITATION_PATTERN.search(lowered)
        or re.search(r"\b(?:follow|instruction|tool)\b", lowered)
    )


def is_false_canvas_absence_limitation(text: str) -> bool:
    return (
        _limitation_denies_canvas_presence(text)
        or _limitation_denies_document_citations(text)
        or _limitation_denies_publish_canvas_tool(text)
    )


def canvas_absence_limitations_suppressed(state: Mapping[str, Any] | None) -> bool:
    if not isinstance(state, Mapping):
        return False
    if state.get("task_canvas_published") or state.get("task_canvas_publish_required"):
        return True
    draft = state.get("task_canvas_draft")
    return isinstance(draft, Mapping) and bool(draft)


def filter_false_canvas_absence_limitations(issues: list[str]) -> list[str]:
    return [issue for issue in issues if not is_false_canvas_absence_limitation(issue)]


def strip_false_canvas_limitations_block(text: str) -> str:
    raw = str(text or "")
    match = re.search(r"\n\nLimitations:\s*(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return raw.strip()
    prefix = raw[:match.start()].rstrip()
    issues = [part.strip() for part in match.group(1).split(";") if part.strip()]
    kept = filter_false_canvas_absence_limitations(issues)
    if not kept:
        return prefix
    return f"{prefix}\n\nLimitations: {'; '.join(kept)}".strip()


def dedupe_critic_issues(issues: list[str], gaps: list[str]) -> list[str]:
    gap_keys = {normalize_limitation_key(value) for value in gaps if str(value).strip()}
    gap_keys = {value for value in gap_keys if value}
    gap_topics: set[str] = set()
    for gap in gaps:
        gap_topics |= limitation_topic_keys(gap)
    deduped: list[str] = []
    for issue in issues:
        issue_key = normalize_limitation_key(issue)
        if not issue_key:
            continue
        issue_topics = limitation_topic_keys(issue)
        if issue_topics and issue_topics <= gap_topics:
            continue
        redundant = False
        for gap_key in gap_keys:
            if issue_key == gap_key:
                redundant = True
                break
            if len(issue_key) >= 8 and issue_key in gap_key:
                redundant = True
                break
            if len(gap_key) >= 8 and gap_key in issue_key:
                redundant = True
                break
        if not redundant:
            deduped.append(issue)
    return deduped
