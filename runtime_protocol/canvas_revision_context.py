"""Serialize stored canvas specs into publish_canvas revision context for quoted updates."""

from __future__ import annotations

import json
from typing import Any, Mapping


def _cell_text(cell: Any) -> str:
    if isinstance(cell, str):
        return cell.strip()
    if isinstance(cell, Mapping):
        return str(cell.get("text") or "").strip()
    return ""


def _tone_from_cell(cell: Any) -> str | None:
    if isinstance(cell, Mapping):
        tone = cell.get("tone")
        if tone in {"supported", "contradicted", "partial", "absent"}:
            return str(tone)
    return None


def _normalized_web_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw.startswith(("http://", "https://")):
        return ""
    return raw.rstrip("/") or raw


def citation_to_source_id(citation: Mapping[str, Any]) -> str:
    kind = str(citation.get("kind") or "").strip()
    if kind == "web":
        url = _normalized_web_url(citation.get("url"))
        return f"web:{url}" if url else ""
    if kind == "document":
        file_hash = str(citation.get("file_hash") or "").strip()
        if not file_hash:
            return ""
        sentence_id = citation.get("sentence_id")
        if sentence_id is not None:
            try:
                return f"doc:{file_hash}:s-{int(sentence_id)}"
            except (TypeError, ValueError):
                pass
        sentence_ids = citation.get("sentence_ids") or []
        if sentence_ids:
            try:
                return f"doc:{file_hash}:s-{int(sentence_ids[0])}"
            except (TypeError, ValueError, IndexError):
                pass
        return f"doc:{file_hash}:document"
    if kind == "conversation":
        message_id = str(citation.get("message_id") or "").strip()
        return f"conversation:{message_id}" if message_id else ""
    if kind == "memory":
        memory_id = str(citation.get("memory_id") or "").strip()
        return f"memory:{memory_id}" if memory_id else ""
    return ""


def _collect_citations(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for section in spec.get("sections") or []:
        if not isinstance(section, Mapping):
            continue
        for block in section.get("blocks") or []:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "source_list":
                for item in block.get("citations") or []:
                    if isinstance(item, Mapping):
                        citations.append(dict(item))
            if block.get("type") == "timeline":
                for event in block.get("events") or []:
                    if isinstance(event, Mapping) and isinstance(event.get("citation"), Mapping):
                        citations.append(dict(event["citation"]))
            if block.get("type") == "claim_pair":
                for side in ("left", "right"):
                    side_obj = block.get(side)
                    if isinstance(side_obj, Mapping) and isinstance(side_obj.get("citation"), Mapping):
                        citations.append(dict(side_obj["citation"]))
    return citations


def _label_to_source_id(label: str, citations: list[Mapping[str, Any]]) -> list[str]:
    needle = str(label or "").strip().lower()
    if not needle:
        return []
    matched: list[str] = []
    for citation in citations:
        cite_label = str(citation.get("label") or "").strip().lower()
        source_id = citation_to_source_id(citation)
        if not source_id:
            continue
        if needle in cite_label or cite_label in needle or cite_label[:40] == needle[:40]:
            matched.append(source_id)
    if matched:
        return matched
    if len(citations) == 1:
        only = citation_to_source_id(citations[0])
        return [only] if only else []
    return []


def _infer_layout(spec: Mapping[str, Any]) -> str:
    for section in spec.get("sections") or []:
        if not isinstance(section, Mapping):
            continue
        title = str(section.get("title") or "").lower()
        if "timeline" in title:
            return "timeline"
        if "comparison" in title or "compare" in title:
            return "compare"
        if "evidence matrix" in title or "matrix" in title:
            return "matrix"
        for block in section.get("blocks") or []:
            if isinstance(block, Mapping):
                if block.get("type") == "timeline":
                    return "timeline"
                if block.get("type") == "claim_pair":
                    return "compare"
                caption = str(block.get("caption") or "").lower()
                if "evidence matrix" in caption:
                    return "matrix"
    return "matrix"


def _claims_from_spec(spec: Mapping[str, Any], citations: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for section in spec.get("sections") or []:
        if not isinstance(section, Mapping):
            continue
        for block in section.get("blocks") or []:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "table":
                headers = [str(item or "").strip() for item in block.get("headers") or []]
                claim_idx = headers.index("Claim") if "Claim" in headers else 0
                source_idx = headers.index("Source") if "Source" in headers else None
                tone_idx = headers.index("Support") if "Support" in headers else None
                for row in block.get("rows") or []:
                    if not isinstance(row, list) or not row:
                        continue
                    text = _cell_text(row[claim_idx] if claim_idx < len(row) else row[0])
                    if not text:
                        continue
                    source_ids: list[str] = []
                    if source_idx is not None and source_idx < len(row):
                        source_ids = _label_to_source_id(_cell_text(row[source_idx]), citations)
                    tone = None
                    if tone_idx is not None and tone_idx < len(row):
                        tone = _tone_from_cell(row[tone_idx])
                    if not source_ids and citations:
                        source_ids = [citation_to_source_id(citations[0])]
                    source_ids = [item for item in source_ids if item]
                    if not source_ids:
                        continue
                    entry: dict[str, Any] = {"text": text, "source_ids": source_ids}
                    if tone:
                        entry["tone"] = tone
                    claims.append(entry)
            if block.get("type") == "timeline":
                for event in block.get("events") or []:
                    if not isinstance(event, Mapping):
                        continue
                    text = str(event.get("label") or "").strip()
                    if not text:
                        continue
                    citation = event.get("citation") if isinstance(event.get("citation"), Mapping) else {}
                    source_id = citation_to_source_id(citation)
                    if not source_id:
                        continue
                    entry = {"text": text, "source_ids": [source_id]}
                    time_val = str(event.get("time") or "").strip()
                    if time_val and time_val != "—":
                        entry["time"] = time_val
                    claims.append(entry)
            if block.get("type") == "claim_pair":
                for side in ("left", "right"):
                    side_obj = block.get(side)
                    if not isinstance(side_obj, Mapping):
                        continue
                    text = str(side_obj.get("text") or "").strip()
                    citation = side_obj.get("citation") if isinstance(side_obj.get("citation"), Mapping) else {}
                    source_id = citation_to_source_id(citation)
                    if text and source_id:
                        claims.append({"text": text, "source_ids": [source_id]})
    return claims


def _overview_summary(spec: Mapping[str, Any]) -> str:
    summary = str(spec.get("summary") or "").strip()
    if summary:
        return summary
    for section in spec.get("sections") or []:
        if not isinstance(section, Mapping):
            continue
        if str(section.get("title") or "").lower() != "overview":
            continue
        for block in section.get("blocks") or []:
            if isinstance(block, Mapping) and block.get("type") == "markdown":
                text = str(block.get("text") or "").strip()
                if text:
                    return text
    return ""


def build_revision_snapshot(
    *,
    canvas_id: str,
    lineage_head_id: str,
    title: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    citations = _collect_citations(spec)
    source_ids = []
    seen_sources: set[str] = set()
    for citation in citations:
        source_id = citation_to_source_id(citation)
        if source_id and source_id not in seen_sources:
            seen_sources.add(source_id)
            source_ids.append(source_id)
    claims = _claims_from_spec(spec, citations)
    layout = _infer_layout(spec)
    publish_draft = {
        "layout": layout,
        "title": str(spec.get("title") or title or "Research canvas").strip(),
        "summary": _overview_summary(spec) or None,
        "claims": claims,
    }
    return {
        "canvas_id": canvas_id,
        "lineage_head_id": lineage_head_id,
        "title": title or publish_draft["title"],
        "claim_count": len(claims),
        "source_count": len(source_ids),
        "publish_canvas_draft": publish_draft,
        "source_ids": source_ids,
    }


def merge_publish_canvas_draft(
    baseline: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    """When superseding, keep baseline claims/sources if the incoming draft dropped them."""

    baseline_claims = [dict(item) for item in (baseline.get("claims") or []) if isinstance(item, Mapping)]
    incoming_claims = [dict(item) for item in (incoming.get("claims") or []) if isinstance(item, Mapping)]
    if len(incoming_claims) >= len(baseline_claims):
        return {**dict(baseline), **dict(incoming), "claims": incoming_claims}

    merged_claims = list(baseline_claims)
    by_text = {str(item.get("text") or "").strip(): item for item in incoming_claims if str(item.get("text") or "").strip()}
    for index, base in enumerate(merged_claims):
        text = str(base.get("text") or "").strip()
        if text in by_text:
            merged_claims[index] = {**base, **by_text[text]}
    known_texts = {str(item.get("text") or "").strip() for item in baseline_claims}
    for claim in incoming_claims:
        text = str(claim.get("text") or "").strip()
        if text and text not in known_texts:
            merged_claims.append(claim)
            known_texts.add(text)

    merged = {**dict(baseline), **dict(incoming)}
    if incoming.get("summary") is not None and str(incoming.get("summary") or "").strip():
        merged["summary"] = incoming.get("summary")
    if str(incoming.get("title") or "").strip():
        merged["title"] = incoming.get("title")
    if str(incoming.get("layout") or "").strip():
        merged["layout"] = incoming.get("layout")
    merged["claims"] = merged_claims
    return merged


def format_quoted_canvas_revisions_for_prompt(revisions: list[Mapping[str, Any]], *, max_json_chars: int = 48_000) -> str:
    if not revisions:
        return ""
    blocks: list[str] = [
        "## Quoted canvas revision baseline",
        "",
        "The user quoted a canvas to revise. Start from the `publish_canvas_draft` below.",
        "When you call `publish_canvas` with `supersedes_id` for that canvas, keep every existing claim "
        "and `source_id` unless the user explicitly asks to remove one. Apply only the requested edit "
        "(for example add a Summary section via `summary` or an extra overview markdown). "
        "Do not shrink the claim or source lists.",
        "",
    ]
    for index, revision in enumerate(revisions, start=1):
        canvas_id = str(revision.get("canvas_id") or "").strip()
        head_id = str(revision.get("lineage_head_id") or "").strip()
        claim_count = int(revision.get("claim_count") or 0)
        source_count = int(revision.get("source_count") or 0)
        draft = revision.get("publish_canvas_draft") or {}
        encoded = json.dumps(draft, ensure_ascii=True, separators=(",", ":"))
        if len(encoded) > max_json_chars:
            encoded = encoded[: max_json_chars - 20] + "…[truncated]"
        blocks.extend([
            f"### Canvas {index}: `{canvas_id}` (lineage head `{head_id}`)",
            f"- Claims: {claim_count} — Sources: {source_count}",
            "```json",
            encoded,
            "```",
            "",
        ])
    return "\n".join(blocks).strip()