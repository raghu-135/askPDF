"""Serialize stored canvas specs into publish_canvas revision context for quoted updates."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

_CANONICAL_META_SECTIONS = frozenset({"overview", "sources"})
_BODY_SECTION_MARKERS = ("matrix", "comparison", "timeline", "compare")


def _section_key(title: Any) -> str:
    return str(title or "").strip().lower()


def _is_core_body_section(title: str) -> bool:
    key = _section_key(title)
    if key in _CANONICAL_META_SECTIONS:
        return True
    return any(marker in key for marker in _BODY_SECTION_MARKERS)


def _spec_sections(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for section in spec.get("sections") or []:
        if isinstance(section, Mapping):
            sections.append(copy.deepcopy(dict(section)))
    return sections


def _normalize_md(text: Any) -> str:
    return " ".join(str(text or "").split())


def _markdown_blocks(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in section.get("blocks") or []:
        if isinstance(block, Mapping) and block.get("type") == "markdown":
            text = str(block.get("text") or "").strip()
            if text:
                blocks.append(dict(block))
    return blocks


def _title_from_markdown(text: str) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                return heading[:120]
    first = str(text or "").splitlines()[0].strip() if str(text or "").strip() else ""
    return (first[:120] if first else "Opening")


def _summary_promoted_as_preface(
    baseline: Mapping[str, Any],
    incoming: Mapping[str, Any],
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """When supersede updates the spec summary, surface it as its own section."""

    incoming_summary = str(incoming.get("summary") or "").strip()
    if not incoming_summary:
        return sections
    baseline_summary = str(baseline.get("summary") or "").strip()
    if _normalize_md(incoming_summary) == _normalize_md(baseline_summary):
        return sections

    incoming_norm = _normalize_md(incoming_summary)
    for section in sections:
        key = _section_key(section.get("title"))
        if key in _CANONICAL_META_SECTIONS:
            continue
        for block in _markdown_blocks(section):
            block_norm = _normalize_md(block.get("text"))
            if block_norm == incoming_norm or incoming_norm in block_norm:
                return sections

    preface_title = _title_from_markdown(incoming_summary)
    if any(_section_key(section.get("title")) == _section_key(preface_title) for section in sections):
        return sections
    preface = {
        "title": preface_title,
        "blocks": [{"type": "markdown", "text": incoming_summary}],
    }
    return [preface, *sections]


def _extra_overview_markdown(
    baseline: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> list[dict[str, Any]]:
    baseline_texts = {_normalize_md(block.get("text")) for block in _markdown_blocks(baseline)}
    extras: list[dict[str, Any]] = []
    for block in _markdown_blocks(incoming):
        text = _normalize_md(block.get("text"))
        if text and text not in baseline_texts:
            extras.append(dict(block))
    return extras


def _non_markdown_blocks(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for block in section.get("blocks") or []:
        if isinstance(block, Mapping) and block.get("type") != "markdown":
            kept.append(copy.deepcopy(dict(block)))
    return kept


def _merge_table_block(baseline: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    base_rows = list(baseline.get("rows") or [])
    incoming_rows = list(incoming.get("rows") or [])
    rows = incoming_rows if len(incoming_rows) >= len(base_rows) else base_rows
    merged = {**dict(baseline), **dict(incoming), "rows": rows}
    return merged


def _merge_table_section(baseline: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    base_blocks = [dict(b) for b in baseline.get("blocks") or [] if isinstance(b, Mapping)]
    incoming_blocks = [dict(b) for b in incoming.get("blocks") or [] if isinstance(b, Mapping)]
    merged_blocks: list[dict[str, Any]] = []
    used_incoming: set[int] = set()
    for base_block in base_blocks:
        if base_block.get("type") != "table":
            merged_blocks.append(copy.deepcopy(base_block))
            continue
        partner = None
        partner_index = None
        for index, inc_block in enumerate(incoming_blocks):
            if index in used_incoming or inc_block.get("type") != "table":
                continue
            partner = inc_block
            partner_index = index
            break
        if partner is None:
            merged_blocks.append(copy.deepcopy(base_block))
        else:
            used_incoming.add(partner_index)
            merged_blocks.append(_merge_table_block(base_block, partner))
    for index, inc_block in enumerate(incoming_blocks):
        if index not in used_incoming:
            merged_blocks.append(copy.deepcopy(inc_block))
    return {**dict(baseline), **dict(incoming), "blocks": merged_blocks}


def _merge_sources_section(baseline: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_citation(item: Mapping[str, Any]) -> None:
        key = json.dumps(item, sort_keys=True, separators=(",", ":"))
        if key in seen:
            return
        seen.add(key)
        citations.append(dict(item))

    for section in (baseline, incoming):
        for block in section.get("blocks") or []:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") not in ("source_list", "sources"):
                continue
            for citation in block.get("citations") or []:
                if isinstance(citation, Mapping):
                    add_citation(citation)
    return {
        **dict(baseline),
        **dict(incoming),
        "blocks": [{"type": "sources", "citations": citations}],
    }


def merge_canvas_supersede_spec(
    baseline: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve baseline evidence while applying additive section and overview edits."""

    baseline_sections = _spec_sections(baseline)
    incoming_sections = _spec_sections(incoming)
    baseline_by_key = {_section_key(section.get("title")): section for section in baseline_sections}
    incoming_by_key = {_section_key(section.get("title")): section for section in incoming_sections}

    preface_sections: list[dict[str, Any]] = []
    for section in incoming_sections:
        title = str(section.get("title") or "")
        key = _section_key(title)
        if key in baseline_by_key or _is_core_body_section(title):
            continue
        if key in _CANONICAL_META_SECTIONS:
            continue
        preface_sections.append(copy.deepcopy(section))

    overview_baseline = baseline_by_key.get("overview")
    overview_incoming = incoming_by_key.get("overview")
    if overview_baseline and overview_incoming:
        extras = _extra_overview_markdown(overview_baseline, overview_incoming)
        if extras:
            preface_sections.insert(
                0,
                {"title": _title_from_markdown(str(extras[0].get("text") or "")), "blocks": extras},
            )
            kept_md = [
                block
                for block in _markdown_blocks(overview_incoming)
                if _normalize_md(block.get("text")) in {
                    _normalize_md(item.get("text")) for item in _markdown_blocks(overview_baseline)
                }
            ]
            overview_incoming = {
                **dict(overview_incoming),
                "blocks": [*_non_markdown_blocks(overview_baseline), *kept_md],
            }
            incoming_by_key["overview"] = overview_incoming

    merged_sections: list[dict[str, Any]] = [*preface_sections]
    for baseline_section in baseline_sections:
        key = _section_key(baseline_section.get("title"))
        incoming_section = incoming_by_key.get(key)
        if incoming_section is None:
            merged_sections.append(copy.deepcopy(baseline_section))
            continue
        if key == "overview":
            merged_sections.append({
                **copy.deepcopy(baseline_section),
                **copy.deepcopy(incoming_section),
                "blocks": [
                    *_non_markdown_blocks(baseline_section),
                    *_markdown_blocks(overview_incoming or incoming_section),
                ],
            })
        elif any(
            isinstance(block, Mapping) and block.get("type") == "table"
            for block in [
                *(baseline_section.get("blocks") or []),
                *(incoming_section.get("blocks") or []),
            ]
        ):
            merged_sections.append(_merge_table_section(baseline_section, incoming_section))
        elif key == "sources":
            merged_sections.append(_merge_sources_section(baseline_section, incoming_section))
        else:
            merged_sections.append(copy.deepcopy(incoming_section))

    for section in incoming_sections:
        key = _section_key(section.get("title"))
        if key in baseline_by_key:
            continue
        if any(_section_key(existing.get("title")) == key for existing in merged_sections):
            continue
        if _is_core_body_section(str(section.get("title") or "")):
            merged_sections.append(copy.deepcopy(section))

    title = str(incoming.get("title") or baseline.get("title") or "").strip()
    summary = incoming.get("summary")
    if summary is None:
        summary = baseline.get("summary")
    merged_sections = _summary_promoted_as_preface(baseline, incoming, merged_sections)
    return {
        **copy.deepcopy(dict(baseline)),
        **copy.deepcopy(dict(incoming)),
        "schema_version": incoming.get("schema_version") or baseline.get("schema_version") or 1,
        "title": title,
        "summary": summary,
        "sections": merged_sections,
    }


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
            if block.get("type") in ("source_list", "sources"):
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
                        row_index = len(claims)
                        if row_index < len(citations):
                            picked = citation_to_source_id(citations[row_index])
                            if picked:
                                source_ids = [picked]
                        if not source_ids:
                            picked = citation_to_source_id(citations[0])
                            if picked:
                                source_ids = [picked]
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
        "When the user asks to change this canvas (summary, wording, sections, etc.), you **must** call "
        "`publish_canvas` with `supersedes_id` set to the quoted canvas id. Start from the draft below, "
        "keep every existing claim and `source_id` unless the user explicitly asks to remove one, and apply "
        "only the requested edit. When the user asks for a new section, publish it as an "
        "extra section (do not fold it into the overview only). "
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