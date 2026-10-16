"""Build citation menus and canvas catalogs for synthesis prompts."""

from __future__ import annotations

import json
from typing import Any, Mapping


def _packet_source_ids(packet: Mapping[str, Any]) -> list[str]:
    ids = packet.get("source_ids")
    if isinstance(ids, list):
        return [str(item) for item in ids if item]
    refs = packet.get("source_refs")
    if isinstance(refs, list):
        return [str(item) for item in refs if item]
    return []


def _locator_sentence_ids(locator: Mapping[str, Any]) -> list[int]:
    raw = locator.get("sentence_ids")
    if isinstance(raw, list):
        out: list[int] = []
        for item in raw:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        if out:
            return out
    single = locator.get("sentence_id")
    if single is not None:
        try:
            return [int(single)]
        except (TypeError, ValueError):
            return []
    return []


def citation_menu_from_state(state: Mapping[str, Any], *, limit: int = 40) -> list[dict[str, Any]]:
    menu: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(entry: dict[str, Any]) -> None:
        source_id = str(entry.get("source_id") or "").strip()
        if not source_id or source_id in seen:
            return
        seen.add(source_id)
        menu.append(entry)

    for packet in state.get("evidence_packets") or []:
        if not isinstance(packet, Mapping):
            continue
        source_ids = _packet_source_ids(packet)
        locator = packet.get("locator") if isinstance(packet.get("locator"), Mapping) else {}
        display = packet.get("display") if isinstance(packet.get("display"), Mapping) else {}
        label = str(display.get("title") or packet.get("label") or "Evidence")[:160]
        sentence_ids = _locator_sentence_ids(locator)
        page = locator.get("page_start") or locator.get("page")
        for source_id in source_ids:
            add({
                "source_id": source_id,
                "label": label,
                "page": page,
                "sentence_ids": sentence_ids[:5],
            })
            if len(menu) >= limit:
                return menu

    for collection_key in ("document_sources", "web_sources"):
        for source in state.get(collection_key) or []:
            if not isinstance(source, Mapping):
                continue
            from langgraph_runtime.agent.evidence_contract import canonical_source_id

            source_id = canonical_source_id(source)
            if not source_id:
                continue
            label = str(source.get("title") or source.get("file_name") or source.get("label") or "Source")[:160]
            locator = {k: source.get(k) for k in ("page_start", "page", "chunk_id", "sentence_ids") if source.get(k) is not None}
            add({
                "source_id": source_id,
                "label": label,
                "page": source.get("page_start") or source.get("page"),
                "sentence_ids": _locator_sentence_ids(locator),
            })
            if len(menu) >= limit:
                return menu

    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), Mapping) else {}
    for source in bundle.get("document_sources") or []:
        if not isinstance(source, Mapping):
            continue
        from langgraph_runtime.agent.evidence_contract import canonical_source_id

        source_id = canonical_source_id(source)
        if not source_id:
            continue
        label = str(source.get("title") or source.get("file_name") or "Document")[:160]
        add({
            "source_id": source_id,
            "label": label,
            "page": source.get("page_start") or source.get("page"),
            "sentence_ids": [],
        })
        if len(menu) >= limit:
            return menu

    return menu


def format_citation_menu_section(state: Mapping[str, Any]) -> str:
    menu = citation_menu_from_state(state)
    if not menu:
        return ""
    lines = [
        "## Citation menu",
        "",
        "Use these exact source_id values in publish_canvas claims (doc:… or src_… from retrieval). Do not invent ids.",
        "",
    ]
    for item in menu:
        sentence_part = ""
        sentence_ids = item.get("sentence_ids") or []
        if sentence_ids:
            sentence_part = f" sentence_ids={sentence_ids}"
        page = item.get("page")
        page_part = f" page={page}" if page not in (None, "") else ""
        lines.append(f"- `{item['source_id']}` — {item['label']}{page_part}{sentence_part}")
    return "\n".join(lines)


def format_current_canvases_section(state: Mapping[str, Any]) -> str:
    canvases = state.get("current_canvases") or []
    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), Mapping) else {}
    if not canvases and isinstance(bundle.get("current_canvases"), list):
        canvases = bundle["current_canvases"]
    if not canvases:
        return ""
    lines = [
        "## Current canvases on this thread",
        "",
        "Use supersedes_id only when replacing one of these (copy id exactly).",
        "",
    ]
    for item in canvases:
        if not isinstance(item, Mapping):
            continue
        canvas_id = str(item.get("id") or "").strip()
        title = str(item.get("title") or "Research canvas").strip()
        if canvas_id:
            lines.append(f"- `{canvas_id}` — {title}")
    return "\n".join(lines) if len(lines) > 4 else ""


def canvas_prompt_sections(state: Mapping[str, Any]) -> str:
    parts = [format_citation_menu_section(state), format_current_canvases_section(state)]
    return "\n\n".join(part for part in parts if part).strip()
