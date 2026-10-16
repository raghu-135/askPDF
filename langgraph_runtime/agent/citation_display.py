"""Human-readable labels for canonical evidence source ids."""

from __future__ import annotations

from typing import Any, Mapping


def _page_label(entry: Mapping[str, Any]) -> str | None:
    for key in ("page_number", "page", "page_start"):
        value = entry.get(key)
        if value is not None and str(value).strip() != "":
            return f"page {value}"
    chunk = entry.get("chunk_id")
    if chunk is not None and str(chunk).strip() != "":
        locator = str(chunk)
        if locator.isdigit():
            return f"page {locator}"
        return f"section {locator}"
    return None


def manifest_entry_display_label(entry: Mapping[str, Any]) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    name = (
        entry.get("file_name")
        or entry.get("filename")
        or entry.get("title")
        or entry.get("document_title")
        or entry.get("label")
    )
    name = str(name or "").strip()
    page = _page_label(entry)
    if name and page:
        return f"{name}, {page}"
    if name:
        return name
    if page:
        return f"document, {page}"
    return None


def build_citation_display_map(manifest: Any) -> dict[str, str]:
    if not isinstance(manifest, list):
        return {}
    result: dict[str, str] = {}
    for entry in manifest:
        if not isinstance(entry, Mapping):
            continue
        label = manifest_entry_display_label(entry)
        if not label:
            continue
        for key in ("source_id", "id", "canonical_source_id"):
            raw = str(entry.get(key) or "").strip()
            if raw:
                result[raw] = label
    return result


def doc_id_fallback_label(source_id: str) -> str | None:
    text = str(source_id or "").strip()
    if not text.startswith("doc:"):
        return None
    parts = text.split(":", 2)
    if len(parts) != 3:
        return None
    locator = parts[2]
    if locator.isdigit():
        return f"document, page {locator}"
    return f"document, section {locator}"
