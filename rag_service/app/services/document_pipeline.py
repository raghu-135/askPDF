"""Deterministic projections from the shared Docling document.

The conversion result is intentionally a plain JSON-compatible structure.  It
can be persisted, replayed, and tested without importing a vector database or
an agent runtime.  Reading and retrieval projections consume this structure
independently.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence


EXTRACTION_PIPELINE_VERSION = "docling-pdf-v3"
CANONICAL_SCHEMA_VERSION = "docling-canonical-v2"
RETRIEVAL_CHUNKING_VERSION = "sentence-pack-v4"
READING_EXCLUDED_LABELS = frozenset({
    "page_header",
    "page_footer",
    "header",
    "footer",
    "footnote",
    "caption",
})
# Keep retrieval units small and structurally local.  The tokenizer remains the
# hard bound, but a chunk must never span more than three consecutive sentences.
CHUNK_MAX_SENTENCES = 3
STRUCTURAL_CONTEXT_TOKEN_LIMIT = 96
DEFAULT_EMBEDDING_TOKEN_LIMIT = 512


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def stable_fingerprint(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(_json_bytes(value))
        digest.update(b"\0")
    return digest.hexdigest()


def stable_identity(*values: Any) -> str:
    return hashlib.sha256("|".join(str(value) for value in values).encode("utf-8")).hexdigest()[:32]


def stable_source_id(file_hash: str, generation: str, chunk_identity: str) -> str:
    """Return the schema-safe, model-independent citation identifier."""
    return f"src_{stable_identity('source', file_hash, generation, chunk_identity)}"


def is_valid_canonical_payload(payload: Any) -> bool:
    """Validate the minimum persisted shape required by current projections."""
    return (
        isinstance(payload, Mapping)
        and payload.get("schema_version") == CANONICAL_SCHEMA_VERSION
        and isinstance(payload.get("docling"), Mapping)
        and isinstance(payload.get("elements"), list)
        and isinstance(payload.get("sections"), list)
    )


def _ref_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        ref = value.get("$ref") or value.get("self_ref")
        return str(ref) if ref else None
    return None


def _pages_from_provenance(value: Any) -> list[int]:
    pages: set[int] = set()
    for prov in value if isinstance(value, list) else []:
        if not isinstance(prov, Mapping):
            continue
        try:
            page = int(prov.get("page_no"))
        except (TypeError, ValueError):
            continue
        if page > 0:
            pages.add(page)
    return sorted(pages)


def _text_for_raw_item(item: Mapping[str, Any]) -> str:
    table_label = str(item.get("label") or "") == "table"
    table_caption = ""
    for key in ("text", "orig", "caption", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            if not table_label:
                return value.strip()
            table_caption = value.strip()
            break
    if table_label:
        structure = _table_structure_for_raw_item(item)
        if structure["headers"] or structure["rows"]:
            lines: list[str] = []
            if table_caption:
                lines.append(table_caption)
            if structure["headers"]:
                lines.append("Table headers: " + " | ".join(value or f"Column {index + 1}" for index, value in enumerate(structure["headers"])))
            for index, row in enumerate(structure["rows"], start=1):
                lines.append(f"Row {index}: " + _table_row_render(structure["headers"], row, index, structure.get("cells") or []))
            return "\n".join(lines).strip()
        if table_caption:
            return table_caption
        return json.dumps(item.get("data") or item, ensure_ascii=False, sort_keys=True)
    return ""


def _cell_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("text", "value", "content", "label"):
            if value.get(key) not in (None, ""):
                return str(value[key]).strip()
        return ""
    return str(value or "").strip()


def _table_row_render(
    headers: Sequence[str],
    row: Sequence[str],
    row_index: int,
    cells: Sequence[Mapping[str, Any]],
) -> str:
    """Render explicit header/value pairs while respecting merged cells."""
    width = max(
        len(headers),
        len(row),
        max(
            (
                int(cell.get("col", 0)) + int(cell.get("col_span", 1))
                for cell in cells
                if int(cell.get("row", -1)) <= row_index < int(cell.get("row", -1)) + int(cell.get("row_span", 1))
            ),
            default=0,
        ),
    )
    values = [str(row[index]).strip() if index < len(row) else "" for index in range(width)]
    labels = [str(headers[index]).strip() or f"Column {index + 1}" for index in range(width)]
    rendered: list[str] = []
    covered: set[int] = set()
    for index in range(width):
        if index in covered:
            continue
        merged = next(
            (
                cell for cell in cells
                if int(cell.get("row", -1)) <= row_index < int(cell.get("row", -1)) + int(cell.get("row_span", 1))
                and int(cell.get("col", -1)) == index
            ),
            None,
        )
        if merged is not None:
            span = max(1, int(merged.get("col_span", 1)))
            label = " / ".join(labels[index:index + span])
            value = str(merged.get("text") or values[index]).strip()
            rendered.append(f"{label}: {value}")
            covered.update(range(index, min(width, index + span)))
            continue
        rendered.append(f"{labels[index]}: {values[index]}")
    return " | ".join(rendered)


def _table_structure_for_raw_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize table exports into a rectangular, position-preserving grid."""
    data = item.get("data") if isinstance(item.get("data"), Mapping) else {}
    headers = [_cell_text(value) for value in (data.get("headers") or item.get("headers") or [])]
    rows: list[list[str]] = []
    cells: list[dict[str, Any]] = []
    grid = data.get("grid") or data.get("rows") or data.get("table")
    if isinstance(grid, list) and all(isinstance(row, (list, tuple)) for row in grid):
        normalized = [[_cell_text(value) for value in row] for row in grid]
        if normalized and not headers:
            headers = list(normalized.pop(0))
        rows = normalized
        for row_index, row in enumerate(rows, start=1):
            for col_index, value in enumerate(row):
                cells.append({"row": row_index, "col": col_index, "row_span": 1, "col_span": 1, "text": value})
    else:
        raw_cells = data.get("table_cells") or data.get("cells") or []
        for cell in raw_cells if isinstance(raw_cells, list) else []:
            if not isinstance(cell, Mapping):
                continue
            try:
                row_index = int(cell.get("row_index", cell.get("row", 0)))
                col_index = int(cell.get("col_index", cell.get("column", 0)))
                row_span = max(1, int(cell.get("row_span", 1)))
                col_span = max(1, int(cell.get("col_span", 1)))
            except (TypeError, ValueError):
                continue
            cells.append({
                "row": row_index,
                "col": col_index,
                "row_span": row_span,
                "col_span": col_span,
                "text": _cell_text(cell),
                "column_header": bool(cell.get("column_header")),
                "row_header": bool(cell.get("row_header")),
            })
        if cells and not headers:
            header_row = min(int(cell["row"]) for cell in cells)
            header_cells = [cell for cell in cells if int(cell["row"]) == header_row]
            header_width = max((int(cell["col"]) + int(cell.get("col_span", 1)) for cell in header_cells), default=0)
            headers = ["" for _ in range(header_width)]
            for cell in header_cells:
                headers[int(cell["col"])] = str(cell["text"] or "")
            cells = [
                {**cell, "row": int(cell["row"]) - header_row}
                for cell in cells
                if int(cell["row"]) != header_row
            ]
        if cells:
            height = max(int(cell["row"]) + int(cell.get("row_span", 1)) - 1 for cell in cells)
            width = max(int(cell["col"]) + int(cell.get("col_span", 1)) for cell in cells)
            rows = [["" for _ in range(width)] for _ in range(height)]
            for cell in cells:
                row_index = int(cell["row"])
                col_index = int(cell["col"])
                target_row = row_index - 1
                if 0 <= target_row < height and col_index < width:
                    rows[target_row][col_index] = str(cell.get("text") or "")

    width = max(
        len(headers),
        max((len(row) for row in rows), default=0),
        max((int(cell.get("col", 0)) + int(cell.get("col_span", 1)) for cell in cells), default=0),
    )
    headers = [*headers, *("" for _ in range(max(0, width - len(headers))))]
    rows = [[*row, *("" for _ in range(max(0, width - len(row))))] for row in rows]
    return {"headers": headers, "rows": rows, "cells": cells}


def _iter_exported_items(docling_json: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield body-order item refs, followed by any orphaned exported items."""
    seen: set[str] = set()
    collections = {
        f"#/{name}/{index}": item
        for name in ("texts", "tables", "pictures", "groups", "key_value_items", "formulas")
        for index, item in enumerate(docling_json.get(name) or [])
        if isinstance(item, Mapping)
    }

    def walk(node: Any) -> Iterable[str]:
        if isinstance(node, Mapping) and "$ref" in node:
            ref = str(node["$ref"])
            if ref not in seen and ref in collections:
                seen.add(ref)
                yield ref
                item = collections[ref]
                for child in item.get("children") or []:
                    yield from walk(child)
            return
        if isinstance(node, Mapping):
            for child in node.get("children") or []:
                yield from walk(child)

    for ref in walk(docling_json.get("body") or {}):
        item = collections.get(ref)
        if item is not None:
            yield ref, item
    for ref, item in collections.items():
        if ref not in seen:
            yield ref, item


def build_canonical_payload(
    docling_document: Any,
    *,
    filename: str,
    source_metadata: Mapping[str, Any] | None = None,
    document_identity: str | None = None,
) -> dict[str, Any]:
    """Serialize a Docling document plus normalized structural elements."""
    exported = docling_document.export_to_dict()
    elements: list[dict[str, Any]] = []
    for order, (ref, raw) in enumerate(_iter_exported_items(exported)):
        label = str(raw.get("label") or "unspecified")
        pages = _pages_from_provenance(raw.get("prov"))
        elements.append(
            {
                "source_ref": ref,
                "element_id": stable_identity("element", document_identity or filename, ref),
                "element_order": order,
                "element_type": label,
                "label": label,
                "text": _text_for_raw_item(raw),
                "pages": pages,
                "provenance": raw.get("prov") or [],
                "parent_ref": _ref_value(raw.get("parent")),
                "paragraph_ref": _ref_value(raw.get("paragraph") or raw.get("paragraph_ref")),
                "raw": dict(raw),
                "table_structure": _table_structure_for_raw_item(raw) if label == "table" else None,
            }
        )

    element_ids_by_ref = {item["source_ref"]: item["element_id"] for item in elements}
    for item in elements:
        parent_ref = item.get("parent_ref")
        item["parent_element_id"] = element_ids_by_ref.get(parent_ref) if parent_ref else None
        paragraph_ref = item.get("paragraph_ref")
        item["paragraph_id"] = element_ids_by_ref.get(paragraph_ref) if paragraph_ref else None

    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "filename": filename,
        "docling": exported,
        "elements": elements,
        "source_metadata": dict(source_metadata or {}),
    }


def _section_header(element: Mapping[str, Any]) -> bool:
    return str(element.get("element_type") or "") in {"title", "section_header", "chapter_header"}


def derive_hierarchy(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a conservative heading hierarchy; do not invent missing ancestry."""
    sections: list[dict[str, Any]] = []
    elements: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    for element in payload.get("elements") or []:
        item = dict(element)
        if _section_header(item):
            raw_level = (item.get("raw") or {}).get("level")
            # Docling does not expose a reliable heading depth in every PDF.
            # Preserve a flat sequence when it does not; never infer ancestry
            # from the number of headings seen so far.
            level = raw_level if isinstance(raw_level, int) and raw_level > 0 else 1
            if level == 1 and not isinstance(raw_level, int):
                stack.clear()
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            parent = stack[-1] if stack else None
            path = [*(parent["heading_path"] if parent else []), str(item.get("text") or "").strip()]
            section = {
                "section_id": stable_identity("section", item.get("element_id")),
                "section_order": len(sections),
                "level": level,
                "title": str(item.get("text") or "").strip(),
                "parent_section_id": parent["section_id"] if parent else None,
                "heading_path": path,
                "element_ids": [],
                "pages": list(item.get("pages") or []),
            }
            sections.append(section)
            stack.append(section)
        current = stack[-1] if stack else None
        if current:
            current["element_ids"].append(item["element_id"])
            current["pages"] = sorted(set([*current.get("pages", []), *(item.get("pages") or [])]))
            item["section_id"] = current["section_id"]
            item["heading_path"] = list(current["heading_path"])
        else:
            item["section_id"] = None
            item["heading_path"] = []
        elements.append(item)
    return sections, elements


class SentencePipelineUnavailableError(RuntimeError):
    """Raised when the configured sentence model cannot be loaded."""


@lru_cache(maxsize=1)
def _sentence_nlp() -> Any:
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except Exception as exc:
        raise SentencePipelineUnavailableError(
            "spaCy model 'en_core_web_sm' is required and must load successfully"
        ) from exc


def sentence_pipeline_identity() -> str:
    """Return an identity only after the configured model has loaded."""
    _sentence_nlp()
    return "en_core_web_sm:loaded"


def _default_sentence_split(text: str) -> list[str]:
    nlp = _sentence_nlp()
    return [sentence.text.strip() for sentence in nlp(text) if sentence.text.strip()]


def _raw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("value")
    return value


def reading_element_policy(element: Mapping[str, Any]) -> bool:
    """Return whether an element belongs in the speech/reading projection."""
    label = str(element.get("label") or element.get("element_type") or "").strip().casefold()
    if label in READING_EXCLUDED_LABELS:
        return False
    raw = element.get("raw") if isinstance(element.get("raw"), Mapping) else {}
    content_layer = str(_raw_value(raw.get("content_layer")) or "").casefold()
    if content_layer == "furniture":
        return False
    parent_ref = str(element.get("parent_ref") or raw.get("parent") or "").casefold()
    return "picture" not in parent_ref


def _element_tags(element: Mapping[str, Any], element_type: str) -> tuple[list[str], dict[str, str]]:
    raw = element.get("raw") if isinstance(element.get("raw"), Mapping) else {}
    candidates = [(element_type, "docling_label")]
    label = str(element.get("label") or "").strip()
    if label and label != element_type:
        candidates.append((label, "docling_label"))
    content_layer = str(_raw_value(raw.get("content_layer")) or "").strip()
    if content_layer:
        candidates.append((f"content_layer:{content_layer}", "docling_content_layer"))
    if element_type in {"title", "section_header", "chapter_header"}:
        candidates.append(("heading", "canonical_structure"))
    if element_type == "table":
        candidates.append(("tabular", "canonical_structure"))
    tags: list[str] = []
    provenance: dict[str, str] = {}
    for tag, source in candidates:
        if tag and tag not in provenance:
            tags.append(tag)
            provenance[tag] = source
    return tags, provenance


def project_sentences(
    payload: Mapping[str, Any],
    *,
    sentence_splitter: Callable[[str], Sequence[str]] | None = None,
    element_policy: Callable[[Mapping[str, Any]], bool] | None = reading_element_policy,
) -> list[dict[str, Any]]:
    """Project canonical text into stable, provenance-bearing sentences.

    The default is the reading policy. Retrieval callers pass ``None`` to
    retain every canonical text element, including captions and footnotes.
    """
    splitter = sentence_splitter or _default_sentence_split
    sentences: list[dict[str, Any]] = []
    for element in payload.get("elements") or []:
        if element_policy is not None and not element_policy(element):
            continue
        text = str(element.get("text") or "").strip()
        if not text:
            continue
        element_type = str(element.get("element_type") or element.get("label") or "text")
        tags, tag_provenance = _element_tags(element, element_type)
        table_id = element.get("element_id") if element_type == "table" else None
        # A generic parent is usually a Docling group/section, not a paragraph.
        # Keep each text element as an independent packing boundary unless the
        # canonical payload contains an explicit paragraph relation.
        paragraph_id = element.get("paragraph_id") or element.get("paragraph_ref") or element.get("element_id")
        table_structure = element.get("table_structure") if element_type == "table" else None
        table_rows: list[tuple[str, str | None]] = []
        if isinstance(table_structure, Mapping):
            headers = [str(value).strip() for value in table_structure.get("headers") or []]
            cells = [dict(value) for value in table_structure.get("cells") or [] if isinstance(value, Mapping)]
            for row_index, row in enumerate(table_structure.get("rows") or [], start=1):
                values = [str(value).strip() for value in row]
                if not any(values) and not cells:
                    continue
                header_text = " | ".join(value or f"Column {index + 1}" for index, value in enumerate(headers))
                row_text = _table_row_render(headers, values, row_index, cells)
                rendered = f"Table headers: {header_text}\nRow {row_index}: {row_text}" if header_text else f"Row {row_index}: {row_text}"
                table_rows.append((rendered, f"{element.get('element_id')}:row:{row_index}"))
        projected = table_rows or [(sentence, None) for sentence in splitter(text)]
        search_cursor = 0
        for local_index, (sentence_text, table_row_id) in enumerate(projected):
            sentence_text = str(sentence_text).strip()
            if not sentence_text:
                continue
            start = text.find(sentence_text, search_cursor)
            if start < 0:
                start = 0
            else:
                search_cursor = start + len(sentence_text)
            sentence_id = len(sentences)
            sentences.append(
                {
                    "id": sentence_id,
                    "text": sentence_text,
                    "label": element.get("label") or element.get("element_type") or "text",
                    "element_type": element_type,
                    "page": (element.get("pages") or [None])[0],
                    "pages": list(element.get("pages") or []),
                    "source_element_ids": [element["element_id"]],
                    "source_element_refs": [element.get("source_ref")] if element.get("source_ref") else [],
                    "source_spans": [{"element_id": element["element_id"], "start": start, "end": start + len(sentence_text)}],
                    "section_id": element.get("section_id"),
                    "paragraph_id": paragraph_id,
                    "table_id": table_id,
                    "table_row_id": table_row_id,
                    "tags": tags,
                    "tag_provenance": tag_provenance,
                    "heading_path": list(element.get("heading_path") or []),
                    "alignment_precision": "coarse",
                    "bboxes": [],
                    "bbox": None,
                    "page_width": None,
                    "page_height": None,
                    "local_index": local_index,
                }
            )
    return sentences


@dataclass(frozen=True)
class TokenCounter:
    count: Callable[[str], int]
    split: Callable[[str, int], Sequence[str]]
    split_with_spans: Callable[[str, int], Sequence[tuple[str, int, int]]] | None = None


def _context_prefix(document_title: str | None, heading_path: Sequence[str], counter: TokenCounter) -> str:
    """Build a bounded prefix that always carries document identity."""
    title = str(document_title or "Untitled document").strip() or "Untitled document"
    title_budget = max(1, STRUCTURAL_CONTEXT_TOKEN_LIMIT - counter.count("Document: "))
    title_parts = list(counter.split(title, title_budget))
    title_fragment = title_parts[0] if title_parts else ""
    while title_fragment and counter.count(f"Document: {title_fragment}") > STRUCTURAL_CONTEXT_TOKEN_LIMIT:
        low, high = 1, len(title_fragment) - 1
        best = ""
        while low <= high:
            middle = (low + high) // 2
            candidate = title_fragment[:middle].rstrip()
            if candidate and counter.count(f"Document: {candidate}") <= STRUCTURAL_CONTEXT_TOKEN_LIMIT:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        title_fragment = best
    if not title_fragment:
        raise ValueError("tokenizer cannot fit a document title in the structural context budget")
    prefix = f"Document: {title_fragment}"
    clean_path = [str(value).strip() for value in heading_path if str(value).strip()]
    for start in range(len(clean_path)):
        candidate = f"{prefix}\nSection: {' > '.join(clean_path[start:])}"
        if counter.count(candidate) <= STRUCTURAL_CONTEXT_TOKEN_LIMIT:
            prefix = candidate
            break
    return prefix


def pack_retrieval_chunks(
    sentences: Sequence[Mapping[str, Any]],
    *,
    token_counter: TokenCounter | None = None,
    embedding_token_limit: int = DEFAULT_EMBEDDING_TOKEN_LIMIT,
    max_sentences: int = CHUNK_MAX_SENTENCES,
    document_identity: str | None = None,
    document_title: str | None = None,
) -> list[dict[str, Any]]:
    """Pack adjacent sentences without crossing structural boundaries.

    A sentence is the atomic provenance unit.  A sentence that exceeds the
    embedding budget is split independently; it is never combined with other
    sentences before splitting.
    """
    if token_counter is None:
        raise ValueError("an exact embedding token counter is required")
    counter = token_counter
    if embedding_token_limit <= 0:
        raise ValueError("embedding_token_limit must be positive")
    chunks: list[dict[str, Any]] = []
    group: list[Mapping[str, Any]] = []

    def emit(
        items: Sequence[Mapping[str, Any]],
        body: str,
        piece_index: int = 0,
        source_spans: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if not items or not body.strip():
            return
        first = items[0]
        prefix = _context_prefix(document_title, first.get("heading_path") or [], counter) + "\n"
        available = embedding_token_limit - counter.count(prefix)
        if available <= 0:
            raise ValueError("embedding token budget is exhausted by structural context")
        text = f"{prefix}{body}" if prefix else body
        if not text.strip() or counter.count(text) > embedding_token_limit:
            raise ValueError("unable to produce a nonempty embedding input within the token budget")
        sentence_ids = [str(item.get("id")) for item in items]
        source_elements = sorted({str(value) for item in items for value in (item.get("source_element_ids") or [])})
        pages = sorted({int(page) for item in items for page in (item.get("pages") or []) if isinstance(page, int) and page > 0})
        spans = [dict(span) for span in (source_spans or [span for item in items for span in (item.get("source_spans") or [])])]
        tags = sorted({str(tag) for item in items for tag in (item.get("tags") or [item.get("element_type") or item.get("label")]) if tag})
        tag_provenance: dict[str, str] = {}
        for item in items:
            provenance = item.get("tag_provenance")
            provenance = provenance if isinstance(provenance, Mapping) else {}
            for tag in (item.get("tags") or [item.get("element_type") or item.get("label")]):
                if tag:
                    tag_provenance.setdefault(str(tag), str(provenance.get(str(tag), "canonical_structure")))
        chunks.append({
            "chunk_id": stable_identity("chunk", document_identity or "document", sentence_ids, piece_index),
            "chunk_order": len(chunks),
            "body_text": body,
            "contextualized_text": text,
            "sentence_ids": sentence_ids,
            "source_element_ids": source_elements,
            "source_spans": spans,
            "section_id": first.get("section_id"),
            "table_id": first.get("table_id"),
            "pages": pages,
            "heading_path": list(first.get("heading_path") or []),
            "tags": tags,
            "tag_provenance": tag_provenance,
            "token_count": counter.count(text),
        })

    def flush() -> None:
        nonlocal group
        if not group:
            return
        first = group[0]
        prefix = _context_prefix(document_title, first.get("heading_path") or [], counter) + "\n"
        available = embedding_token_limit - counter.count(prefix)
        if available <= 0:
            raise ValueError("embedding token budget is exhausted by structural context")
        body = " ".join(str(item.get("text") or "").strip() for item in group).strip()
        if counter.count(body) <= available:
            emit(group, body)
        else:
            # This path is only reachable for a single oversized sentence;
            # normal packing flushes before adding an overflowing sentence.
            if len(group) != 1:
                raise ValueError("chunk packer attempted to split multiple sentences")
            item = group[0]
            splitter = counter.split_with_spans
            if splitter is not None:
                pieces = list(splitter(body, available))
            else:
                pieces = []
                cursor = 0
                for piece in counter.split(body, available):
                    start = body.find(piece, cursor)
                    if start < 0:
                        start = cursor
                    end = start + len(piece)
                    pieces.append((piece, start, end))
                    cursor = end
            bounded_pieces: list[tuple[str, int, int]] = []
            for piece, start, end in pieces:
                if counter.count(f"{prefix}{piece}") <= embedding_token_limit:
                    bounded_pieces.append((piece, start, end))
                    continue
                cursor = 0
                while cursor < len(piece):
                    low, high = cursor + 1, len(piece)
                    best = cursor
                    while low <= high:
                        middle = (low + high) // 2
                        if counter.count(f"{prefix}{piece[cursor:middle]}") <= embedding_token_limit:
                            best = middle
                            low = middle + 1
                        else:
                            high = middle - 1
                    if best == cursor:
                        raise ValueError("tokenizer cannot fit a fragment with its structural context")
                    bounded_pieces.append((piece[cursor:best], start + cursor, start + best))
                    cursor = best
            pieces = bounded_pieces
            for piece_index, (piece, start, end) in enumerate(pieces):
                original_spans = list(item.get("source_spans") or [])
                fragment_spans: list[dict[str, Any]] = []
                fragment_element_ids: set[str] = set()
                fragment_pages: set[int] = set()
                for span_value in original_spans:
                    span = dict(span_value)
                    span_start = int(span.get("start", 0))
                    span_end = int(span.get("end", span_start))
                    # ``start``/``end`` are sentence-relative, while the
                    # persisted source span is element-relative. Translate
                    # the fragment before intersecting with the source span.
                    overlap_start = max(span_start, span_start + start)
                    overlap_end = min(span_end, span_start + end)
                    if overlap_start >= overlap_end:
                        continue
                    span["start"] = overlap_start
                    span["end"] = overlap_end
                    fragment_spans.append(span)
                    if span.get("element_id"):
                        fragment_element_ids.add(str(span["element_id"]))
                for page in item.get("pages") or []:
                    if isinstance(page, int) and page > 0:
                        fragment_pages.add(page)
                fragment = dict(item)
                fragment["source_spans"] = fragment_spans
                fragment["source_element_ids"] = sorted(fragment_element_ids)
                fragment["pages"] = sorted(fragment_pages)
                emit([fragment], piece, piece_index, source_spans=fragment_spans)
        group = []

    previous_section: Any = object()
    previous_table: Any = object()
    previous_paragraph: Any = object()
    for sentence in sentences:
        section = sentence.get("section_id")
        table = sentence.get("table_id")
        paragraph = sentence.get("paragraph_id")
        structural_change = section != previous_section or table != previous_table or paragraph != previous_paragraph
        if group and (structural_change or len(group) >= max_sentences):
            flush()
        if group:
            prefix = _context_prefix(document_title, group[0].get("heading_path") or [], counter) + "\n"
            candidate = " ".join([*(str(item.get("text") or "").strip() for item in group), str(sentence.get("text") or "").strip()]).strip()
            if counter.count(prefix) + counter.count(candidate) > embedding_token_limit:
                flush()
        if not group and counter.count(str(sentence.get("text") or "").strip()) + counter.count(
            _context_prefix(document_title, sentence.get("heading_path") or [], counter) + "\n"
        ) > embedding_token_limit:
            group.append(sentence)
            flush()
            previous_section, previous_table, previous_paragraph = section, table, paragraph
            continue
        group.append(sentence)
        previous_section = section
        previous_table = table
        previous_paragraph = paragraph
    flush()
    return chunks


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "CHUNK_MAX_SENTENCES",
    "DEFAULT_EMBEDDING_TOKEN_LIMIT",
    "EXTRACTION_PIPELINE_VERSION",
    "READING_EXCLUDED_LABELS",
    "RETRIEVAL_CHUNKING_VERSION",
    "STRUCTURAL_CONTEXT_TOKEN_LIMIT",
    "TokenCounter",
    "build_canonical_payload",
    "derive_hierarchy",
    "pack_retrieval_chunks",
    "project_sentences",
    "reading_element_policy",
    "stable_fingerprint",
    "stable_identity",
    "stable_source_id",
    "SentencePipelineUnavailableError",
    "sentence_pipeline_identity",
]
