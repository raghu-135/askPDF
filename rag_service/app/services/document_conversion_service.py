"""Shared durable PDF conversion and reading projection service."""

from __future__ import annotations

import hashlib
import re
import asyncio
import logging
from collections import defaultdict
from typing import Any, Mapping, Sequence

from app.db.repositories.canonical_document_repo import get_canonical_document_repo
from app.services.document_pipeline import (
    CANONICAL_SCHEMA_VERSION,
    EXTRACTION_PIPELINE_VERSION,
    build_canonical_payload,
    derive_hierarchy,
    is_valid_canonical_payload,
    project_sentences,
    stable_fingerprint,
    stable_identity,
    sentence_pipeline_identity,
)
from app.services.parsing_service import DoclingParseResult, extraction_configuration, parse_with_docling


logger = logging.getLogger(__name__)


def conversion_filename(file_name: str | None) -> str:
    """Return a safe PDF filename for Docling while retaining source metadata."""
    raw = str(file_name or "document.pdf").strip()
    raw = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "document"
    return raw if raw.lower().endswith(".pdf") else f"{raw}.pdf"


_ALIGNMENT_CHAR_MAP: dict[str, str] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2033": '"',
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u00ad": "",
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u00a0": " ",
    "\u2022": "-",
}


def _normalize_alignment_text(text: str) -> tuple[str, list[int]]:
    """Strip whitespace and map unicode for deterministic cell alignment."""
    out: list[str] = []
    index_map: list[int] = []
    for index, char in enumerate(str(text or "")):
        replacement = _ALIGNMENT_CHAR_MAP.get(char, char)
        for mapped in replacement:
            if mapped.isspace():
                continue
            out.append(mapped)
            index_map.append(index)
    return "".join(out), index_map


def _bbox_has_area(bbox: Any) -> bool:
    if not isinstance(bbox, Mapping):
        return False
    try:
        x0 = float(bbox.get("x0", bbox.get("x", 0)))
        y0 = float(bbox.get("y0", bbox.get("y", 0)))
        x1 = float(bbox.get("x1", x0))
        y1 = float(bbox.get("y1", y0))
    except (TypeError, ValueError):
        return False
    return (x1 - x0) > 0.1 and (y1 - y0) > 0.1


def _sentence_has_bbox(sentence: Mapping[str, Any]) -> bool:
    for key in ("bboxes", "bbox"):
        raw = sentence.get(key)
        if isinstance(raw, list):
            if any(_bbox_has_area(item) for item in raw):
                return True
        elif _bbox_has_area(raw):
            return True
    return False


def _convert_bottomleft_to_topleft(bbox: tuple[float, float, float, float], page_height: float) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox
    return (left, page_height - top, right, page_height - bottom)


def _positive_dimension(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _merge_page_dimensions(*maps: Mapping[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    merged: dict[int, tuple[float, float]] = {}
    for mapping in maps:
        for page, size in mapping.items():
            try:
                page_number = int(page)
            except (TypeError, ValueError):
                continue
            if page_number <= 0:
                continue
            width = _positive_dimension(size[0])
            height = _positive_dimension(size[1])
            if width and height:
                merged[page_number] = (width, height)
    return merged


def _page_heights_from_dimensions(page_dimensions: Mapping[int, tuple[float, float]]) -> dict[int, float]:
    return {page: height for page, (_, height) in page_dimensions.items()}


def _bbox_within_page(bbox: Mapping[str, Any], page_width: float, page_height: float) -> bool:
    if not _bbox_has_area(bbox):
        return False
    try:
        x0 = float(bbox.get("x0", bbox.get("x", 0)))
        y0 = float(bbox.get("y0", bbox.get("y", 0)))
        x1 = float(bbox.get("x1", x0))
        y1 = float(bbox.get("y1", y0))
    except (TypeError, ValueError):
        return False
    return (
        0 <= x0 <= page_width
        and 0 <= x1 <= page_width
        and 0 <= y0 <= page_height
        and 0 <= y1 <= page_height
        and x0 <= x1
        and y0 <= y1
    )


def _stamp_bbox_page_dimensions(
    bbox: Any,
    *,
    default_page: int | None,
    page_dimensions: Mapping[int, tuple[float, float]],
) -> dict[str, Any] | None:
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            return None
        page = default_page or 0
        stamped: dict[str, Any] = {"page": page, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
    elif isinstance(bbox, Mapping):
        stamped = dict(bbox)
        page_raw = stamped.get("page", default_page)
        try:
            page = int(page_raw)
        except (TypeError, ValueError):
            page = 0
        stamped["page"] = page
    else:
        return None
    size = page_dimensions.get(int(stamped.get("page") or 0))
    if not size:
        return None
    page_width, page_height = size
    stamped["page_width"] = page_width
    stamped["page_height"] = page_height
    if not _bbox_within_page(stamped, page_width, page_height):
        return None
    return stamped


def _apply_page_dimensions_to_sentences(
    sentences: list[dict[str, Any]],
    page_dimensions: Mapping[int, tuple[float, float]],
) -> None:
    for sentence in sentences:
        default_page = sentence.get("page")
        try:
            default_page_number = int(default_page) if default_page not in (None, "") else None
        except (TypeError, ValueError):
            default_page_number = None
        if default_page_number is None:
            pages = sentence.get("pages") or []
            if pages:
                try:
                    default_page_number = int(pages[0])
                except (TypeError, ValueError):
                    default_page_number = None

        stamped_boxes: list[dict[str, Any]] = []
        for raw in sentence.get("bboxes") or []:
            stamped = _stamp_bbox_page_dimensions(
                raw,
                default_page=default_page_number,
                page_dimensions=page_dimensions,
            )
            if stamped is not None:
                stamped_boxes.append(stamped)
        if stamped_boxes:
            sentence["bboxes"] = stamped_boxes
            sentence["bbox"] = stamped_boxes[0]
            sentence["page"] = stamped_boxes[0].get("page")
        elif sentence.get("bbox") not in (None, "", []):
            stamped = _stamp_bbox_page_dimensions(
                sentence.get("bbox"),
                default_page=default_page_number,
                page_dimensions=page_dimensions,
            )
            if stamped is not None:
                sentence["bbox"] = stamped
                sentence["bboxes"] = [stamped]
                sentence["page"] = stamped.get("page")
            else:
                sentence.pop("bbox", None)
                sentence["bboxes"] = []

        primary_page = sentence.get("page")
        try:
            page_number = int(primary_page)
        except (TypeError, ValueError):
            page_number = 0
        size = page_dimensions.get(page_number)
        if size:
            sentence["page_width"] = size[0]
            sentence["page_height"] = size[1]


def _bboxes_from_element_provenance(
    element: Mapping[str, Any],
    *,
    page_dimensions: Mapping[int, tuple[float, float]],
    default_page_height: float = 792.0,
) -> list[dict[str, Any]]:
    bboxes: list[dict[str, Any]] = []
    for prov in element.get("provenance") or []:
        if not isinstance(prov, Mapping):
            continue
        page_no = prov.get("page_no")
        bbox_data = prov.get("bbox")
        if page_no in (None, "") or not isinstance(bbox_data, Mapping):
            continue
        try:
            page = int(page_no)
            left = float(bbox_data["l"])
            top = float(bbox_data["t"])
            right = float(bbox_data["r"])
            bottom = float(bbox_data["b"])
        except (KeyError, TypeError, ValueError):
            continue
        page_height = float(page_dimensions.get(page, (0.0, default_page_height))[1])
        page_width = float(page_dimensions.get(page, (612.0, default_page_height))[0])
        x0, y0, x1, y1 = _convert_bottomleft_to_topleft((left, top, right, bottom), page_height)
        if (x1 - x0) <= 0.1 or (y1 - y0) <= 0.1:
            continue
        candidate = {"page": page, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
        stamped = _stamp_bbox_page_dimensions(
            candidate,
            default_page=page,
            page_dimensions=page_dimensions,
        )
        if stamped is not None:
            bboxes.append(stamped)
    return bboxes


def _apply_provenance_bbox_fallback(
    sentences: list[dict[str, Any]],
    *,
    elements_by_id: Mapping[str, Mapping[str, Any]],
    page_dimensions: Mapping[int, tuple[float, float]],
) -> None:
    for sentence in sentences:
        if _sentence_has_bbox(sentence):
            continue
        element_ids = sentence.get("source_element_ids") or []
        element = elements_by_id.get(str(element_ids[0])) if element_ids else None
        if element is None:
            continue
        bboxes = _bboxes_from_element_provenance(element, page_dimensions=page_dimensions)
        if not bboxes:
            continue
        sentence["bboxes"] = bboxes
        sentence["bbox"] = bboxes[0]
        sentence["page"] = bboxes[0].get("page")
        sentence["page_width"] = bboxes[0].get("page_width")
        sentence["page_height"] = bboxes[0].get("page_height")
        sentence["alignment_precision"] = "coarse"


def _pages(value: Mapping[str, Any]) -> set[int]:
    values = list(value.get("pages") or [])
    if value.get("page") not in (None, ""):
        values.append(value.get("page"))
    result: set[int] = set()
    for page in values:
        try:
            number = int(page)
        except (TypeError, ValueError):
            continue
        if number > 0:
            result.add(number)
    return result


def _provenance_box_topleft(
    prov: Mapping[str, Any],
    page_height: float,
) -> tuple[float, float, float, float, int] | None:
    bbox_data = prov.get("bbox")
    if not isinstance(bbox_data, Mapping):
        return None
    try:
        page = int(prov.get("page_no"))
        left = float(bbox_data["l"])
        top = float(bbox_data["t"])
        right = float(bbox_data["r"])
        bottom = float(bbox_data["b"])
    except (KeyError, TypeError, ValueError):
        return None
    origin = str(bbox_data.get("coord_origin") or "BOTTOMLEFT").upper()
    if origin == "BOTTOMLEFT":
        x0, y0, x1, y1 = _convert_bottomleft_to_topleft((left, top, right, bottom), page_height)
    else:
        x0, y0, x1, y1 = left, top, right, bottom
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1, page


def _resolve_docling_ref(ref: str, docling_json: Mapping[str, Any]) -> Mapping[str, Any] | None:
    parts = ref.lstrip("#/").split("/")
    if len(parts) != 2:
        return None
    collection = docling_json.get(parts[0])
    if not isinstance(collection, list):
        return None
    try:
        item = collection[int(parts[1])]
    except (IndexError, TypeError, ValueError):
        return None
    return item if isinstance(item, Mapping) else None


def _provenance_from_raw_tree(raw: Mapping[str, Any], docling_json: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [
        dict(prov)
        for prov in raw.get("prov") or []
        if isinstance(prov, Mapping) and prov.get("bbox")
    ]
    if entries:
        return entries
    for child in raw.get("children") or []:
        if not isinstance(child, Mapping):
            continue
        ref = child.get("$ref")
        if not ref:
            continue
        resolved = _resolve_docling_ref(str(ref), docling_json)
        if resolved is not None:
            entries.extend(_provenance_from_raw_tree(resolved, docling_json))
    return entries


def _ensure_elements_have_provenance(
    elements: Sequence[Mapping[str, Any]],
    docling_json: Mapping[str, Any],
) -> None:
    for element in elements:
        if any(isinstance(prov, Mapping) and prov.get("bbox") for prov in element.get("provenance") or []):
            continue
        raw = element.get("raw")
        if not isinstance(raw, Mapping):
            continue
        recovered = _provenance_from_raw_tree(raw, docling_json)
        if recovered:
            element["provenance"] = recovered


def _element_alignment_markers(element: Mapping[str, Any], docling_json: Mapping[str, Any]) -> list[str]:
    raw = element.get("raw")
    if not isinstance(raw, Mapping):
        return []
    explicit = raw.get("_alignment_markers")
    if isinstance(explicit, list):
        return [str(value) for value in explicit if str(value).strip()]

    def walk(node: Mapping[str, Any]) -> list[str]:
        markers: list[str] = []
        marker = node.get("marker")
        if marker not in (None, ""):
            markers.append(str(marker))
        for child in node.get("children") or []:
            if not isinstance(child, Mapping):
                continue
            ref = child.get("$ref")
            if ref:
                resolved = _resolve_docling_ref(str(ref), docling_json)
                if resolved is not None:
                    markers.extend(walk(resolved))
        return markers

    return [value for value in walk(raw) if value.strip()]


def _assign_word_cells_to_elements(
    elements: Sequence[Mapping[str, Any]],
    word_cells_by_page: Mapping[int, Sequence[Mapping[str, Any]]],
    page_sizes: Mapping[int, tuple[float, float]],
) -> dict[str, list[dict[str, Any]]]:
    boxes_by_page: dict[int, list[tuple[float, str, float, float, float, float]]] = defaultdict(list)
    for element in elements:
        if str(element.get("element_type") or "").casefold() == "table":
            continue
        element_id = str(element.get("element_id") or "")
        if not element_id:
            continue
        for prov in element.get("provenance") or []:
            if not isinstance(prov, Mapping):
                continue
            page_no = prov.get("page_no")
            try:
                page = int(page_no)
            except (TypeError, ValueError):
                continue
            page_height = float(page_sizes.get(page, (612.0, 792.0))[1])
            bounds = _provenance_box_topleft(prov, page_height)
            if bounds is None:
                continue
            x0, y0, x1, y1, _page = bounds
            area = (x1 - x0) * (y1 - y0)
            boxes_by_page[page].append((area, element_id, x0, x1, y0, y1))

    owned: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page, cells in word_cells_by_page.items():
        for cell in cells:
            cx = (float(cell["x0"]) + float(cell["x1"])) / 2.0
            cy = (float(cell["top"]) + float(cell["bottom"])) / 2.0
            hits = sorted(
                hit
                for hit in boxes_by_page.get(int(page), [])
                if hit[2] <= cx <= hit[3] and hit[4] <= cy <= hit[5]
            )
            if not hits:
                continue
            owner = hits[0][1]
            owned[owner].append(dict(cell, page=int(page)))
    return owned


def _group_cells_into_lines(
    cells: Sequence[Mapping[str, Any]],
    page_sizes: Mapping[int, tuple[float, float]],
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for cell in cells:
        page = int(cell["page"])
        width, height = page_sizes.get(page, (612.0, 792.0))
        cell_height = float(cell["bottom"]) - float(cell["top"])
        current = lines[-1] if lines else None
        if (
            current
            and int(current["page"]) == page
            and float(cell["x0"]) >= float(current["x1"]) - 1.0
            and abs(float(cell["top"]) - float(current["y0"])) < 0.5 * cell_height
        ):
            current["x1"] = max(float(current["x1"]), float(cell["x1"]))
            current["y1"] = max(float(current["y1"]), float(cell["bottom"]))
            current["y0"] = min(float(current["y0"]), float(cell["top"]))
            continue
        lines.append(
            {
                "page": page,
                "x0": float(cell["x0"]),
                "y0": float(cell["top"]),
                "x1": float(cell["x1"]),
                "y1": float(cell["bottom"]),
                "page_width": width,
                "page_height": height,
            }
        )
    return lines


def _align_sentence_to_owned_cells(
    sentence: Mapping[str, Any],
    element: Mapping[str, Any],
    owned_cells: Sequence[Mapping[str, Any]],
    *,
    docling_json: Mapping[str, Any],
    page_sizes: Mapping[int, tuple[float, float]],
) -> list[dict[str, Any]] | None:
    if not owned_cells:
        return None
    element_text = str(element.get("text") or "").strip()
    element_norm, element_index_map = _normalize_alignment_text(element_text)
    if not element_norm:
        return None

    cell_norm_parts: list[str] = []
    cell_char_to_cell: list[int] = []
    cell_final_hyphen: list[bool] = []
    for cell_index, cell in enumerate(owned_cells):
        cell_norm, _ = _normalize_alignment_text(str(cell.get("text") or ""))
        cell_norm_parts.append(cell_norm)
        cell_final_hyphen.extend([False] * (len(cell_norm) - 1) + ([cell_norm.endswith("-")] if cell_norm else []))
        cell_char_to_cell.extend([cell_index] * len(cell_norm))
    cells_norm = "".join(cell_norm_parts)
    if not cells_norm:
        return None

    pending_markers = [
        _normalize_alignment_text(marker)[0]
        for marker in _element_alignment_markers(element, docling_json)
        if _normalize_alignment_text(marker)[0]
    ]

    char_to_cell_index: list[int | None] = [None] * len(element_norm)
    element_pos = 0
    cell_pos = 0
    while element_pos < len(element_norm) and cell_pos < len(cells_norm):
        if element_norm[element_pos] == cells_norm[cell_pos]:
            char_to_cell_index[element_pos] = cell_char_to_cell[cell_pos]
            element_pos += 1
            cell_pos += 1
        elif cell_final_hyphen[cell_pos]:
            cell_pos += 1
        elif pending_markers and cells_norm.startswith(pending_markers[0], cell_pos):
            cell_pos += len(pending_markers.pop(0))
        else:
            return None
    if element_pos < len(element_norm):
        return None

    spans = sentence.get("source_spans") or []
    if not spans:
        return None
    span = spans[0] if isinstance(spans[0], Mapping) else None
    if span is None:
        return None
    try:
        span_start = int(span["start"])
        span_end = int(span["end"])
    except (KeyError, TypeError, ValueError):
        return None

    norm_start = next((index for index, original in enumerate(element_index_map) if original >= span_start), None)
    norm_end = max(
        (index for index, original in enumerate(element_index_map) if original < span_end),
        default=None,
    )
    if norm_start is None or norm_end is None:
        return None
    start_cell = char_to_cell_index[norm_start]
    end_cell = char_to_cell_index[norm_end]
    if start_cell is None or end_cell is None:
        return None
    used_cells = list(owned_cells[start_cell : end_cell + 1])
    return _group_cells_into_lines(used_cells, page_sizes)


def _align_sentences_to_cells(
    canonical_sentences: list[dict[str, Any]],
    parse_result: DoclingParseResult,
    *,
    elements_by_id: Mapping[str, Mapping[str, Any]],
    docling_json: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Attach Docling word-cell coordinates without changing canonical sentence text."""
    page_sizes = dict(parse_result.page_sizes or {})
    elements = list(elements_by_id.values())
    docling = docling_json if isinstance(docling_json, Mapping) else {}
    _ensure_elements_have_provenance(elements, docling)
    owned_cells = _assign_word_cells_to_elements(elements, parse_result.word_cells_by_page, page_sizes)

    for sentence in canonical_sentences:
        if str(sentence.get("element_type") or "").casefold() == "table":
            continue
        element_ids = sentence.get("source_element_ids") or []
        element_id = str(element_ids[0]) if element_ids else ""
        element = elements_by_id.get(element_id)
        if element is None:
            sentence.setdefault("alignment_precision", "coarse")
            continue
        lines = _align_sentence_to_owned_cells(
            sentence,
            element,
            owned_cells.get(element_id, []),
            docling_json=docling,
            page_sizes=page_sizes,
        )
        if not lines:
            sentence.setdefault("alignment_precision", "coarse")
            continue
        sentence["bboxes"] = lines
        sentence["bbox"] = lines[0]
        sentence["page"] = lines[0].get("page")
        sentence["page_width"] = lines[0].get("page_width")
        sentence["page_height"] = lines[0].get("page_height")
        pages = sorted({int(line["page"]) for line in lines})
        if pages:
            sentence["pages"] = pages
        sentence["alignment_precision"] = "exact"

    _apply_provenance_bbox_fallback(
        canonical_sentences,
        elements_by_id=elements_by_id,
        page_dimensions=page_sizes,
    )
    if page_sizes:
        _apply_page_dimensions_to_sentences(canonical_sentences, page_sizes)
    return canonical_sentences


def current_extraction_fingerprint(data: bytes, *, merge_multi_bbox: bool = True) -> str:
    """Build the cache key from content, parser settings, and package versions."""
    return stable_fingerprint(
        hashlib.sha256(data).hexdigest(),
        current_extraction_contract_fingerprint(merge_multi_bbox=merge_multi_bbox),
    )


def current_extraction_contract_fingerprint(*, merge_multi_bbox: bool = True) -> str:
    """Fingerprint parser/schema inputs without reading a PDF."""
    return stable_fingerprint(
        EXTRACTION_PIPELINE_VERSION,
        CANONICAL_SCHEMA_VERSION,
        extraction_configuration(sentence_model=sentence_pipeline_identity()),
        merge_multi_bbox,
    )


async def enqueue_pdf_conversion(
    *,
    file_hash: str,
    data: bytes,
    file_name: str,
    merge_multi_bbox: bool = True,
    force_rebuild: bool = False,
    retry_failed: bool = False,
):
    """Persist a conversion target; rag-service drains it in-process."""
    fingerprint = current_extraction_fingerprint(data, merge_multi_bbox=merge_multi_bbox)
    generation = stable_identity("conversion", file_hash, fingerprint)
    repo = get_canonical_document_repo()
    existing = await repo.get(file_hash)
    same_target_invalid = bool(
        existing
        and existing.status == "completed"
        and existing.extraction_fingerprint == fingerprint
        and (
            not is_valid_canonical_payload(existing.document_json)
            or dict(existing.source_metadata_json or {}).get("_file_hash") != file_hash
            or dict(existing.source_metadata_json or {}).get("_extraction_contract_fingerprint")
            != current_extraction_contract_fingerprint(merge_multi_bbox=merge_multi_bbox)
            or dict(existing.source_metadata_json or {}).get("_extraction_pipeline_version")
            != EXTRACTION_PIPELINE_VERSION
        )
    )
    return await repo.ensure_conversion_job(
        file_hash=file_hash,
        generation=generation,
        extraction_fingerprint=fingerprint,
        force_rebuild=force_rebuild or same_target_invalid,
        retry_failed=retry_failed,
    )


async def convert_pdf_and_project(
    *,
    file_hash: str,
    data: bytes,
    file_name: str,
    source_metadata: Mapping[str, Any] | None = None,
    merge_multi_bbox: bool = True,
    claim_token: str | None = None,
) -> dict[str, Any]:
    """Convert once, persist the canonical representation, and return reading data."""
    source = dict(source_metadata or {})
    source.setdefault("original_title", file_name)
    source.setdefault("_file_hash", file_hash)
    source.setdefault(
        "_extraction_contract_fingerprint",
        current_extraction_contract_fingerprint(merge_multi_bbox=merge_multi_bbox),
    )
    source.setdefault("_extraction_pipeline_version", EXTRACTION_PIPELINE_VERSION)
    fingerprint = current_extraction_fingerprint(data, merge_multi_bbox=merge_multi_bbox)
    import docling
    generation = stable_identity("conversion", file_hash, fingerprint)
    repo = get_canonical_document_repo()
    existing = await repo.get(file_hash)
    existing_metadata = dict(existing.source_metadata_json or {}) if existing else {}
    existing_payload_valid = bool(
        existing
        and is_valid_canonical_payload(existing.document_json)
        and existing_metadata.get("_file_hash") == file_hash
        and existing_metadata.get("_extraction_contract_fingerprint")
        == current_extraction_contract_fingerprint(merge_multi_bbox=merge_multi_bbox)
        and existing_metadata.get("_extraction_pipeline_version") == EXTRACTION_PIPELINE_VERSION
    )
    claim_token = await repo.claim_conversion(
        file_hash,
        fingerprint,
        generation,
        force_rebuild=bool(existing and existing.status == "completed" and not existing_payload_valid),
        claim_token=claim_token,
    )
    if not claim_token:
        existing = await repo.get(file_hash)
        if existing and existing.status == "completed" and existing.extraction_fingerprint == fingerprint and is_valid_canonical_payload(existing.document_json):
            reading = existing.document_json.get("reading_projection") if isinstance(existing.document_json, dict) else None
            return {
                "version": "2.0",
                "sentences": reading if isinstance(reading, list) else project_sentences(existing.document_json),
                "generation": existing.generation,
                "extraction_fingerprint": existing.extraction_fingerprint,
            }
        if existing and existing.status == "running":
            for _ in range(1200):
                await asyncio.sleep(0.5)
                existing = await repo.get(file_hash)
                if existing and existing.status == "completed" and existing.extraction_fingerprint == fingerprint and is_valid_canonical_payload(existing.document_json):
                    reading = existing.document_json.get("reading_projection") if isinstance(existing.document_json, dict) else None
                    return {
                        "version": "2.0",
                        "sentences": reading if isinstance(reading, list) else project_sentences(existing.document_json),
                        "generation": existing.generation,
                        "extraction_fingerprint": existing.extraction_fingerprint,
                    }
                if existing and existing.status == "failed":
                    raise RuntimeError(str(existing.failure_json or "canonical conversion failed"))
            raise TimeoutError(f"timed out waiting for canonical conversion of {file_hash}")

    heartbeat_stop = asyncio.Event()

    async def renew_claim() -> None:
        interval = 300
        while not heartbeat_stop.is_set():
            try:
                await asyncio.wait_for(heartbeat_stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                try:
                    if not await repo.renew_conversion_claim(file_hash, claim_token):
                        return
                except Exception as exc:
                    logger.warning("Could not renew conversion claim for %s: %s", file_hash, exc)

    heartbeat_task = asyncio.create_task(renew_claim())
    try:
        safe_name = conversion_filename(file_name)
        parse_result = await asyncio.to_thread(parse_with_docling, data, safe_name)
        if parse_result is None:
            raise RuntimeError("Docling conversion returned no document")
        payload = build_canonical_payload(
            parse_result.document,
            filename=safe_name,
            source_metadata=source,
            document_identity=file_hash,
        )
        sections, elements = derive_hierarchy(payload)
        payload["sections"] = sections
        payload["elements"] = elements

        sentences = project_sentences(payload)
        elements_by_id = {
            str(item.get("element_id")): item
            for item in elements
            if isinstance(item, Mapping) and item.get("element_id")
        }
        sentences = _align_sentences_to_cells(
            sentences,
            parse_result,
            elements_by_id=elements_by_id,
            docling_json=payload.get("docling") if isinstance(payload.get("docling"), Mapping) else {},
        )
        payload["reading_projection"] = sentences
        published = await repo.complete_conversion(
            file_hash=file_hash,
            claim_token=claim_token,
            generation=generation,
            fingerprint=fingerprint,
            docling_version=str(getattr(docling, "__version__", "unknown")),
            document_json=payload,
            source_metadata=source,
            sections=sections,
            elements=elements,
        )
        if not published:
            # A stale worker may finish after its claim was reclaimed.  Return
            # the newer canonical result and never publish the old payload.
            current = await repo.get(file_hash)
            if current and current.status == "completed":
                current_json = current.document_json if isinstance(current.document_json, dict) else {}
                return {
                    "version": "2.0",
                    "sentences": current_json.get("reading_projection") or project_sentences(current_json),
                    "generation": current.generation,
                    "extraction_fingerprint": current.extraction_fingerprint,
                }
            raise RuntimeError("canonical conversion claim was lost before publication")

        return {"version": "2.0", "sentences": sentences, "generation": generation, "extraction_fingerprint": fingerprint}
    except Exception as exc:
        await repo.fail_conversion(
            file_hash,
            {"code": "conversion_failed", "message": str(exc), "type": type(exc).__name__},
            claim_token=claim_token,
        )
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


__all__ = [
    "_align_sentences_to_cells",
    "_apply_page_dimensions_to_sentences",
    "_assign_word_cells_to_elements",
    "_bbox_within_page",
    "_bboxes_from_element_provenance",
    "_group_cells_into_lines",
    "_normalize_alignment_text",
    "conversion_filename",
    "convert_pdf_and_project",
    "current_extraction_contract_fingerprint",
    "current_extraction_fingerprint",
]
