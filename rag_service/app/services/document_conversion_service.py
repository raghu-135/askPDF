"""Shared durable PDF conversion and reading projection service."""

from __future__ import annotations

import hashlib
import re
import asyncio
import logging
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Mapping

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
from app.services.document_extraction_contract import extraction_configuration
from app.services.parsing_service import parse_with_docling, parse_with_pdfplumber


logger = logging.getLogger(__name__)


def conversion_filename(file_name: str | None) -> str:
    """Return a safe PDF filename for Docling while retaining source metadata."""
    raw = str(file_name or "document.pdf").strip()
    raw = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "document"
    return raw if raw.lower().endswith(".pdf") else f"{raw}.pdf"


def _normalise(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


_LIGATURE_REPLACEMENTS = (
    ("ﬁ", "fi"),
    ("ﬂ", "fl"),
    ("ﬀ", "ff"),
    ("ﬃ", "ffi"),
    ("ﬄ", "ffl"),
)


def _normalize_match_text(text: str) -> str:
    """Normalize text for near-exact sentence alignment (whitespace, unicode, hyphenation)."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    for source, target in _LIGATURE_REPLACEMENTS:
        value = value.replace(source, target)
    value = value.replace("\u00ad", "")
    value = re.sub(r"-\s*\n\s*", "", value)
    value = re.sub(r"-\s+", "", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*([,.;:!?])\s*", r"\1", value)
    return value.casefold().strip()


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


def _page_heights_from_candidates(candidates: list[dict[str, Any]]) -> dict[int, float]:
    heights: dict[int, float] = {}
    for candidate in candidates:
        page_height = candidate.get("page_height")
        if page_height in (None, ""):
            continue
        for page in _pages(candidate):
            heights[page] = float(page_height)
        page = candidate.get("page")
        if page not in (None, ""):
            try:
                heights[int(page)] = float(page_height)
            except (TypeError, ValueError):
                pass
    return heights


def _bboxes_from_element_provenance(
    element: Mapping[str, Any],
    *,
    page_heights: Mapping[int, float],
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
        page_height = float(page_heights.get(page, default_page_height))
        x0, y0, x1, y1 = _convert_bottomleft_to_topleft((left, top, right, bottom), page_height)
        if (x1 - x0) <= 0.1 or (y1 - y0) <= 0.1:
            continue
        bboxes.append({"page": page, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return bboxes


def _apply_provenance_bbox_fallback(
    sentences: list[dict[str, Any]],
    *,
    elements_by_id: Mapping[str, Mapping[str, Any]],
    page_heights: Mapping[int, float],
) -> None:
    for sentence in sentences:
        if _sentence_has_bbox(sentence):
            continue
        element_ids = sentence.get("source_element_ids") or []
        element = elements_by_id.get(str(element_ids[0])) if element_ids else None
        if element is None:
            continue
        bboxes = _bboxes_from_element_provenance(element, page_heights=page_heights)
        if not bboxes:
            continue
        sentence["bboxes"] = bboxes
        sentence["bbox"] = bboxes[0]
        sentence["page"] = bboxes[0].get("page")
        if sentence.get("page_width") in (None, ""):
            sentence["page_width"] = element.get("page_width")
        if sentence.get("page_height") in (None, ""):
            sentence["page_height"] = element.get("page_height")
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


def _verified_text_match(target: str, candidate: str) -> bool:
    """Accept a near-exact full-sentence match after normalization."""
    left = _normalize_match_text(target)
    right = _normalize_match_text(candidate)
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    if shorter in longer and len(shorter) / max(len(longer), 1) >= 0.85:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.94


def _candidate_compatible(sentence: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    pages = _pages(sentence)
    candidate_pages = _pages(candidate)
    if pages and candidate_pages and not pages.intersection(candidate_pages):
        return False
    if sentence.get("label") and candidate.get("label") and sentence.get("label") != candidate.get("label"):
        return False
    sentence_refs = {str(value) for value in sentence.get("source_element_refs") or [] if value}
    candidate_ref = str(candidate.get("source_ref") or "")
    if sentence_refs and candidate_ref and candidate_ref not in sentence_refs:
        return False
    return True


def _merge_candidate_span(candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    texts = [str(item.get("text") or "") for item in candidates]
    merged_text = " ".join(part for part in texts if part).strip()
    bboxes: list[dict[str, Any]] = []
    for item in candidates:
        raw = item.get("bboxes") or []
        if isinstance(raw, list):
            bboxes.extend([box for box in raw if isinstance(box, Mapping)])
        elif isinstance(item.get("bbox"), Mapping):
            bboxes.append(dict(item["bbox"]))
    merged: dict[str, Any] = {"text": merged_text, "bboxes": bboxes}
    for key in ("bbox", "page_width", "page_height", "words", "font", "page", "label", "source_ref"):
        for item in candidates:
            if item.get(key) not in (None, [], ""):
                merged[key] = item[key]
                break
    pages: set[int] = set()
    for item in candidates:
        pages.update(_pages(item))
    if pages:
        merged["pages"] = sorted(pages)
        merged["page"] = sorted(pages)[0]
    return merged


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


def _align_coordinates(
    canonical_sentences: list[dict[str, Any]],
    coordinate_candidates: list[dict[str, Any]],
    *,
    elements_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    max_span: int = 6,
) -> list[dict[str, Any]]:
    """Attach pdfplumber coordinates without replacing canonical Docling text."""
    candidates = list(coordinate_candidates or [])
    page_heights = _page_heights_from_candidates(candidates)
    search_from = 0
    for sentence in canonical_sentences:
        target_text = str(sentence.get("text") or "")
        target = _normalize_match_text(target_text)
        if not target:
            sentence.setdefault("alignment_precision", "coarse")
            continue
        best: dict[str, Any] | None = None
        best_score = 0.0
        best_end = search_from
        for start in range(search_from, len(candidates)):
            span: list[Mapping[str, Any]] = []
            for end in range(start, min(start + max_span, len(candidates))):
                candidate = candidates[end]
                if not _candidate_compatible(sentence, candidate):
                    break
                span.append(candidate)
                merged = _merge_candidate_span(span)
                candidate_text = _normalize_match_text(str(merged.get("text") or ""))
                if not candidate_text:
                    continue
                if not _verified_text_match(target_text, str(merged.get("text") or "")):
                    continue
                score = SequenceMatcher(None, target, candidate_text).ratio()
                if score >= best_score:
                    best = merged
                    best_score = score
                    best_end = end
                if target == candidate_text:
                    break
            if best is not None and target == _normalize_match_text(str(best.get("text") or "")):
                break
        if best is None:
            sentence.setdefault("alignment_precision", "coarse")
            continue
        for key in ("bbox", "bboxes", "page_width", "page_height", "words", "font", "page", "pages"):
            if best.get(key) not in (None, [], ""):
                sentence[key] = best[key]
        sentence["alignment_precision"] = "exact"
        search_from = best_end + 1
    if elements_by_id:
        _apply_provenance_bbox_fallback(
            canonical_sentences,
            elements_by_id=elements_by_id,
            page_heights=page_heights,
        )
    return canonical_sentences


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
        docling_doc = await asyncio.to_thread(parse_with_docling, data, safe_name)
        if docling_doc is None:
            raise RuntimeError("Docling conversion returned no document")
        payload = build_canonical_payload(
            docling_doc,
            filename=safe_name,
            source_metadata=source,
            document_identity=file_hash,
        )
        sections, elements = derive_hierarchy(payload)
        payload["sections"] = sections
        payload["elements"] = elements

        # Persist the reading projection together with the canonical payload so
        # cache hits retain the exact pdfplumber alignment used for highlighting.
        coordinate_candidates = await asyncio.to_thread(
            parse_with_pdfplumber,
            data,
            docling_doc,
            safe_name,
            merge_multi_bbox=merge_multi_bbox,
        )
        # Docling remains authoritative for text, sentence identity, OCR, and
        # structure. pdfplumber contributes coordinates only when a candidate
        # can be aligned to an individual canonical sentence.
        sentences = project_sentences(payload)
        elements_by_id = {
            str(item.get("element_id")): item
            for item in elements
            if isinstance(item, Mapping) and item.get("element_id")
        }
        sentences = _align_coordinates(
            sentences,
            coordinate_candidates or [],
            elements_by_id=elements_by_id,
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

        # pdfplumber is used only to align words/coordinates to canonical text.
        # It never replaces the canonical text or structure.
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
    "conversion_filename",
    "convert_pdf_and_project",
    "current_extraction_contract_fingerprint",
    "current_extraction_fingerprint",
]
