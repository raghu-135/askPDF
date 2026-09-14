"""Shared durable PDF conversion and reading projection service."""

from __future__ import annotations

import hashlib
import re
import asyncio
import logging
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
    """Accept only a full sentence match, allowing modest line-wrap drift."""
    if not target or not candidate:
        return False
    if target == candidate:
        return True
    shorter, longer = sorted((target, candidate), key=len)
    if len(shorter) < 24 or shorter not in longer:
        return False
    return len(shorter) / len(longer) >= 0.8 or SequenceMatcher(None, target, candidate).ratio() >= 0.92


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
        extraction_configuration(),
        merge_multi_bbox,
    )


def _align_coordinates(
    canonical_sentences: list[dict[str, Any]],
    coordinate_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach pdfplumber coordinates without replacing canonical Docling text."""
    unused = list(coordinate_candidates or [])
    for sentence in canonical_sentences:
        target = _normalise(sentence.get("text", ""))
        pages = _pages(sentence)
        best = None
        best_score = 0.0
        for candidate in unused:
            candidate_text = _normalise(candidate.get("text", ""))
            candidate_pages = _pages(candidate)
            if pages and (not candidate_pages or not pages.intersection(candidate_pages)):
                continue
            if not target or not candidate_text:
                continue
            if sentence.get("label") and candidate.get("label") and sentence.get("label") != candidate.get("label"):
                continue
            sentence_refs = {str(value) for value in sentence.get("source_element_refs") or [] if value}
            candidate_ref = str(candidate.get("source_ref") or "")
            if sentence_refs and (not candidate_ref or candidate_ref not in sentence_refs):
                continue
            if not _verified_text_match(target, candidate_text):
                continue
            score = SequenceMatcher(None, target, candidate_text).ratio()
            if score > best_score:
                best, best_score = candidate, score
        if best is None:
            sentence["alignment_precision"] = "coarse"
            continue
        for key in ("bbox", "bboxes", "page_width", "page_height", "words", "font", "page"):
            if best.get(key) not in (None, [], ""):
                sentence[key] = best[key]
        sentence["alignment_precision"] = "exact" if sentence.get("bboxes") else "coarse"
        unused.remove(best)
    return canonical_sentences


async def convert_pdf_and_project(
    *,
    file_hash: str,
    data: bytes,
    file_name: str,
    source_metadata: Mapping[str, Any] | None = None,
    merge_multi_bbox: bool = True,
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
    existing_payload_valid = bool(existing and is_valid_canonical_payload(existing.document_json))
    claim_token = await repo.claim_conversion(
        file_hash,
        fingerprint,
        generation,
        force_rebuild=bool(existing and existing.status == "completed" and not existing_payload_valid),
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
        sentences = _align_coordinates(sentences, coordinate_candidates or [])
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
