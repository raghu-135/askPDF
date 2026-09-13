"""Shared durable PDF conversion and reading projection service."""

from __future__ import annotations

import hashlib
import re
import asyncio
from typing import Any, Mapping

from app.db.repositories.canonical_document_repo import get_canonical_document_repo
from app.services.document_pipeline import build_canonical_payload, derive_hierarchy, project_sentences, stable_fingerprint, stable_identity
from app.services.parsing_service import extraction_configuration, parse_with_docling, parse_with_pdfplumber


def conversion_filename(file_name: str | None) -> str:
    """Return a safe PDF filename for Docling while retaining source metadata."""
    raw = str(file_name or "document.pdf").strip()
    raw = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "document"
    return raw if raw.lower().endswith(".pdf") else f"{raw}.pdf"


def _normalise(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def current_extraction_fingerprint(data: bytes, *, merge_multi_bbox: bool = True) -> str:
    """Build the cache key from content, parser settings, and package versions."""
    import docling

    return stable_fingerprint(
        hashlib.sha256(data).hexdigest(),
        "docling-pdf-v1",
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
        pages = set(sentence.get("pages") or [])
        best = None
        best_score = 0
        for candidate in unused:
            candidate_text = _normalise(candidate.get("text", ""))
            candidate_pages = set(candidate.get("pages") or [])
            if pages and candidate_pages and not pages.intersection(candidate_pages):
                continue
            if not target or not candidate_text:
                continue
            if candidate_text in target or target in candidate_text:
                score = min(len(target), len(candidate_text))
            else:
                overlap = set(target.split()).intersection(candidate_text.split())
                score = len(overlap) * 4
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
    fingerprint = current_extraction_fingerprint(data, merge_multi_bbox=merge_multi_bbox)
    import docling
    generation = stable_identity("conversion", file_hash, fingerprint)
    repo = get_canonical_document_repo()
    if not await repo.claim_conversion(file_hash, fingerprint, generation):
        existing = await repo.get(file_hash)
        if existing and existing.status == "completed" and existing.extraction_fingerprint == fingerprint:
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
                if existing and existing.status == "completed" and existing.extraction_fingerprint == fingerprint:
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
        await repo.complete_conversion(
            file_hash=file_hash,
            generation=generation,
            fingerprint=fingerprint,
            docling_version=str(getattr(docling, "__version__", "unknown")),
            document_json=payload,
            source_metadata=source,
            sections=sections,
            elements=elements,
        )

        # pdfplumber is used only to align words/coordinates to canonical text.
        # It never replaces the canonical text or structure.
        return {"version": "2.0", "sentences": sentences, "generation": generation, "extraction_fingerprint": fingerprint}
    except Exception as exc:
        await repo.fail_conversion(file_hash, {"code": "conversion_failed", "message": str(exc), "type": type(exc).__name__})
        raise


__all__ = ["conversion_filename", "convert_pdf_and_project", "current_extraction_fingerprint"]
