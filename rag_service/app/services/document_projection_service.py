"""Model-specific retrieval projection over persisted canonical document data."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Mapping

from app.db import get_file
from app.db.repositories.canonical_document_repo import get_canonical_document_repo
from app.services.content_store import get_content_store, pdf_content_key
from app.services.document_conversion_service import convert_pdf_and_project, current_extraction_fingerprint
from app.services.document_pipeline import pack_retrieval_chunks, project_sentences, stable_fingerprint
from app.services.embedding_tokenizer import resolve_embedding_tokenizer


async def ensure_retrieval_projection(
    *,
    file_hash: str,
    embedding_model: str,
    file_name: str | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Return a complete chunk manifest, lazily repairing canonical data if needed."""
    repo = get_canonical_document_repo()
    canonical = await repo.get(file_hash)
    store = get_content_store()
    key = pdf_content_key(file_hash)
    data = await store.read(key) if await store.exists(key) else None
    extraction_stale = False
    if canonical is not None and canonical.status == "completed":
        if data is None:
            raise FileNotFoundError(f"PDF content not found for {file_hash}; cannot validate extraction fingerprint")
        extraction_stale = canonical.extraction_fingerprint != current_extraction_fingerprint(data)
    if canonical is None or canonical.status != "completed" or extraction_stale:
        if data is None:
            raise FileNotFoundError(f"PDF content not found for {file_hash}")
        file = await get_file(file_hash)
        parsed = await convert_pdf_and_project(
            file_hash=file_hash,
            data=data,
            file_name=file_name or (file.file_name if file else f"{file_hash}.pdf"),
            source_metadata=source_metadata,
        )
        canonical = await repo.get(file_hash)
        if canonical is None or canonical.status != "completed":
            raise RuntimeError("canonical conversion did not publish a completed document")
    # Retrieval repair can perform conversion without the upload/parser task.
    # Keep the File JSON cache consumed by reading endpoints in sync with the
    # canonical generation on every projection path.
    from app.services.file_processing_service import publish_reading_projection
    await publish_reading_projection(
        file_hash,
        {
            "generation": canonical.generation,
            "extraction_fingerprint": canonical.extraction_fingerprint,
            "sentences": (canonical.document_json or {}).get("reading_projection", []),
        },
    )
    config, counter = resolve_embedding_tokenizer(embedding_model)
    payload = canonical.document_json or {}
    sentences = project_sentences(payload)
    chunks = pack_retrieval_chunks(
        sentences,
        token_counter=counter,
        embedding_token_limit=config.effective_input_limit,
        document_identity=f"{file_hash}:{canonical.generation}",
    )
    fingerprint = stable_fingerprint(
        canonical.generation,
        embedding_model,
        config.fingerprint,
        "sentence-pack-v2",
    )
    manifest = await repo.get_manifest(file_hash, embedding_model, canonical.generation, fingerprint)
    if (
        manifest is None
        or manifest.status not in {"completed", "running"}
        or manifest.expected_chunk_ids != [str(item["chunk_id"]) for item in chunks]
        or (manifest.status == "completed" and manifest.vector_status == "failed")
    ):
        manifest = await repo.replace_manifest(
            file_hash=file_hash,
            embedding_model=embedding_model,
            generation=canonical.generation,
            chunking_fingerprint=fingerprint,
            chunks=chunks,
        )
    metadata = {
        "generation": canonical.generation,
        "extraction_fingerprint": canonical.extraction_fingerprint,
        "chunking_fingerprint": fingerprint,
        "tokenizer": config.identity,
        "tokenizer_revision": config.revision,
        "effective_input_limit": config.effective_input_limit,
        "sentence_count": len(sentences),
        "page_count": len({page for sentence in sentences for page in (sentence.get("pages") or []) if page}),
        # Document-wide structural metadata is for inspection only. Chunk
        # filtering uses the per-chunk tags persisted by the repository.
        "element_types": sorted({str(item.get("element_type")) for item in payload.get("elements") or []}),
    }
    return manifest, chunks, metadata


async def evaluate_retrieval_readiness(file_hash: str, embedding_model: str) -> dict[str, Any]:
    """Evaluate canonical, manifest, and vector readiness as one contract."""
    repo = get_canonical_document_repo()
    canonical = await repo.get(file_hash)
    if canonical is None or canonical.status != "completed":
        return {"ready": False, "reason": "conversion_incomplete", "manifest": None}
    try:
        manifest, chunks, metadata = await ensure_retrieval_projection(
            file_hash=file_hash,
            embedding_model=embedding_model,
        )
    except Exception as exc:
        return {"ready": False, "reason": "projection_unavailable", "error": str(exc), "manifest": None}
    expected = list(manifest.expected_source_ids or [])
    manifest_complete = not (
        manifest.status != "completed"
        or manifest.vector_status != "completed"
        or manifest.published_at is None
        or manifest.superseded_at is not None
        or manifest.generation != canonical.generation
        or manifest.expected_chunk_count != len(chunks)
        or len(expected) != len(chunks)
    )
    from app.db.vector import get_vector_db
    try:
        vector_db = get_vector_db()
        vectors_complete = False
        if manifest_complete:
            vectors_complete = await vector_db.has_file_indexed_chunks(
                file_hash,
                embedding_model,
                expected,
                manifest_id=manifest.manifest_id,
            )
        else:
            fallback = await repo.get_ready_manifest(file_hash, embedding_model)
            if fallback is not None and fallback.manifest_id != manifest.manifest_id:
                fallback_expected = list(fallback.expected_source_ids or [])
                fallback_complete = (
                    fallback.status == "completed"
                    and fallback.vector_status == "completed"
                    and fallback.published_at is not None
                    and fallback.superseded_at is None
                    and fallback.expected_chunk_count == len(fallback_expected)
                    and bool(fallback_expected)
                )
                if fallback_complete:
                    fallback_vectors_complete = await vector_db.has_file_indexed_chunks(
                        file_hash,
                        embedding_model,
                        fallback_expected,
                        manifest_id=fallback.manifest_id,
                    )
                    if fallback_vectors_complete and fallback.vector_count == len(fallback_expected):
                        return {
                            "ready": False,
                            "reason": "repair_in_progress",
                            "manifest": manifest,
                            "fallback_manifest": fallback,
                            "fallback_ready": True,
                            "metadata": metadata,
                        }
    except Exception as exc:
        return {"ready": False, "reason": "vector_check_failed", "error": str(exc), "manifest": manifest, "metadata": metadata}
    return {
        "ready": bool(manifest_complete and vectors_complete and manifest.vector_count == len(expected)),
        "reason": "ready" if manifest_complete and vectors_complete and manifest.vector_count == len(expected) else "manifest_incomplete",
        "manifest": manifest,
        "metadata": metadata,
    }


__all__ = ["ensure_retrieval_projection", "evaluate_retrieval_readiness"]
