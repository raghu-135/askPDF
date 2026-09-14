"""Model-specific retrieval projection over persisted canonical document data."""

from __future__ import annotations

from typing import Any, Mapping

from app.db import get_file, get_file_parsed_sentences
from app.db.repositories.canonical_document_repo import get_canonical_document_repo
from app.services.content_store import get_content_store, pdf_content_key
from app.services.document_pipeline import (
    CANONICAL_SCHEMA_VERSION,
    EXTRACTION_PIPELINE_VERSION,
    RETRIEVAL_CHUNKING_VERSION,
    is_valid_canonical_payload,
    pack_retrieval_chunks,
    project_sentences,
    stable_fingerprint,
)
from app.services.document_extraction_contract import extraction_configuration
from app.services.embedding_tokenizer import resolve_embedding_tokenizer


def retrieval_chunking_fingerprint(canonical: Any, embedding_model: str, tokenizer_fingerprint: str) -> str:
    return stable_fingerprint(
        canonical.generation,
        embedding_model,
        tokenizer_fingerprint,
        RETRIEVAL_CHUNKING_VERSION,
    )


def _repair_source_version(file_hash: str, canonical: Any = None, chunking_fingerprint: str = "") -> str:
    return stable_fingerprint(
        "document-repair-v2",
        file_hash,
        getattr(canonical, "generation", "conversion"),
        getattr(canonical, "extraction_fingerprint", ""),
        chunking_fingerprint,
    )


def _current_extraction_contract_fingerprint() -> str:
    return stable_fingerprint(
        EXTRACTION_PIPELINE_VERSION,
        CANONICAL_SCHEMA_VERSION,
        extraction_configuration(),
        True,
    )


def _metadata_for_canonical(canonical: Any, embedding_model: str | None = None, config: Any = None) -> dict[str, Any]:
    payload = canonical.document_json if isinstance(getattr(canonical, "document_json", None), dict) else {}
    reading = payload.get("reading_projection") if isinstance(payload.get("reading_projection"), list) else []
    pages = {
        int(page)
        for sentence in reading
        if isinstance(sentence, dict)
        for page in (sentence.get("pages") or [])
        if str(page).isdigit() and int(page) > 0
    }
    fingerprint = ""
    if embedding_model and config is not None:
        fingerprint = retrieval_chunking_fingerprint(canonical, embedding_model, config.fingerprint)
    return {
        "generation": canonical.generation,
        "extraction_fingerprint": canonical.extraction_fingerprint,
        "chunking_fingerprint": fingerprint,
        "tokenizer": config.identity if config is not None else None,
        "tokenizer_revision": config.revision if config is not None else None,
        "effective_input_limit": config.effective_input_limit if config is not None else None,
        "sentence_count": len(reading),
        "page_count": len(pages),
        "element_types": sorted({
            str(item.get("element_type"))
            for item in payload.get("elements") or []
            if isinstance(item, dict) and item.get("element_type")
        }),
        "repair_source_version": _repair_source_version(
            str(getattr(canonical, "file_hash", "")), canonical, fingerprint
        ),
    }


def _chunk_from_row(row: Any) -> dict[str, Any]:
    metadata = dict(getattr(row, "metadata_json", None) or {})
    pages = [int(page) for page in (metadata.get("pages") or []) if str(page).isdigit()]
    return {
        "chunk_id": str(row.chunk_id),
        "chunk_order": int(row.chunk_order or 0),
        "body_text": row.body_text,
        "contextualized_text": row.contextualized_text,
        "sentence_ids": list(row.sentence_ids or []),
        "source_element_ids": list(row.source_element_ids or []),
        "section_id": row.section_id,
        "table_id": row.table_id,
        "pages": pages,
        "heading_path": list(metadata.get("heading_path") or []),
        "token_count": metadata.get("token_count"),
        "source_spans": list(metadata.get("source_spans") or []),
        "tags": list(metadata.get("tags") or []),
    }


async def evaluate_document_freshness(
    file_hash: str,
    embedding_model: str | None = None,
    *,
    require_reading: bool = False,
    require_manifest: bool = False,
    verify_vectors: bool = False,
) -> dict[str, Any]:
    """Check persisted document state without reading or projecting the PDF."""
    repo = get_canonical_document_repo()
    canonical = await repo.get(file_hash)
    result: dict[str, Any] = {
        "file_hash": file_hash,
        "canonical": canonical,
        "canonical_ready": False,
        "reading_ready": not require_reading,
        "manifest_ready": not require_manifest,
        "vectors_ready": not verify_vectors,
        "manifest": None,
        "repair_source_version": _repair_source_version(file_hash),
    }
    if canonical is None or canonical.status != "completed":
        result.update(reason="conversion_incomplete", ready=False)
        return result

    source_metadata = dict(getattr(canonical, "source_metadata_json", None) or {})
    canonical_ready = (
        is_valid_canonical_payload(getattr(canonical, "document_json", None))
        and source_metadata.get("_file_hash") == file_hash
        and bool(source_metadata.get("_extraction_contract_fingerprint"))
        and source_metadata.get("_extraction_contract_fingerprint") == _current_extraction_contract_fingerprint()
        and source_metadata.get("_extraction_pipeline_version") == EXTRACTION_PIPELINE_VERSION
    )
    result["canonical_ready"] = canonical_ready
    result["repair_source_version"] = _repair_source_version(file_hash, canonical)
    if not canonical_ready:
        result.update(reason="canonical_stale", ready=False)
        return result

    if require_reading:
        parsed = await get_file_parsed_sentences(file_hash)
        result["reading_ready"] = bool(
            isinstance(parsed, dict)
            and parsed.get("version") == "2.0"
            and parsed.get("generation") == canonical.generation
            and parsed.get("extraction_fingerprint") == canonical.extraction_fingerprint
            and isinstance(parsed.get("sentences"), list)
        )
        if not result["reading_ready"]:
            result.update(reason="reading_stale", ready=False)
            return result

    if embedding_model and (require_manifest or verify_vectors):
        try:
            config, _counter = resolve_embedding_tokenizer(embedding_model)
        except Exception as exc:
            result.update(reason="tokenizer_unavailable", error=str(exc), ready=False)
            return result
        fingerprint = retrieval_chunking_fingerprint(canonical, embedding_model, config.fingerprint)
        manifest = await repo.get_manifest(file_hash, embedding_model, canonical.generation, fingerprint)
        result["manifest"] = manifest
        expected = list(getattr(manifest, "expected_source_ids", None) or []) if manifest else []
        manifest_ready = bool(
            manifest
            and manifest.status == "completed"
            and manifest.published_at is not None
            and manifest.superseded_at is None
            and manifest.generation == canonical.generation
            and manifest.chunking_fingerprint == fingerprint
            and manifest.expected_chunk_count == len(expected)
        )
        result["manifest_ready"] = manifest_ready
        if verify_vectors and manifest_ready:
            try:
                from app.db.vector import get_vector_db
                result["vectors_ready"] = bool(
                    await get_vector_db().has_file_indexed_chunks(
                        file_hash,
                        embedding_model,
                        expected,
                        manifest_id=manifest.manifest_id,
                    )
                    and manifest.vector_status == "completed"
                    and manifest.vector_count == len(expected)
                )
            except Exception as exc:
                result.update(reason="vector_check_failed", error=str(exc), ready=False)
                return result
        elif verify_vectors:
            result["vectors_ready"] = False

        result["metadata"] = _metadata_for_canonical(canonical, embedding_model, config)
    else:
        result["metadata"] = _metadata_for_canonical(canonical)

    result["ready"] = bool(
        result["canonical_ready"]
        and result["reading_ready"]
        and result["manifest_ready"]
        and result["vectors_ready"]
    )
    result["reason"] = "ready" if result["ready"] else "manifest_incomplete"
    return result


async def ensure_retrieval_projection(
    *,
    file_hash: str,
    embedding_model: str,
    file_name: str | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Return a complete chunk manifest, lazily repairing canonical data if needed."""
    from app.services.document_conversion_service import convert_pdf_and_project

    repo = get_canonical_document_repo()
    canonical = await repo.get(file_hash)
    store = get_content_store()
    key = pdf_content_key(file_hash)
    freshness = await evaluate_document_freshness(file_hash, embedding_model, require_manifest=True)
    data = None
    if not freshness.get("canonical_ready"):
        data = await store.read(key) if await store.exists(key) else None
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
        freshness = await evaluate_document_freshness(file_hash, embedding_model, require_manifest=True)

    config, counter = resolve_embedding_tokenizer(embedding_model)
    existing_manifest = freshness.get("manifest")
    if freshness.get("canonical_ready") and existing_manifest is not None and existing_manifest.status == "completed":
        existing_rows = await repo.get_chunks(file_hash, embedding_model, generation=canonical.generation)
        if len(existing_rows) == int(existing_manifest.expected_chunk_count or 0):
            return (
                existing_manifest,
                [_chunk_from_row(row) for row in existing_rows],
                freshness.get("metadata") or _metadata_for_canonical(canonical, embedding_model, config),
            )

    # Materialization is the only path allowed to publish the reading cache.
    from app.services.file_processing_service import publish_reading_projection
    await publish_reading_projection(
        file_hash,
        {
            "generation": canonical.generation,
            "extraction_fingerprint": canonical.extraction_fingerprint,
            "sentences": (canonical.document_json or {}).get("reading_projection", []),
        },
    )
    payload = canonical.document_json or {}
    sentences = project_sentences(payload, element_policy=None)
    chunks = pack_retrieval_chunks(
        sentences,
        token_counter=counter,
        embedding_token_limit=config.effective_input_limit,
        document_identity=f"{file_hash}:{canonical.generation}",
    )
    fingerprint = retrieval_chunking_fingerprint(canonical, embedding_model, config.fingerprint)
    manifest = await repo.get_manifest(file_hash, embedding_model, canonical.generation, fingerprint)
    if manifest is not None and manifest.status == "completed":
        existing_rows = await repo.get_chunks(file_hash, embedding_model, generation=canonical.generation)
        if len(existing_rows) == int(manifest.expected_chunk_count or 0):
            return manifest, [_chunk_from_row(row) for row in existing_rows], _metadata_for_canonical(canonical, embedding_model, config)
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
        "repair_source_version": stable_fingerprint(
            "document-repair-v1",
            file_hash,
            canonical.generation,
            canonical.extraction_fingerprint,
            fingerprint,
        ),
    }
    return manifest, chunks, metadata


async def evaluate_retrieval_readiness(file_hash: str, embedding_model: str) -> dict[str, Any]:
    """Evaluate canonical, manifest, and vector readiness as one contract."""
    freshness = await evaluate_document_freshness(
        file_hash,
        embedding_model,
        require_manifest=True,
        verify_vectors=True,
    )
    canonical = freshness.get("canonical")
    metadata = freshness.get("metadata") or {}
    manifest = freshness.get("manifest")
    return {
        "ready": bool(freshness.get("ready")),
        "reason": freshness.get("reason"),
        "manifest": manifest,
        "metadata": metadata,
        "error": freshness.get("error"),
        "repair_source_version": _repair_source_version(file_hash, canonical, metadata.get("chunking_fingerprint", "")),
    }


__all__ = [
    "ensure_retrieval_projection",
    "evaluate_document_freshness",
    "evaluate_retrieval_readiness",
    "retrieval_chunking_fingerprint",
]
