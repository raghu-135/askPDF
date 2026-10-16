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
    RETRIEVAL_SOURCE_VERSION,
    is_valid_canonical_payload,
    pack_retrieval_chunks,
    project_sentences,
    sentence_pipeline_identity,
    stable_fingerprint,
)
from app.services.document_extraction_contract import extraction_configuration
from app.services.embedding_tokenizer import EmbeddingTokenizerUnavailableError, resolve_embedding_tokenizer


class DocumentConversionPendingError(RuntimeError):
    """Raised when retrieval is requested before canonical conversion is ready."""

    def __init__(self, file_hash: str):
        self.file_hash = str(file_hash)
        super().__init__(f"document conversion is pending for {self.file_hash}")


class DocumentConversionFailedError(RuntimeError):
    """Raised when the exact conversion target has reached a stable failure."""

    def __init__(self, file_hash: str, error: str | None = None):
        self.file_hash = str(file_hash)
        self.error = str(error or "document conversion failed")
        super().__init__(f"document conversion failed for {self.file_hash}: {self.error}")


def retrieval_chunking_fingerprint(canonical: Any, embedding_model: str, tokenizer_fingerprint: str) -> str:
    return stable_fingerprint(
        canonical.generation,
        embedding_model,
        tokenizer_fingerprint,
        RETRIEVAL_CHUNKING_VERSION,
        _document_title(canonical),
    )


def _document_title(canonical: Any, fallback: str | None = None) -> str:
    payload = canonical.document_json if isinstance(getattr(canonical, "document_json", None), dict) else {}
    source_metadata = dict(getattr(canonical, "source_metadata_json", None) or {})
    return str(
        source_metadata.get("original_title")
        or payload.get("filename")
        or getattr(canonical, "file_name", None)
        or fallback
        or "Untitled document"
    )


def conversion_source_version(file_hash: str) -> str:
    """Version a conversion repair from current code and the source identity."""
    return stable_fingerprint(
        "document-conversion-v3",
        file_hash,
        _current_extraction_contract_fingerprint(),
    )


def retrieval_source_version(
    file_hash: str,
    canonical: Any,
    embedding_model: str,
    chunking_fingerprint: str,
) -> str:
    """Return the exact version a thread/document/model pointer must hold."""
    generation = str(getattr(canonical, "generation", "") or "")
    extraction_fingerprint = str(getattr(canonical, "extraction_fingerprint", "") or "")
    if not generation or not extraction_fingerprint or not chunking_fingerprint:
        raise ValueError("canonical retrieval version metadata is incomplete")
    return stable_fingerprint(
        RETRIEVAL_SOURCE_VERSION,
        file_hash,
        generation,
        extraction_fingerprint,
        embedding_model,
        chunking_fingerprint,
    )


def _current_extraction_contract_fingerprint() -> str:
    return stable_fingerprint(
        EXTRACTION_PIPELINE_VERSION,
        CANONICAL_SCHEMA_VERSION,
        extraction_configuration(sentence_model=sentence_pipeline_identity()),
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
    source_version = (
        retrieval_source_version(str(canonical.file_hash), canonical, embedding_model, fingerprint)
        if embedding_model and fingerprint
        else conversion_source_version(str(canonical.file_hash))
    )
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
        "source_version": source_version,
        "repair_source_version": source_version,
        "document_title": _document_title(canonical),
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
        "table_row_id": metadata.get("table_row_id"),
        "table_row_ids": list(metadata.get("table_row_ids") or []),
        "table_headers": list(metadata.get("table_headers") or []),
        "table_cell_ids": list(metadata.get("table_cell_ids") or []),
        "synthetic_span": bool(metadata.get("synthetic_span")),
        "pages": pages,
        "heading_path": list(metadata.get("heading_path") or []),
        "token_count": metadata.get("token_count"),
        "source_spans": list(metadata.get("source_spans") or []),
        "tags": list(metadata.get("tags") or []),
        "tag_provenance": dict(metadata.get("tag_provenance") or {}),
    }


def _manifest_rows_complete(manifest: Any, rows: list[Any], canonical: Any, embedding_model: str) -> bool:
    if (
        str(getattr(manifest, "file_hash", "")) != str(canonical.file_hash)
        or str(getattr(manifest, "embedding_model", "")) != str(embedding_model)
        or str(getattr(manifest, "generation", "")) != str(canonical.generation)
        or str(getattr(manifest, "extraction_fingerprint", "")) != str(canonical.extraction_fingerprint)
        or str(getattr(manifest, "source_version", "")) != retrieval_source_version(
            str(canonical.file_hash),
            canonical,
            embedding_model,
            str(getattr(manifest, "chunking_fingerprint", "")),
        )
    ):
        return False
    expected_chunks = [str(value) for value in (getattr(manifest, "expected_chunk_ids", None) or [])]
    expected_sources = [str(value) for value in (getattr(manifest, "expected_source_ids", None) or [])]
    if len(rows) != int(getattr(manifest, "expected_chunk_count", -1)):
        return False
    if [str(row.chunk_id) for row in rows] != expected_chunks:
        return False
    if [str(row.source_id) for row in rows] != expected_sources:
        return False
    return all(
        str(getattr(row, "manifest_id", "")) == str(manifest.manifest_id)
        and str(getattr(row, "file_hash", "")) == str(canonical.file_hash)
        and str(getattr(row, "embedding_model", "")) == str(embedding_model)
        and str(getattr(row, "generation", getattr(manifest, "generation", ""))) == str(manifest.generation)
        for row in rows
    )


async def evaluate_document_freshness(
    file_hash: str,
    embedding_model: str | None = None,
    *,
    require_reading: bool = False,
    require_manifest: bool = False,
    verify_vectors: bool = False,
    thread_id: str | None = None,
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
        "source_version": None,
        "repair_source_version": conversion_source_version(file_hash),
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
    result["source_version"] = None
    result["repair_source_version"] = conversion_source_version(file_hash)
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
        except EmbeddingTokenizerUnavailableError:
            raise
        except Exception as exc:
            result.update(reason="tokenizer_unavailable", error=str(exc), ready=False)
            return result
        fingerprint = retrieval_chunking_fingerprint(canonical, embedding_model, config.fingerprint)
        result["source_version"] = retrieval_source_version(file_hash, canonical, embedding_model, fingerprint)
        result["repair_source_version"] = result["source_version"]
        manifest = await repo.get_manifest(file_hash, embedding_model, canonical.generation, fingerprint)
        result["manifest"] = manifest
        expected = list(getattr(manifest, "expected_source_ids", None) or []) if manifest else []
        manifest_rows = await repo.get_chunks(file_hash, embedding_model, manifest_id=manifest.manifest_id) if manifest else []
        manifest_ready = bool(
            manifest
            and manifest.status == "completed"
            and manifest.published_at is not None
            and manifest.superseded_at is None
            and manifest.is_current
            and manifest.generation == canonical.generation
            and manifest.extraction_fingerprint == canonical.extraction_fingerprint
            and manifest.chunking_fingerprint == fingerprint
            and manifest.source_version == result["source_version"]
            and manifest.expected_chunk_count == len(expected)
            and _manifest_rows_complete(manifest, manifest_rows, canonical, embedding_model)
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

    thread_ready = True
    thread_job = None
    thread_reason = None
    if thread_id and result.get("source_version"):
        from app.services.embedding_materialization_service import get_document_embedding_job
        thread_job = await get_document_embedding_job(
            file_hash=file_hash,
            thread_id=thread_id,
            embedding_model=embedding_model or "",
        )
        if thread_job is None:
            thread_ready = False
            thread_reason = "thread_version_missing"
        elif str(thread_job.source_version) != str(result["source_version"]):
            thread_ready = False
            thread_reason = "thread_version_stale"
        elif thread_job.status != "completed":
            thread_ready = False
            thread_reason = f"thread_job_{thread_job.status}"
    result["thread_ready"] = thread_ready
    result["thread_job"] = thread_job
    result["ready"] = bool(
        result["canonical_ready"]
        and result["reading_ready"]
        and result["manifest_ready"]
        and result["vectors_ready"]
        and thread_ready
    )
    result["reason"] = "ready" if result["ready"] else (thread_reason or "manifest_incomplete")
    return result


async def ensure_retrieval_projection(
    *,
    file_hash: str,
    embedding_model: str,
    file_name: str | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Return a complete chunk manifest, lazily repairing canonical data if needed."""
    from app.services.document_conversion_service import enqueue_pdf_conversion

    repo = get_canonical_document_repo()
    canonical = await repo.get(file_hash)
    store = get_content_store()
    key = pdf_content_key(file_hash)
    freshness = await evaluate_document_freshness(file_hash, embedding_model, require_manifest=True)
    data = None
    if not freshness.get("canonical_ready"):
        if canonical is not None and canonical.status == "failed":
            raise DocumentConversionFailedError(
                file_hash,
                dict(getattr(canonical, "failure_json", None) or {}).get("message")
                or dict(getattr(canonical, "failure_json", None) or {}).get("error"),
            )
        data = await store.read(key) if await store.exists(key) else None
        if data is None:
            raise FileNotFoundError(f"PDF content not found for {file_hash}")
        file = await get_file(file_hash)
        await enqueue_pdf_conversion(
            file_hash=file_hash,
            data=data,
            file_name=file_name or (file.file_name if file else f"{file_hash}.pdf"),
            force_rebuild=True,
            retry_failed=False,
        )
        raise DocumentConversionPendingError(file_hash)

    config, counter = resolve_embedding_tokenizer(embedding_model)
    existing_manifest = freshness.get("manifest")
    if (
        freshness.get("canonical_ready")
        and existing_manifest is not None
        and existing_manifest.is_current
        and existing_manifest.status == "completed"
        and existing_manifest.vector_status != "failed"
    ):
        existing_rows = await repo.get_chunks(file_hash, embedding_model, manifest_id=existing_manifest.manifest_id)
        if _manifest_rows_complete(existing_manifest, existing_rows, canonical, embedding_model):
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
        document_title=_document_title(canonical, file_name),
    )
    fingerprint = retrieval_chunking_fingerprint(canonical, embedding_model, config.fingerprint)
    source_version = retrieval_source_version(file_hash, canonical, embedding_model, fingerprint)
    manifest = await repo.get_manifest(file_hash, embedding_model, canonical.generation, fingerprint)
    if manifest is not None and manifest.is_current and manifest.status == "completed" and manifest.vector_status != "failed":
        existing_rows = await repo.get_chunks(file_hash, embedding_model, manifest_id=manifest.manifest_id)
        if _manifest_rows_complete(manifest, existing_rows, canonical, embedding_model):
            return manifest, [_chunk_from_row(row) for row in existing_rows], _metadata_for_canonical(canonical, embedding_model, config)
    if (
        manifest is None
        or manifest.status not in {"completed", "running"}
        or manifest.expected_chunk_ids != [str(item["chunk_id"]) for item in chunks]
        or not _manifest_rows_complete(
            manifest,
            await repo.get_chunks(file_hash, embedding_model, manifest_id=manifest.manifest_id),
            canonical,
            embedding_model,
        )
        or (manifest.status == "completed" and manifest.vector_status == "failed")
    ):
        manifest = await repo.replace_manifest(
            file_hash=file_hash,
            embedding_model=embedding_model,
            generation=canonical.generation,
            extraction_fingerprint=canonical.extraction_fingerprint,
            source_version=source_version,
            chunking_fingerprint=fingerprint,
            chunks=chunks,
        )
    metadata = _metadata_for_canonical(canonical, embedding_model, config)
    return manifest, chunks, metadata


async def evaluate_retrieval_readiness(
    file_hash: str,
    embedding_model: str,
    *,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate canonical, manifest, and vector readiness as one contract."""
    # Retrieval readiness is never repairable without the exact tokenizer. Do
    # this before inspecting stale canonical state so a missing configuration
    # cannot enqueue background work that is guaranteed to fail.
    resolve_embedding_tokenizer(embedding_model)
    freshness = await evaluate_document_freshness(
        file_hash,
        embedding_model,
        require_manifest=True,
        verify_vectors=True,
        thread_id=thread_id,
    )
    canonical = freshness.get("canonical")
    metadata = freshness.get("metadata") or {}
    manifest = freshness.get("manifest")
    return {
        **freshness,
        "ready": bool(freshness.get("ready")),
        "reason": freshness.get("reason"),
        "manifest": manifest,
        "metadata": metadata,
        "error": freshness.get("error"),
        "source_version": freshness.get("source_version"),
        "repair_source_version": freshness.get("source_version"),
        "thread_job": freshness.get("thread_job"),
    }


__all__ = [
    "ensure_retrieval_projection",
    "evaluate_document_freshness",
    "evaluate_retrieval_readiness",
    "DocumentConversionPendingError",
    "DocumentConversionFailedError",
    "conversion_source_version",
    "retrieval_source_version",
    "retrieval_chunking_fingerprint",
]
