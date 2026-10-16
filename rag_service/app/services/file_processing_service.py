"""
File Processing Service - Handles background file parsing and indexing.

This module contains business logic for:
- Queueing files for background processing
- Parsing PDF files and extracting text with coordinates
- Indexing documents for semantic search
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks

from app.db import FileSourceType, OperationResultStatus, ProcessStatus

# SQLModel repositories for atomic transactions

# Database operations (SQLModel/PostgreSQL)
from app.db import (
    add_file_to_thread,
    add_file_to_project,
    create_or_get_file,
    get_file_parsed_sentences,
    update_file_parsed_sentences,
    get_file_status,
    update_indexing_status,
    update_parsing_status,
)
from app.rag.indexer import index_document_for_thread
from app.services.document_conversion_service import enqueue_pdf_conversion
from app.db.repositories.canonical_document_repo import get_canonical_document_repo
from app.services.content_store import get_content_store, pdf_content_key
from app.time_utils import iso_utc_z

logger = logging.getLogger(__name__)


async def _enqueue_pdf_conversion(file_hash: str, filename: str) -> None:
    store = get_content_store()
    key = pdf_content_key(file_hash)
    if not await store.exists(key):
        raise FileNotFoundError(f"PDF content not found for {file_hash}")
    data = await store.read(key)
    await enqueue_pdf_conversion(file_hash=file_hash, data=data, file_name=filename)


async def publish_reading_projection(file_hash: str, parsed_data: Dict[str, Any]) -> bool:
    """Publish canonical reading data to the File cache used by read endpoints."""
    repo = get_canonical_document_repo()
    canonical = await repo.get(file_hash)
    if canonical is None or canonical.status != "completed":
        return False
    sentences = parsed_data.get("sentences") if isinstance(parsed_data, dict) else None
    generation = parsed_data.get("generation") if isinstance(parsed_data, dict) else None
    fingerprint = parsed_data.get("extraction_fingerprint") if isinstance(parsed_data, dict) else None
    document_json = canonical.document_json if isinstance(canonical.document_json, dict) else {}
    if not isinstance(sentences, list) or not generation or not fingerprint:
        raise ValueError("reading projection must include sentences, generation, and extraction_fingerprint")
    if str(generation) != str(canonical.generation) or str(fingerprint) != str(canonical.extraction_fingerprint):
        raise ValueError("reading projection version does not match canonical document")
    payload = {
        "version": "2.0",
        "sentences": sentences,
        "generation": str(generation),
        "extraction_fingerprint": str(fingerprint),
    }
    current = await get_file_parsed_sentences(file_hash)
    if (
        isinstance(current, dict)
        and current.get("generation") == payload["generation"]
        and current.get("extraction_fingerprint") == payload["extraction_fingerprint"]
        and isinstance(current.get("sentences"), list)
    ):
        return True
    return await update_file_parsed_sentences(file_hash, json.dumps(payload))


def _default_file_status(file_hash: str) -> Dict[str, Any]:
    """Return the default status payload for an unknown file."""
    return {
        "file_hash": file_hash,
        "parsing": {"status": ProcessStatus.UNKNOWN.value},
        "indexing": {"status": ProcessStatus.UNKNOWN.value},
        "indexing_status": {
            "summary": {"status": ProcessStatus.UNKNOWN.value},
            "models": {},
        },
        "updated_at": None,
    }


def _scoped_status_payload(
    file_hash: str,
    status: Optional[Dict[str, Any]],
    embedding_model: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a backward-compatible file-status payload with a scoped top-level indexing section."""
    from app.db import get_scoped_indexing_status
    payload = dict(status or _default_file_status(file_hash))
    payload["file_hash"] = file_hash
    payload["indexing"] = get_scoped_indexing_status(
        payload,
        embedding_model=embedding_model,
        thread_id=thread_id,
    )
    return payload


async def queue_file_processing(
    background_tasks: BackgroundTasks,
    thread,
    file_hash: str,
    file_name: str,
    backend_url: str = "",  # No longer needed, files are read locally
    file_path: Optional[str] = None,
    source_type: str = FileSourceType.PDF.value,
    indexing_metadata: Optional[Dict[str, Any]] = None,
    markdown_content: Optional[str] = None,
) -> None:
    """Ensure a file is attached to a thread and background parse/index work is queued."""
    await create_or_get_file(
        file_hash=file_hash,
        file_name=file_name,
        file_path=file_path,
        source_type=source_type,
    )
    await add_file_to_thread(thread.id, file_hash)
    if FileSourceType.uses_pdf_conversion(source_type):
        await _enqueue_pdf_conversion(file_hash, file_name)

    file_status = await get_file_status(file_hash)
    parsing_status = (file_status or {}).get("parsing", {"status": ProcessStatus.UNKNOWN.value})

    from app.db import get_scoped_indexing_status
    scoped_indexing = get_scoped_indexing_status(
        file_status,
        embedding_model=thread.embedding_model,
        thread_id=thread.id,
    )
    if not ProcessStatus.is_completed(scoped_indexing.get("status", ProcessStatus.UNKNOWN.value)) and not ProcessStatus.is_running(scoped_indexing.get("status", ProcessStatus.UNKNOWN.value)):
        await update_indexing_status(
            file_hash=file_hash,
            status=ProcessStatus.PENDING.value,
            embedding_model=thread.embedding_model,
            thread_id=thread.id,
        )
        background_tasks.add_task(
            _background_index,
            file_hash,
            thread.id,
            thread.embedding_model,
            file_name,
            backend_url,
            indexing_metadata or {},
            markdown_content,
        )

    parsed_data = await get_file_parsed_sentences(file_hash)
    canonical = await get_canonical_document_repo().get(file_hash)
    from app.services.document_projection_service import evaluate_document_freshness
    freshness = await evaluate_document_freshness(file_hash, require_reading=True)
    if canonical and canonical.status == "completed" and freshness.get("canonical_ready") and freshness.get("reading_ready"):
        if not ProcessStatus.is_completed(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
            await update_parsing_status(file_hash, ProcessStatus.COMPLETED.value)
    elif not FileSourceType.uses_pdf_conversion(source_type) and parsed_data and isinstance(parsed_data.get("sentences"), list):
        if not ProcessStatus.is_completed(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
            await update_parsing_status(file_hash, ProcessStatus.COMPLETED.value)
    elif not ProcessStatus.is_running(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
        await update_parsing_status(file_hash, ProcessStatus.PENDING.value)
        background_tasks.add_task(_background_parse, file_hash, file_name, backend_url, indexing_metadata or {})


async def queue_project_file_processing(
    background_tasks: BackgroundTasks,
    project,
    file_hash: str,
    file_name: str,
    file_path: Optional[str] = None,
    source_type: str = FileSourceType.PDF.value,
    indexing_metadata: Optional[Dict[str, Any]] = None,
    markdown_content: Optional[str] = None,
) -> None:
    """Attach a canonical file to project knowledge and queue shared model indexing."""
    await create_or_get_file(
        file_hash=file_hash,
        file_name=file_name,
        file_path=file_path,
        source_type=source_type,
    )
    await add_file_to_project(project.id, file_hash)
    if FileSourceType.uses_pdf_conversion(source_type):
        await _enqueue_pdf_conversion(file_hash, file_name)
    file_status = await get_file_status(file_hash)
    parsing_status = (file_status or {}).get("parsing", {"status": ProcessStatus.UNKNOWN.value})
    from app.db import get_scoped_indexing_status
    scoped_indexing = get_scoped_indexing_status(file_status, embedding_model=project.embedding_model)
    if (
        not ProcessStatus.is_completed(scoped_indexing.get("status", ProcessStatus.UNKNOWN.value))
        and not ProcessStatus.is_running(scoped_indexing.get("status", ProcessStatus.UNKNOWN.value))
    ):
        await update_indexing_status(
            file_hash=file_hash,
            status=ProcessStatus.PENDING.value,
            embedding_model=project.embedding_model,
        )
        background_tasks.add_task(
            _background_index,
            file_hash,
            f"project:{project.id}",
            project.embedding_model,
            file_name,
            "",
            indexing_metadata or {},
            markdown_content,
            False,
        )
    parsed_data = await get_file_parsed_sentences(file_hash)
    canonical = await get_canonical_document_repo().get(file_hash)
    from app.services.document_projection_service import evaluate_document_freshness
    freshness = await evaluate_document_freshness(file_hash, require_reading=True)
    if canonical and canonical.status == "completed" and freshness.get("canonical_ready") and freshness.get("reading_ready"):
        if not ProcessStatus.is_completed(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
            await update_parsing_status(file_hash, ProcessStatus.COMPLETED.value)
    elif not FileSourceType.uses_pdf_conversion(source_type) and parsed_data and isinstance(parsed_data.get("sentences"), list):
        if not ProcessStatus.is_completed(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
            await update_parsing_status(file_hash, ProcessStatus.COMPLETED.value)
    elif not ProcessStatus.is_running(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
        await update_parsing_status(file_hash, ProcessStatus.PENDING.value)
        background_tasks.add_task(_background_parse, file_hash, file_name, "", indexing_metadata or {})


async def _background_parse(file_hash: str, filename: str, backend_url: str = "", source_metadata: Optional[Dict[str, Any]] = None):
    """Queue durable conversion work without treating it as completed reading."""
    await _enqueue_pdf_conversion(file_hash, filename)
    await update_parsing_status(file_hash, ProcessStatus.PENDING.value)
    return


async def _enqueue_document_embedding_job(
    *,
    file_hash: str,
    scope_id: str,
    embedding_model: str,
    file_name: str,
    persist_thread_state: bool,
) -> None:
    from app.services.document_projection_service import DocumentConversionFailedError
    from app.services.embedding_materialization_service import enqueue_document_embedding_if_needed
    from app.services.embedding_tokenizer import EmbeddingTokenizerUnavailableError

    await update_indexing_status(
        file_hash=file_hash,
        status=ProcessStatus.PENDING.value,
        embedding_model=embedding_model,
        thread_id=scope_id if persist_thread_state else None,
    )
    try:
        await enqueue_document_embedding_if_needed(
            file_hash=file_hash,
            thread_id=scope_id,
            embedding_model=embedding_model,
            file_name=file_name,
        )
    except EmbeddingTokenizerUnavailableError:
        raise
    except DocumentConversionFailedError as exc:
        await update_indexing_status(
            file_hash=file_hash,
            status=ProcessStatus.FAILED.value,
            embedding_model=embedding_model,
            thread_id=scope_id if persist_thread_state else None,
            finished_at=iso_utc_z(),
            error=str(exc),
        )
        raise


async def _background_index(
    file_hash: str,
    thread_id: str,
    embedding_model: str,
    file_name: str,
    backend_url: str,
    metadata: Optional[Dict[str, Any]] = None,
    markdown_content: Optional[str] = None,
    persist_thread_state: bool = True,
):
    """Queue durable conversion and embedding work; do not index PDFs inline."""
    if markdown_content is None:
        try:
            await _enqueue_pdf_conversion(file_hash, file_name)
            await _enqueue_document_embedding_job(
                file_hash=file_hash,
                scope_id=thread_id,
                embedding_model=embedding_model,
                file_name=file_name,
                persist_thread_state=persist_thread_state,
            )
        except Exception as exc:
            logger.exception("Background document indexing enqueue failed for %s", file_hash)
            try:
                await update_indexing_status(
                    file_hash=file_hash,
                    status=ProcessStatus.FAILED.value,
                    embedding_model=embedding_model,
                    thread_id=thread_id if persist_thread_state else None,
                    finished_at=iso_utc_z(),
                    error=str(exc),
                )
            except Exception as update_error:
                logger.error("Failed to update indexing status to failed for %s: %s", file_hash, update_error)
        return

    started_at = iso_utc_z()
    try:
        claimed = await update_indexing_status(
            file_hash=file_hash,
            status=ProcessStatus.RUNNING.value,
            embedding_model=embedding_model,
            thread_id=thread_id if persist_thread_state else None,
            started_at=started_at,
            claim=True,
        )
        if not claimed:
            return
        result = await index_document_for_thread(
            thread_id=thread_id,
            file_hash=file_hash,
            embedding_model=embedding_model,
            metadata=metadata,
            markdown_content=markdown_content,
            persist_thread_state=persist_thread_state,
        )
        if result.get("status") != OperationResultStatus.SUCCESS.value:
            raise Exception(result.get("message", "Indexing failed"))
        logger.info("Background markdown indexing completed for %s in thread %s", file_hash, thread_id)
    except Exception as e:
        logger.exception("Background indexing failed for %s", file_hash)
        try:
            await update_indexing_status(
                file_hash=file_hash,
                status=ProcessStatus.FAILED.value,
                embedding_model=embedding_model,
                thread_id=thread_id if persist_thread_state else None,
                started_at=started_at,
                finished_at=iso_utc_z(),
                error=str(e),
            )
        except Exception as update_error:
            logger.error("Failed to update indexing status to failed for %s: %s", file_hash, update_error)
