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
import os
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks

from app.db import ProcessStatus

# SQLModel repositories for atomic transactions
from app.db.repositories.file_repo_sqlmodel import FileRepository
from app.db.repositories.thread_file_repo_sqlmodel import ThreadFileRepository

# Database operations (SQLModel/PostgreSQL)
from app.db import (
    add_file_to_thread,
    create_or_get_file,
    get_file_parsed_sentences,
    get_file_status,
    update_indexing_status,
    update_parsing_status,
)
from app.db.vector import get_vector_db
from app.rag.indexer import index_document_for_thread
from app.services.nlp_service import split_into_sentences
from app.services.parsing_service import extract_text_with_coordinates

logger = logging.getLogger(__name__)


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
    source_type: str = "pdf",
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

    file_status = await get_file_status(file_hash)
    parsing_status = (file_status or {}).get("parsing", {"status": ProcessStatus.UNKNOWN.value})

    from app.db import get_scoped_indexing_status
    scoped_indexing = get_scoped_indexing_status(
        file_status,
        embedding_model=thread.embed_model,
        thread_id=thread.id,
    )
    if not ProcessStatus.is_completed(scoped_indexing.get("status", ProcessStatus.UNKNOWN.value)) and not ProcessStatus.is_running(scoped_indexing.get("status", ProcessStatus.UNKNOWN.value)):
        await update_indexing_status(
            file_hash=file_hash,
            status=ProcessStatus.PENDING.value,
            embedding_model=thread.embed_model,
            thread_id=thread.id,
        )
        background_tasks.add_task(
            _background_index,
            file_hash,
            thread.id,
            thread.embed_model,
            file_name,
            backend_url,
            indexing_metadata or {},
            markdown_content,
        )

    parsed_data = await get_file_parsed_sentences(file_hash)
    if parsed_data and parsed_data.get("sentences"):
        if not ProcessStatus.is_completed(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
            await update_parsing_status(file_hash, ProcessStatus.COMPLETED.value)
    elif not ProcessStatus.is_running(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
        await update_parsing_status(file_hash, ProcessStatus.PENDING.value)
        background_tasks.add_task(_background_parse, file_hash, file_name, backend_url)


async def _background_parse(file_hash: str, filename: str, backend_url: str = ""):
    """
    Background task to parse PDF and update status with atomic transactions.
    Uses SQLModel repository for transaction safety - sentences and status updated together.
    Reads PDF from local disk at /static/{file_hash}.pdf
    """
    file_repo = FileRepository()

    current_status = await get_file_status(file_hash)
    parsing_status = (current_status or {}).get("parsing", {"status": ProcessStatus.UNKNOWN.value})

    if ProcessStatus.is_completed(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
        parsed = await get_file_parsed_sentences(file_hash)
        if parsed and parsed.get("sentences"):
            return

    started_at = datetime.utcnow().isoformat()
    try:
        # Claim the parsing job using legacy function (this is already atomic via claim mechanism)
        claimed = await update_parsing_status(
            file_hash,
            ProcessStatus.RUNNING.value,
            started_at=started_at,
            claim=True,
        )
        if not claimed:
            return

        # Read PDF from local disk
        pdf_path = f"/static/{file_hash}.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        sentences = extract_text_with_coordinates(pdf_data, filename=filename)
        parsed_data = {
            "version": "1.0",
            "sentences": sentences
        }

        # ATOMIC: Store sentences AND update status to completed in ONE transaction
        finished_at = datetime.utcnow().isoformat()
        success = await file_repo.complete_parsing_atomically(
            file_hash=file_hash,
            parsed_data_json=json.dumps(parsed_data),
            finished_at=finished_at
        )

        if success:
            logger.info(f"Background parsing completed for {file_hash} - {len(sentences)} sentences stored atomically")
        else:
            logger.error(f"Failed to atomically complete parsing for {file_hash}")

    except Exception as e:
        traceback.print_exc()
        finished_at = datetime.utcnow().isoformat()
        try:
            # ATOMIC: Update status to failed with error message
            await file_repo.fail_parsing_atomically(
                file_hash=file_hash,
                error=str(e),
                finished_at=finished_at
            )
        except Exception as update_error:
            logger.error(f"Failed to update parsing status to failed for {file_hash}: {update_error}")
        logger.error(f"Background parsing failed for {file_hash}: {e}")


async def _background_index(
    file_hash: str,
    thread_id: str,
    embedding_model: str,
    file_name: str,
    backend_url: str,
    metadata: Optional[Dict[str, Any]] = None,
    markdown_content: Optional[str] = None,
):
    """
    Background task to index a document for a thread after parsing completes.
    """
    started_at = datetime.utcnow().isoformat()
    try:
        claimed = await update_indexing_status(
            file_hash=file_hash,
            status=ProcessStatus.RUNNING.value,
            embedding_model=embedding_model,
            thread_id=thread_id,
            started_at=started_at,
            claim=True,
        )
        if not claimed:
            return

        result = await index_document_for_thread(
            thread_id=thread_id,
            file_hash=file_hash,
            embedding_model_name=embedding_model,
            metadata=metadata,
            markdown_content=markdown_content,
        )
        if result.get("status") != "success":
            raise Exception(result.get("message", "Indexing failed"))
        logger.info(f"Background indexing completed for %s in thread %s", file_hash, thread_id)

        # Trigger PDF parsing for sentence extraction (needed for both PDFs and web sources)
        await _background_parse(file_hash, file_name, backend_url)
    except Exception as e:
        traceback.print_exc()
        finished_at = datetime.utcnow().isoformat()
        try:
            await update_indexing_status(
                file_hash=file_hash,
                status=ProcessStatus.FAILED.value,
                embedding_model=embedding_model,
                thread_id=thread_id,
                started_at=started_at,
                finished_at=finished_at,
                error=str(e),
            )
        except Exception as update_error:
            logger.error(f"Failed to update indexing status to failed for {file_hash}: {update_error}")
        logger.error(f"Background indexing failed for {file_hash}: {e}")
