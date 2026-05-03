"""
Thread Management Service - Handles thread-level document metadata operations.

This module contains business logic for:
- Repairing thread document metadata
"""

import hashlib
import logging
from typing import Any, Dict, List

from app.db import (
    ProcessStatus,
    get_scoped_indexing_status,
    get_thread_shape,
    remove_document_from_stats,
    upsert_document_in_stats,
)
from app.db.vector import get_vector_db

logger = logging.getLogger(__name__)


async def repair_thread_documents_meta(thread_id: str, embedding_model: str, files: List[Any]) -> None:
    """Rebuild thread_stats.documents_meta for already-indexed files attached to a thread."""
    vector_db = get_vector_db()
    attached_hashes = {f.file_hash for f in files}
    shape = await get_thread_shape(thread_id)
    for stale_hash in list(shape.get("documents", {}).keys()):
        if stale_hash not in attached_hashes:
            await remove_document_from_stats(thread_id, stale_hash)

    for file in files:
        file_status = await get_file_status(file.file_hash)
        scoped_status = get_scoped_indexing_status(
            file_status,
            embedding_model=embedding_model,
            thread_id=thread_id,
        )
        chunk_count = await vector_db.get_file_chunk_count(file.file_hash, embedding_model)
        is_ready = ProcessStatus.is_completed(scoped_status.get("status", ProcessStatus.UNKNOWN.value)) or chunk_count > 0
        if not is_ready:
            await remove_document_from_stats(thread_id, file.file_hash)
            continue

        total_chars = int(scoped_status.get("total_chars", 0) or 0)
        await upsert_document_in_stats(
            thread_id,
            file.file_hash,
            {
                "file_name": file.file_name,
                "source_type": file.source_type,
                "chunk_count": chunk_count,
                "total_chars": total_chars,
                "indexing_status": ProcessStatus.COMPLETED.value,
                "indexed_at": scoped_status.get("finished_at"),
            },
        )


# Import at the end to avoid circular dependencies
from app.db import get_file_status
