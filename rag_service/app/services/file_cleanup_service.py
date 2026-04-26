"""
File Cleanup Service - Handles cleanup of file artifacts and detached files.

This module contains business logic for:
- Deleting file artifacts (PDFs, webpage mappings)
- Cleaning up detached files from threads
- Removing orphaned vector data
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from app.db import (
    count_threads_with_file,
    count_threads_with_file_for_model,
    delete_file_record,
    get_file_status,
    remove_document_from_stats,
    remove_thread_indexing_status,
)
from app.db.vector import get_vector_db

logger = logging.getLogger(__name__)
WEBPAGES_DIR = "/static/webpages"


async def delete_file_artifacts(file_hash: str) -> None:
    """Delete the stored PDF and any webpage mapping that points to it."""
    pdf_path = f"/static/{file_hash}.pdf"
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except Exception as exc:
            logger.warning("Failed to delete PDF artifact %s: %s", pdf_path, exc)

    if not os.path.isdir(WEBPAGES_DIR):
        return

    for filename in os.listdir(WEBPAGES_DIR):
        if not filename.endswith(".mapping.json"):
            continue
        mapping_path = os.path.join(WEBPAGES_DIR, filename)
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if mapping.get("pdf_hash") != file_hash:
                continue
            os.remove(mapping_path)
        except Exception as exc:
            logger.warning("Failed to delete webpage mapping %s: %s", mapping_path, exc)


async def cleanup_detached_file(file_hash: str, thread_id: str, embed_model: str) -> None:
    """Apply post-detach cleanup for status, vector data, and orphaned file artifacts."""
    await remove_document_from_stats(thread_id, file_hash)
    await remove_thread_indexing_status(file_hash, embed_model, thread_id)

    vector_db = get_vector_db()
    remaining_model_refs = await count_threads_with_file_for_model(file_hash, embed_model)
    if remaining_model_refs == 0:
        await vector_db.delete_document_vectors_by_file_hash_and_model(
            file_hash=file_hash,
            embedding_model_name=embed_model,
        )

    remaining_refs = await count_threads_with_file(file_hash)
    if remaining_refs == 0:
        file_status = await get_file_status(file_hash) or {}
        indexing_status = file_status.get("indexing_status", {})
        models = indexing_status.get("models", {}) if isinstance(indexing_status, dict) else {}
        model_names = [name for name in models.keys() if isinstance(name, str) and name]
        if not model_names:
            model_names = [embed_model]
        for model_name in model_names:
            await vector_db.delete_document_vectors_by_file_hash_and_model(
                file_hash=file_hash,
                embedding_model_name=model_name,
            )
        await delete_file_record(file_hash)
        await delete_file_artifacts(file_hash)
