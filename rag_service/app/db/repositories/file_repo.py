"""
file_repo.py - File CRUD operations.

This module provides repository methods for managing file entities in the database,
including file status management for parsing and indexing operations.
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any

import aiosqlite

from app.db.config import DB_PATH
from app.db.models import File, ProcessStatus
from app.db.repositories.base import BaseRepository

from app.db.status import (
    _parse_json_list,
    _normalize_file_status,
    _copy_process_section,
    _collapse_process_sections,
    get_scoped_indexing_status,
)

# Default value for parsed_sentences_json column - used at row creation and API fallback
DEFAULT_SENTENCES_JSON = {"version": "1.0", "sentences": None}

# Default value for file_status column - used at row creation
DEFAULT_FILE_STATUS = {
    "parsing": {"status": "unknown"},
    "parsing_status": {"status": "unknown"},
    "indexing": {"status": "unknown"},
    "indexing_status": {"summary": {"status": "unknown"}, "models": {}},
}


class FileRepository(BaseRepository):
    """Repository for file database operations."""

    async def create_or_get(
        self,
        file_hash: str,
        file_name: str,
        file_path: Optional[str] = None,
        source_type: str = "pdf",
    ) -> File:
        """Create a new file record or return existing one."""
        async def _create_or_get_in_transaction(db):
            # Try to insert with initialized columns, ignore if exists
            await db.execute(
                "INSERT OR IGNORE INTO files (file_hash, file_name, file_path, source_type, parsed_sentences_json, file_status) VALUES (?, ?, ?, ?, ?, ?)",
                (file_hash, file_name, file_path, source_type, json.dumps(DEFAULT_SENTENCES_JSON), json.dumps(DEFAULT_FILE_STATUS))
            )
            if file_name or file_path or source_type:
                await db.execute(
                    """
                    UPDATE files
                    SET
                        file_name = COALESCE(NULLIF(?, ''), file_name),
                        file_path = COALESCE(file_path, ?),
                        source_type = COALESCE(NULLIF(?, ''), source_type)
                    WHERE file_hash = ?
                    """,
                    (file_name, file_path, source_type, file_hash),
                )
            await db.commit()

            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT file_hash, file_name, file_path, source_type FROM files WHERE file_hash = ?",
                (file_hash,)
            )
            row = await cursor.fetchone()
            return File(
                file_hash=row["file_hash"],
                file_name=row["file_name"],
                file_path=row["file_path"],
                source_type=row["source_type"] or "pdf",
            )

        try:
            return await self._transaction(_create_or_get_in_transaction)
        except Exception:
            raise

    async def get(self, file_hash: str) -> Optional[File]:
        """Get a file by hash."""
        row = await self._fetch_one(
            "SELECT file_hash, file_name, file_path, source_type FROM files WHERE file_hash = ?",
            (file_hash,)
        )
        if row:
            return File(
                file_hash=row["file_hash"],
                file_name=row["file_name"],
                file_path=row["file_path"],
                source_type=row["source_type"] or "pdf",
            )
        return None

    async def update_parsed_sentences(self, file_hash: str, parsed_data_json: str) -> bool:
        """Store parsed sentences JSON in the files table."""
        cursor = await self._execute(
            "UPDATE files SET parsed_sentences_json = ? WHERE file_hash = ?",
            (parsed_data_json, file_hash)
        )
        return cursor.rowcount > 0

    async def get_parsed_sentences(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve parsed sentences JSON from the files table."""
        row = await self._fetch_one(
            "SELECT parsed_sentences_json FROM files WHERE file_hash = ?",
            (file_hash,)
        )
        if row and row["parsed_sentences_json"]:
            try:
                return json.loads(row["parsed_sentences_json"])
            except Exception:
                return None
        return None

    async def get_status(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve file_status JSON from the files table."""
        row = await self._fetch_one(
            "SELECT file_status FROM files WHERE file_hash = ?",
            (file_hash,)
        )
        if row and row["file_status"]:
            try:
                return _normalize_file_status(json.loads(row["file_status"]))
            except Exception:
                return _normalize_file_status({})
        return None

    async def update_status(self, file_hash: str, status_data: Dict[str, Any]) -> bool:
        """Update file_status JSON for a file, merging with existing status."""
        async def _update_status_in_transaction(db):
            # Get existing status
            cursor = await db.execute(
                "SELECT file_status FROM files WHERE file_hash = ?",
                (file_hash,)
            )
            row = await cursor.fetchone()
            existing: Dict[str, Any] = {}
            if row and row["file_status"]:
                try:
                    existing = _normalize_file_status(json.loads(row["file_status"]))
                except Exception:
                    existing = _normalize_file_status({})

            # Merge status data
            merged = {**existing, **status_data}
            merged["updated_at"] = datetime.utcnow().isoformat()
            normalized = _normalize_file_status(merged)

            # Update
            cursor = await db.execute(
                "UPDATE files SET file_status = ? WHERE file_hash = ?",
                (json.dumps(normalized), file_hash)
            )
            await db.commit()
            return cursor.rowcount > 0

        try:
            return await self._transaction(_update_status_in_transaction)
        except Exception:
            raise

    async def update_parsing_status(
        self,
        file_hash: str,
        status: str,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        error: Optional[str] = None,
        claim: bool = False,
    ) -> bool:
        """Update parsing section of file_status."""
        if claim:
            return await self._claim_file_status(file_hash, "parsing", status, started_at=started_at)

        current_status = await self.get_status(file_hash) or {}
        parsing = _copy_process_section(current_status.get("parsing_status") or current_status.get("parsing"))
        parsing["status"] = status
        if started_at:
            parsing["started_at"] = started_at
        if finished_at:
            parsing["finished_at"] = finished_at
        if error:
            parsing["error"] = error
        else:
            parsing.pop("error", None)
        return await self.update_status(
            file_hash,
            {
                "parsing": parsing,
                "parsing_status": parsing,
            },
        )

    async def update_indexing_status(
        self,
        file_hash: str,
        status: str,
        embedding_model: Optional[str] = None,
        thread_id: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        error: Optional[str] = None,
        chunk_count: Optional[int] = None,
        total_chars: Optional[int] = None,
        reused_existing_embeddings: Optional[bool] = None,
        claim: bool = False,
    ) -> bool:
        """Update indexing section of file_status."""
        if claim:
            return await self._claim_indexing_status(
                file_hash, status, embedding_model, thread_id, started_at
            )

        current_status = _normalize_file_status(await self.get_status(file_hash) or {})
        indexing_status = current_status.get("indexing_status", {})
        models = dict(indexing_status.get("models", {}))

        if embedding_model:
            model_status = _copy_process_section(models.get(embedding_model))
        else:
            model_status = _copy_process_section(indexing_status.get("summary"))

        model_status["status"] = status
        if started_at:
            model_status["started_at"] = started_at
        if finished_at:
            model_status["finished_at"] = finished_at
        if error:
            model_status["error"] = error
        else:
            model_status.pop("error", None)
        if chunk_count is not None:
            model_status["chunk_count"] = chunk_count
        if total_chars is not None:
            model_status["total_chars"] = total_chars
        if reused_existing_embeddings is not None:
            model_status["reused_existing_embeddings"] = reused_existing_embeddings

        if embedding_model:
            threads = dict(model_status.get("threads", {}))
            if thread_id:
                thread_status = _copy_process_section(threads.get(thread_id))
                thread_status["status"] = status
                if started_at:
                    thread_status["started_at"] = started_at
                if finished_at:
                    thread_status["finished_at"] = finished_at
                if error:
                    thread_status["error"] = error
                else:
                    thread_status.pop("error", None)
                if chunk_count is not None:
                    thread_status["chunk_count"] = chunk_count
                if total_chars is not None:
                    thread_status["total_chars"] = total_chars
                if reused_existing_embeddings is not None:
                    thread_status["reused_existing_embeddings"] = reused_existing_embeddings
                threads[thread_id] = thread_status
            model_status["threads"] = threads
            models[embedding_model] = model_status
            summary = _collapse_process_sections(list(models.values()))
        else:
            summary = model_status

        return await self.update_status(
            file_hash,
            {
                "indexing": summary,
                "indexing_status": {
                    "summary": summary,
                    "models": models,
                },
            },
        )

    async def _claim_file_status(
        self,
        file_hash: str,
        section: str,
        claim_status: str,
        started_at: Optional[str] = None,
    ) -> bool:
        """Atomically claim a process section if it is not already running or completed."""
        async def _claim_in_transaction(db):
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT file_status FROM files WHERE file_hash = ?",
                (file_hash,),
            )
            row = await cursor.fetchone()
            if not row:
                return False

            try:
                current = _normalize_file_status(json.loads(row["file_status"])) if row["file_status"] else _normalize_file_status({})
            except Exception:
                current = _normalize_file_status({})

            section_payload = _copy_process_section(current.get(section))
            current_status = section_payload.get("status", ProcessStatus.UNKNOWN.value)
            if ProcessStatus.is_running(current_status) or ProcessStatus.is_completed(current_status):
                return False

            section_payload["status"] = claim_status
            if started_at:
                section_payload["started_at"] = started_at
            section_payload.pop("error", None)

            merged = {**current, section: section_payload, f"{section}_status": section_payload}
            merged["updated_at"] = datetime.utcnow().isoformat()
            normalized = _normalize_file_status(merged)

            cursor = await db.execute(
                "UPDATE files SET file_status = ? WHERE file_hash = ?",
                (json.dumps(normalized), file_hash),
            )
            if cursor.rowcount <= 0:
                return False
            await db.commit()
            return True

        try:
            return await self._transaction(_claim_in_transaction)
        except Exception:
            raise

    async def _claim_indexing_status(
        self,
        file_hash: str,
        status: str,
        embedding_model: Optional[str],
        thread_id: Optional[str],
        started_at: Optional[str],
    ) -> bool:
        """Atomically claim indexing status for a model/thread."""
        async def _claim_in_transaction(db):
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT file_status FROM files WHERE file_hash = ?",
                (file_hash,),
            )
            row = await cursor.fetchone()
            if not row:
                return False

            try:
                current = _normalize_file_status(json.loads(row["file_status"])) if row["file_status"] else _normalize_file_status({})
            except Exception:
                current = _normalize_file_status({})

            indexing_status = current.get("indexing_status", {})
            models = dict(indexing_status.get("models", {}))
            if embedding_model:
                model_status = _copy_process_section(models.get(embedding_model))
                threads = dict(model_status.get("threads", {}))
                thread_status = _copy_process_section(threads.get(thread_id))
                current_status = thread_status.get("status", ProcessStatus.UNKNOWN.value)
                if ProcessStatus.is_running(current_status) or ProcessStatus.is_completed(current_status):
                    return False
                thread_status["status"] = status
                if started_at:
                    thread_status["started_at"] = started_at
                thread_status.pop("error", None)
                threads[thread_id] = thread_status
                model_status["threads"] = threads
                models[embedding_model] = model_status
                summary = _collapse_process_sections(list(models.values()))
            else:
                summary = _copy_process_section(indexing_status.get("summary"))
                current_status = summary.get("status", ProcessStatus.UNKNOWN.value)
                if ProcessStatus.is_running(current_status) or ProcessStatus.is_completed(current_status):
                    return False
                summary["status"] = status
                if started_at:
                    summary["started_at"] = started_at
                summary.pop("error", None)

            merged = {
                **current,
                "indexing": summary,
                "indexing_status": {
                    "summary": summary,
                    "models": models,
                },
            }
            merged["updated_at"] = datetime.utcnow().isoformat()
            normalized = _normalize_file_status(merged)

            cursor = await db.execute(
                "UPDATE files SET file_status = ? WHERE file_hash = ?",
                (json.dumps(normalized), file_hash),
            )
            if cursor.rowcount <= 0:
                return False
            await db.commit()
            return True

        try:
            return await self._transaction(_claim_in_transaction)
        except Exception:
            raise

    async def remove_thread_indexing_status(self, file_hash: str, embedding_model: str, thread_id: str) -> bool:
        """Remove a thread-scoped indexing entry and recompute the remaining summaries."""
        current_status = _normalize_file_status(await self.get_status(file_hash) or {})
        indexing_status = current_status.get("indexing_status", {})
        models = dict(indexing_status.get("models", {}))
        model_status = dict(models.get(embedding_model, {}))
        threads = dict(model_status.get("threads", {}))

        if thread_id not in threads and embedding_model not in models:
            return True

        threads.pop(thread_id, None)
        if threads:
            recomputed = _collapse_process_sections(list(threads.values()))
            for key in ("chunk_count", "total_chars"):
                existing = model_status.get(key)
                if existing is not None:
                    recomputed[key] = existing
            model_status = {**model_status, **recomputed, "threads": threads}
            models[embedding_model] = model_status
        else:
            models.pop(embedding_model, None)

        summary = _collapse_process_sections(list(models.values()))
        return await self.update_status(
            file_hash,
            {
                "indexing": summary,
                "indexing_status": {
                    "summary": summary,
                    "models": models,
                },
            },
        )

    async def delete(self, file_hash: str) -> bool:
        """Delete a file row once all thread associations have been removed."""
        cursor = await self._execute(
            "DELETE FROM files WHERE file_hash = ?",
            (file_hash,),
        )
        return cursor.rowcount > 0
