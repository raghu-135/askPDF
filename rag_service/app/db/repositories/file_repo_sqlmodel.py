"""
file_repo_sqlmodel.py - File CRUD operations with SQLModel.

This module provides repository methods for managing file entities
using SQLModel with PostgreSQL, including JSONB status handling and
parsed sentences management.
"""

import json
from typing import Optional, Dict, Any, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.models_sqlmodel import File, ProcessStatus
from app.db.jsonb_utils import set_jsonb_field, merge_jsonb_field, replace_jsonb_field
from app.db.connection_sqlmodel import async_session_maker
from app.db.status import (
    _normalize_file_status,
    _copy_process_section,
    _collapse_process_sections,
    get_scoped_indexing_status,
)
from app.time_utils import iso_utc_z

# Default values for file columns
DEFAULT_SENTENCES_JSON = {"version": "1.0", "sentences": None}
DEFAULT_FILE_STATUS = {
    "parsing": {"status": "unknown"},
    "parsing_status": {"status": "unknown"},
    "indexing": {"status": "unknown"},
    "indexing_status": {"summary": {"status": "unknown"}, "models": {}},
}


class FileRepository:
    """Repository for file database operations using SQLModel."""

    def __init__(self, session: Optional[AsyncSession] = None):
        """Initialize with optional session for dependency injection."""
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get a database session - injected for tests, default for production."""
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def create_or_get(
        self,
        file_hash: str,
        file_name: str,
        file_path: Optional[str] = None,
        source_type: str = "pdf",
    ) -> File:
        """Create a new file record or return existing one."""
        session = await self._get_session()
        async with session.begin():
            # Try to get existing file
            result = await session.execute(
                select(File).where(File.file_hash == file_hash)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update fields if provided
                if file_name:
                    existing.file_name = file_name
                if file_path:
                    existing.file_path = file_path
                if source_type:
                    existing.source_type = source_type
                await session.flush()
                await session.refresh(existing)
                return existing

            # Create new file
            file = File(
                file_hash=file_hash,
                file_name=file_name,
                file_path=file_path,
                source_type=source_type,
                parsed_sentences_json=json.dumps(DEFAULT_SENTENCES_JSON),
                file_status=dict(DEFAULT_FILE_STATUS),
            )
            session.add(file)
            await session.flush()
            await session.refresh(file)
        return file

    async def get(self, file_hash: str) -> Optional[File]:
        """Get a file by hash."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(File).where(File.file_hash == file_hash)
            )
            return result.scalar_one_or_none()

    async def update_parsed_sentences(
        self,
        file_hash: str,
        parsed_data_json: str
    ) -> bool:
        """Store parsed sentences JSON in the files table."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(File).where(File.file_hash == file_hash)
            )
            file = result.scalar_one_or_none()
            if not file:
                return False

            file.parsed_sentences_json = parsed_data_json
            await session.flush()
        return True

    async def get_parsed_sentences(
        self,
        file_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve parsed sentences JSON from the files table."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(File.parsed_sentences_json).where(File.file_hash == file_hash)
            )
            data = result.scalar_one_or_none()
            if data:
                try:
                    return json.loads(data)
                except Exception:
                    return None
            return None

    async def get_status(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve file_status JSON from the files table."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(File.file_status).where(File.file_hash == file_hash)
            )
            status = result.scalar_one_or_none()
            if status:
                return _normalize_file_status(status)
            return _normalize_file_status({})

    async def update_status(
        self,
        file_hash: str,
        status_data: Dict[str, Any]
    ) -> bool:
        """Update file_status JSON for a file, merging with existing status."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(File).where(File.file_hash == file_hash)
            )
            file = result.scalar_one_or_none()
            if not file:
                return False

            # Get existing status and merge
            existing = _normalize_file_status(file.file_status or {})
            merged = {**existing, **status_data}
            merged["updated_at"] = iso_utc_z()
            normalized = _normalize_file_status(merged)

            # Use replace_jsonb_field for complete replacement
            replace_jsonb_field(file, "file_status", normalized)
            await session.flush()
        return True

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
            return await self._claim_file_status(
                file_hash, "parsing", status, started_at=started_at
            )

        current_status = await self.get_status(file_hash) or {}
        parsing = _copy_process_section(
            current_status.get("parsing_status") or current_status.get("parsing")
        )
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
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(File).where(File.file_hash == file_hash)
            )
            file = result.scalar_one_or_none()
            if not file:
                return False

            current = _normalize_file_status(file.file_status or {})
            section_payload = _copy_process_section(current.get(section))
            current_status = section_payload.get("status", ProcessStatus.UNKNOWN.value)

            if ProcessStatus.is_running(current_status) or ProcessStatus.is_completed(current_status):
                return False

            section_payload["status"] = claim_status
            if started_at:
                section_payload["started_at"] = started_at
            section_payload.pop("error", None)

            merged = {**current, section: section_payload, f"{section}_status": section_payload}
            merged["updated_at"] = iso_utc_z()
            normalized = _normalize_file_status(merged)

            replace_jsonb_field(file, "file_status", normalized)
            await session.flush()
        return True

    async def _claim_indexing_status(
        self,
        file_hash: str,
        status: str,
        embedding_model: Optional[str],
        thread_id: Optional[str],
        started_at: Optional[str],
    ) -> bool:
        """Atomically claim indexing status for a model/thread."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(File).where(File.file_hash == file_hash)
            )
            file = result.scalar_one_or_none()
            if not file:
                return False

            current = _normalize_file_status(file.file_status or {})
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
            merged["updated_at"] = iso_utc_z()
            normalized = _normalize_file_status(merged)

            replace_jsonb_field(file, "file_status", normalized)
            await session.flush()
        return True

    async def remove_thread_indexing_status(
        self,
        file_hash: str,
        embedding_model: str,
        thread_id: str
    ) -> bool:
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
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(File).where(File.file_hash == file_hash)
            )
            file = result.scalar_one_or_none()
            if not file:
                return False

            await session.delete(file)
        return True

    async def complete_parsing_atomically(
        self,
        file_hash: str,
        parsed_data_json: str,
        finished_at: str
    ) -> bool:
        """
        Atomically store parsed sentences AND update parsing status to completed.
        This ensures both operations happen in a single transaction - no race conditions.
        """
        session = await self._get_session()
        async with session.begin():
            # Fetch file with row-level lock to prevent race conditions
            result = await session.execute(
                select(File).where(File.file_hash == file_hash).with_for_update()
            )
            file = result.scalar_one_or_none()
            if not file:
                return False

            # Update parsed sentences
            file.parsed_sentences_json = parsed_data_json

            # Get current status and update parsing section
            current_status = _normalize_file_status(file.file_status or {})
            parsing = _copy_process_section(
                current_status.get("parsing_status") or current_status.get("parsing")
            )
            parsing["status"] = ProcessStatus.COMPLETED.value
            parsing["finished_at"] = finished_at
            parsing.pop("error", None)

            # Merge into status and use replace_jsonb_field for atomic update
            merged = {**current_status, "parsing": parsing, "parsing_status": parsing}
            merged["updated_at"] = iso_utc_z()
            normalized = _normalize_file_status(merged)

            replace_jsonb_field(file, "file_status", normalized)
            await session.flush()
            await session.refresh(file)
        return True

    async def fail_parsing_atomically(
        self,
        file_hash: str,
        error: str,
        finished_at: str
    ) -> bool:
        """
        Atomically update parsing status to failed with error message.
        Uses row-level locking to prevent race conditions.
        """
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(File).where(File.file_hash == file_hash).with_for_update()
            )
            file = result.scalar_one_or_none()
            if not file:
                return False

            current_status = _normalize_file_status(file.file_status or {})
            parsing = _copy_process_section(
                current_status.get("parsing_status") or current_status.get("parsing")
            )
            parsing["status"] = ProcessStatus.FAILED.value
            parsing["finished_at"] = finished_at
            parsing["error"] = error

            merged = {**current_status, "parsing": parsing, "parsing_status": parsing}
            merged["updated_at"] = iso_utc_z()
            normalized = _normalize_file_status(merged)

            replace_jsonb_field(file, "file_status", normalized)
            await session.flush()
        return True

    async def is_processing_complete(self, file_hash: str) -> bool:
        """Check if both parsing and indexing are completed."""
        status = await self.get_status(file_hash)
        if not status:
            return False

        parsing = status.get("parsing", {})
        indexing = status.get("indexing", {})

        return (
            parsing.get("status") == ProcessStatus.COMPLETED.value
            and indexing.get("status") == ProcessStatus.COMPLETED.value
        )
