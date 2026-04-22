"""
database.py - SQLite database for threads, files, and messages

This module provides async database operations for managing chat threads,
file associations, and message history for the RAG service.
"""

import os
import uuid
import logging
import json
import aiosqlite
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ProcessStatus(str, Enum):
    """Status values for processing operations (parsing, indexing, etc.)"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"

    @classmethod
    def is_completed(cls, status: str) -> bool:
        return status == cls.COMPLETED.value

    @classmethod
    def is_failed(cls, status: str) -> bool:
        return status == cls.FAILED.value

    @classmethod
    def is_running(cls, status: str) -> bool:
        return status == cls.RUNNING.value

    @classmethod
    def is_pending(cls, status: str) -> bool:
        return status == cls.PENDING.value

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        return status in (cls.COMPLETED.value, cls.FAILED.value)


# Database path - use /data for persistence in Docker
DATA_DIR = os.getenv("DATA_DIR")
if DATA_DIR is None:
    raise ValueError("DATA_DIR environment variable is not set")
DB_PATH = os.path.join(DATA_DIR, "rag.db")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


logger = logging.getLogger(__name__)


def _parse_settings(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _parse_json_list(raw: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Deserialize a JSON-encoded list from a SQLite text column."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return None
    except Exception:
        return None


def _default_process_section(status: str = ProcessStatus.UNKNOWN.value) -> Dict[str, Any]:
    """Return the default shape for a process-status section."""
    return {"status": status}


def _copy_process_section(section: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a shallow copy of a process-status section with a default status."""
    copied = dict(section or {})
    copied.setdefault("status", ProcessStatus.UNKNOWN.value)
    return copied


def _collapse_process_sections(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collapse many process-status sections into a single summary.

    Priority reflects the most actionable state for the UI:
    failed > running > pending > completed > unknown
    """
    if not sections:
        return _default_process_section()

    copied = [_copy_process_section(section) for section in sections]
    statuses = [section.get("status", ProcessStatus.UNKNOWN.value) for section in copied]

    if any(status == ProcessStatus.FAILED.value for status in statuses):
        status = ProcessStatus.FAILED.value
    elif any(status == ProcessStatus.RUNNING.value for status in statuses):
        status = ProcessStatus.RUNNING.value
    elif any(status == ProcessStatus.PENDING.value for status in statuses):
        status = ProcessStatus.PENDING.value
    elif all(status == ProcessStatus.COMPLETED.value for status in statuses):
        status = ProcessStatus.COMPLETED.value
    else:
        status = ProcessStatus.UNKNOWN.value

    summary: Dict[str, Any] = {"status": status}

    started_candidates = [
        section.get("started_at")
        for section in copied
        if section.get("started_at")
    ]
    finished_candidates = [
        section.get("finished_at")
        for section in copied
        if section.get("finished_at")
    ]
    if started_candidates:
        summary["started_at"] = min(started_candidates)
    if finished_candidates and all(
        section.get("status") == ProcessStatus.COMPLETED.value for section in copied
    ):
        summary["finished_at"] = max(finished_candidates)

    errors = [section.get("error") for section in copied if section.get("error")]
    if errors:
        summary["error"] = errors[-1]

    chunk_counts = [
        int(section.get("chunk_count", 0) or 0)
        for section in copied
        if section.get("chunk_count") is not None
    ]
    total_chars = [
        int(section.get("total_chars", 0) or 0)
        for section in copied
        if section.get("total_chars") is not None
    ]
    if chunk_counts:
        summary["chunk_count"] = max(chunk_counts)
    if total_chars:
        summary["total_chars"] = max(total_chars)

    return summary


def _normalize_file_status(status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize legacy and current file_status payloads into a single shape."""
    raw = dict(status or {})

    parsing = _copy_process_section(
        raw.get("parsing_status") if isinstance(raw.get("parsing_status"), dict) else raw.get("parsing")
    )

    raw_indexing_status = raw.get("indexing_status")
    if isinstance(raw_indexing_status, dict):
        raw_models = raw_indexing_status.get("models")
        raw_summary = raw_indexing_status.get("summary")
    else:
        raw_models = None
        raw_summary = None

    models: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_models, dict):
        for model_name, model_entry in raw_models.items():
            if not isinstance(model_entry, dict):
                continue
            normalized_model = _copy_process_section(model_entry)
            threads = model_entry.get("threads", {})
            normalized_threads: Dict[str, Dict[str, Any]] = {}
            if isinstance(threads, dict):
                for thread_id, thread_entry in threads.items():
                    if isinstance(thread_entry, dict):
                        normalized_threads[thread_id] = _copy_process_section(thread_entry)
            normalized_model["threads"] = normalized_threads
            models[model_name] = normalized_model

    summary = _copy_process_section(raw_summary if isinstance(raw_summary, dict) else raw.get("indexing"))
    if models:
        summary = _collapse_process_sections(list(models.values()))

    return {
        **raw,
        "parsing": parsing,
        "parsing_status": parsing,
        "indexing": summary,
        "indexing_status": {
            "summary": summary,
            "models": models,
        },
        "updated_at": raw.get("updated_at") or datetime.utcnow().isoformat(),
    }


def get_scoped_indexing_status(
    status: Optional[Dict[str, Any]],
    embedding_model: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Select the summary, model, or thread-specific indexing section."""
    normalized = _normalize_file_status(status)
    indexing_status = normalized.get("indexing_status", {})
    models = indexing_status.get("models", {})

    if embedding_model:
        model_status = _copy_process_section(models.get(embedding_model))
        if thread_id:
            return _copy_process_section(model_status.get("threads", {}).get(thread_id))
        return model_status

    if thread_id:
        thread_sections = []
        for model_status in models.values():
            if not isinstance(model_status, dict):
                continue
            threads = model_status.get("threads", {})
            if thread_id in threads and isinstance(threads[thread_id], dict):
                thread_sections.append(threads[thread_id])
        return _collapse_process_sections(thread_sections)

    return _copy_process_section(indexing_status.get("summary"))


class Thread(BaseModel):
    id: str
    name: str
    embed_model: str
    settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class File(BaseModel):
    file_hash: str
    file_name: str
    file_path: Optional[str] = None
    source_type: str = "pdf"  # 'pdf' or 'web'


class ThreadFile(BaseModel):
    thread_id: str
    file_hash: str


class ThreadFileAnnotation(BaseModel):
    thread_id: str
    file_hash: str
    annotations_json: str
    created_at: datetime
    updated_at: datetime


class Message(BaseModel):
    id: str
    thread_id: str
    role: MessageRole
    content: str
    context_compact: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_available: bool = False
    reasoning_format: str = "none"
    web_sources: Optional[List[Dict[str, Any]]] = None
    created_at: datetime


# SQL Schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    settings TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS files (
    file_hash TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_path TEXT,
    source_type TEXT NOT NULL DEFAULT 'pdf',
    file_status TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS thread_files (
    thread_id TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, file_hash),
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE,
    FOREIGN KEY (file_hash) REFERENCES files(file_hash)
);

CREATE TABLE IF NOT EXISTS thread_file_annotations (
    thread_id TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    annotations_json TEXT NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, file_hash),
    FOREIGN KEY (thread_id, file_hash)
        REFERENCES thread_files(thread_id, file_hash)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    context_compact TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_thread_files_thread_id ON thread_files(thread_id);
CREATE INDEX IF NOT EXISTS idx_thread_files_file_hash ON thread_files(file_hash);

CREATE TABLE IF NOT EXISTS thread_stats (
    thread_id       TEXT PRIMARY KEY,
    total_qa_pairs  INTEGER NOT NULL DEFAULT 0,
    total_qa_chars  INTEGER NOT NULL DEFAULT 0,
    avg_qa_chars    REAL    NOT NULL DEFAULT 0.0,
    last_qa_at      TIMESTAMP,
    documents_meta  TEXT    NOT NULL DEFAULT '{}',
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
);
"""


async def init_db():
    """Initialize the database with the schema."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.executescript(SCHEMA)

        # Lightweight migration for existing installations.
        migrations = [
            "ALTER TABLE messages ADD COLUMN reasoning TEXT",
            "ALTER TABLE messages ADD COLUMN reasoning_available INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE messages ADD COLUMN reasoning_format TEXT NOT NULL DEFAULT 'none'",
            "ALTER TABLE messages ADD COLUMN context_compact TEXT",
            "ALTER TABLE threads ADD COLUMN settings TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE messages ADD COLUMN web_sources TEXT",
            "ALTER TABLE files ADD COLUMN source_type TEXT NOT NULL DEFAULT 'pdf'",
            "ALTER TABLE files ADD COLUMN parsed_sentences_json TEXT",
            "ALTER TABLE files ADD COLUMN file_status TEXT NOT NULL DEFAULT '{}'",
            # thread_stats is created by SCHEMA above; new columns go here if needed later
        ]
        for stmt in migrations:
            try:
                await db.execute(stmt)
            except aiosqlite.OperationalError as e:
                # Ignore duplicate-column errors for already-migrated DBs.
                if "duplicate column name" not in str(e).lower():
                    raise
        await db.commit()
    logger.info(f"Database initialized at {DB_PATH}")


async def get_db():
    """Get a database connection."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys = ON")
    return db


# ============ Thread Operations ============

async def create_thread(name: str, embed_model: str) -> Thread:
    """Create a new thread."""
    thread_id = str(uuid.uuid4())
    created_at = datetime.utcnow()
    
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute(
            "INSERT INTO threads (id, name, embed_model, settings, created_at) VALUES (?, ?, ?, ?, ?)",
            (thread_id, name, embed_model, "{}", created_at)
        )
        await db.commit()
    
    return Thread(id=thread_id, name=name, embed_model=embed_model, settings={}, created_at=created_at)


async def get_thread(thread_id: str) -> Optional[Thread]:
    """Get a thread by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "SELECT id, name, embed_model, settings, created_at FROM threads WHERE id = ?",
            (thread_id,)
        )
        row = await cursor.fetchone()
        if row:
            return Thread(
                id=row["id"],
                name=row["name"],
                embed_model=row["embed_model"],
                settings=_parse_settings(row["settings"]),
                created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
            )
    return None


async def get_thread_settings(thread_id: str) -> Dict[str, Any]:
    """Get persisted settings for a thread."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "SELECT settings FROM threads WHERE id = ?",
            (thread_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return {}
        return _parse_settings(row["settings"])


async def update_thread_settings(thread_id: str, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Replace persisted settings for a thread."""
    payload = json.dumps(settings or {})
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "UPDATE threads SET settings = ? WHERE id = ?",
            (payload, thread_id)
        )
        await db.commit()
        if cursor.rowcount == 0:
            return None
    return await get_thread_settings(thread_id)


async def list_threads() -> List[Dict[str, Any]]:
    """List all threads with message counts and file counts."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute("""
            SELECT 
                t.id, t.name, t.embed_model, t.settings, t.created_at,
                COUNT(DISTINCT m.id) as message_count,
                COUNT(DISTINCT tf.file_hash) as file_count,
                MAX(m.created_at) as last_message_at
            FROM threads t
            LEFT JOIN messages m ON t.id = m.thread_id
            LEFT JOIN thread_files tf ON t.id = tf.thread_id
            GROUP BY t.id
            ORDER BY COALESCE(last_message_at, t.created_at) DESC
        """)
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "embed_model": row["embed_model"],
                "settings": _parse_settings(row["settings"]),
                "created_at": row["created_at"],
                "message_count": row["message_count"],
                "file_count": row["file_count"]
            }
            for row in rows
        ]


async def update_thread(thread_id: str, name: str) -> Optional[Thread]:
    """Update a thread's name."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute(
            "UPDATE threads SET name = ? WHERE id = ?",
            (name, thread_id)
        )
        await db.commit()
    return await get_thread(thread_id)


async def delete_thread(thread_id: str) -> bool:
    """Delete a thread and all associated data."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        # Delete messages first (cascade should handle this, but being explicit)
        await db.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        await db.execute("DELETE FROM thread_file_annotations WHERE thread_id = ?", (thread_id,))
        # Delete thread_files associations
        await db.execute("DELETE FROM thread_files WHERE thread_id = ?", (thread_id,))
        # Delete the thread
        cursor = await db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        await db.commit()
        return cursor.rowcount > 0


# ============ File Operations ============

async def create_or_get_file(
    file_hash: str,
    file_name: str,
    file_path: Optional[str] = None,
    source_type: str = "pdf",
) -> File:
    """Create a new file record or return existing one."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        # Try to insert, ignore if exists
        await db.execute(
            "INSERT OR IGNORE INTO files (file_hash, file_name, file_path, source_type) VALUES (?, ?, ?, ?)",
            (file_hash, file_name, file_path, source_type)
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


async def get_file(file_hash: str) -> Optional[File]:
    """Get a file by hash."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "SELECT file_hash, file_name, file_path, source_type FROM files WHERE file_hash = ?",
            (file_hash,)
        )
        row = await cursor.fetchone()
        if row:
            return File(
                file_hash=row["file_hash"],
                file_name=row["file_name"],
                file_path=row["file_path"],
                source_type=row["source_type"] or "pdf",
            )
    return None


async def update_file_parsed_sentences(file_hash: str, parsed_data_json: str) -> bool:
    """Store parsed sentences JSON in the files table."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "UPDATE files SET parsed_sentences_json = ? WHERE file_hash = ?",
            (parsed_data_json, file_hash)
        )
        await db.commit()
        return cursor.rowcount > 0


async def get_file_parsed_sentences(file_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieve parsed sentences JSON from the files table."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "SELECT parsed_sentences_json FROM files WHERE file_hash = ?",
            (file_hash,)
        )
        row = await cursor.fetchone()
        if row and row["parsed_sentences_json"]:
            try:
                return json.loads(row["parsed_sentences_json"])
            except Exception:
                return None
    return None


async def get_file_status(file_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieve file_status JSON from the files table."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "SELECT file_status FROM files WHERE file_hash = ?",
            (file_hash,)
        )
        row = await cursor.fetchone()
        if row and row["file_status"]:
            try:
                return _normalize_file_status(json.loads(row["file_status"]))
            except Exception:
                return _normalize_file_status({})
    return None


async def update_file_status(file_hash: str, status_data: Dict[str, Any]) -> bool:
    """Update file_status JSON for a file, merging with existing status."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        
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


async def _claim_file_status(
    file_hash: str,
    section: str,
    claim_status: str,
    started_at: Optional[str] = None,
) -> bool:
    """Atomically claim a process section if it is not already running or completed."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("BEGIN IMMEDIATE")

        cursor = await db.execute(
            "SELECT file_status FROM files WHERE file_hash = ?",
            (file_hash,),
        )
        row = await cursor.fetchone()
        if not row:
            await db.rollback()
            return False

        try:
            current = _normalize_file_status(json.loads(row["file_status"])) if row["file_status"] else _normalize_file_status({})
        except Exception:
            current = _normalize_file_status({})

        section_payload = _copy_process_section(current.get(section))
        current_status = section_payload.get("status", ProcessStatus.UNKNOWN.value)
        if ProcessStatus.is_running(current_status) or ProcessStatus.is_completed(current_status):
            await db.rollback()
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
            await db.rollback()
            return False
        await db.commit()
        return True


async def update_parsing_status(
    file_hash: str,
    status: str,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    error: Optional[str] = None,
    claim: bool = False,
) -> bool:
    """Update parsing section of file_status."""
    if claim:
        return await _claim_file_status(file_hash, "parsing", status, started_at=started_at)

    current_status = await get_file_status(file_hash) or {}
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
    return await update_file_status(
        file_hash,
        {
            "parsing": parsing,
            "parsing_status": parsing,
        },
    )


async def update_indexing_status(
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
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("BEGIN IMMEDIATE")

            cursor = await db.execute(
                "SELECT file_status FROM files WHERE file_hash = ?",
                (file_hash,),
            )
            row = await cursor.fetchone()
            if not row:
                await db.rollback()
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
                    await db.rollback()
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
                    await db.rollback()
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
                await db.rollback()
                return False
            await db.commit()
            return True

    current_status = _normalize_file_status(await get_file_status(file_hash) or {})
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

    return await update_file_status(
        file_hash,
        {
            "indexing": summary,
            "indexing_status": {
                "summary": summary,
                "models": models,
            },
        },
    )


async def remove_thread_indexing_status(file_hash: str, embedding_model: str, thread_id: str) -> bool:
    """Remove a thread-scoped indexing entry and recompute the remaining summaries."""
    current_status = _normalize_file_status(await get_file_status(file_hash) or {})
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
    return await update_file_status(
        file_hash,
        {
            "indexing": summary,
            "indexing_status": {
                "summary": summary,
                "models": models,
            },
        },
    )


# ============ Thread-File Association Operations ============

async def add_file_to_thread(thread_id: str, file_hash: str) -> bool:
    """Associate a file with a thread."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                "INSERT OR IGNORE INTO thread_files (thread_id, file_hash) VALUES (?, ?)",
                (thread_id, file_hash)
            )
            await db.commit()
            return True
    except Exception as e:
        logger.error(f"Error adding file to thread: {e}")
        return False


async def get_thread_files(thread_id: str) -> List[File]:
    """Get all files associated with a thread."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute("""
            SELECT f.file_hash, f.file_name, f.file_path, f.source_type
            FROM files f
            JOIN thread_files tf ON f.file_hash = tf.file_hash
            WHERE tf.thread_id = ?
            ORDER BY tf.added_at DESC
        """, (thread_id,))
        rows = await cursor.fetchall()
        return [
            File(
                file_hash=row["file_hash"],
                file_name=row["file_name"],
                file_path=row["file_path"],
                source_type=row["source_type"] or "pdf",
            )
            for row in rows
        ]


async def remove_file_from_thread(thread_id: str, file_hash: str) -> bool:
    """Remove a file association from a thread (does not delete the file record itself)."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                "DELETE FROM thread_file_annotations WHERE thread_id = ? AND file_hash = ?",
                (thread_id, file_hash)
            )
            cursor = await db.execute(
                "DELETE FROM thread_files WHERE thread_id = ? AND file_hash = ?",
                (thread_id, file_hash)
            )
            await db.commit()
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error removing file from thread: {e}")
        return False


async def is_file_in_thread(thread_id: str, file_hash: str) -> bool:
    """Check if a file is associated with a thread."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "SELECT 1 FROM thread_files WHERE thread_id = ? AND file_hash = ?",
            (thread_id, file_hash)
        )
        row = await cursor.fetchone()
        return row is not None


async def count_threads_with_file_for_model(
    file_hash: str,
    embed_model: str,
    exclude_thread_id: Optional[str] = None,
) -> int:
    """
    Count thread associations for a file restricted to a specific embedding model.
    Optionally exclude one thread (useful after detaching a file from that thread).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        if exclude_thread_id:
            cursor = await db.execute(
                """
                SELECT COUNT(*)
                FROM thread_files tf
                JOIN threads t ON t.id = tf.thread_id
                WHERE tf.file_hash = ? AND t.embed_model = ? AND tf.thread_id != ?
                """,
                (file_hash, embed_model, exclude_thread_id),
            )
        else:
            cursor = await db.execute(
                """
                SELECT COUNT(*)
                FROM thread_files tf
                JOIN threads t ON t.id = tf.thread_id
                WHERE tf.file_hash = ? AND t.embed_model = ?
                """,
                (file_hash, embed_model),
            )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0


async def count_threads_with_file(file_hash: str) -> int:
    """Count how many threads currently reference a file."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "SELECT COUNT(*) FROM thread_files WHERE file_hash = ?",
            (file_hash,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0


async def delete_file_record(file_hash: str) -> bool:
    """Delete a file row once all thread associations have been removed."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "DELETE FROM files WHERE file_hash = ?",
            (file_hash,),
        )
        await db.commit()
        return cursor.rowcount > 0


def _load_thread_file_annotations(raw: Optional[str]) -> List[Dict[str, Any]]:
    """Deserialize the annotation snapshot list from SQLite."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _serialize_thread_file_annotation_row(row: ThreadFileAnnotation) -> Dict[str, Any]:
    """Convert an annotation row into the API payload shape."""
    return {
        "thread_id": row.thread_id,
        "file_hash": row.file_hash,
        "annotations": _load_thread_file_annotations(row.annotations_json),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


async def get_thread_file_annotation_row(
    thread_id: str,
    file_hash: str,
) -> Optional[ThreadFileAnnotation]:
    """Load the persisted row for a thread/file annotation snapshot."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            """
            SELECT thread_id, file_hash, annotations_json, created_at, updated_at
            FROM thread_file_annotations
            WHERE thread_id = ? AND file_hash = ?
            """,
            (thread_id, file_hash),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return ThreadFileAnnotation(
            thread_id=row["thread_id"],
            file_hash=row["file_hash"],
            annotations_json=row["annotations_json"] or "[]",
            created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
            updated_at=datetime.fromisoformat(row["updated_at"]) if isinstance(row["updated_at"], str) else row["updated_at"],
        )


async def get_thread_file_annotations(thread_id: str, file_hash: str) -> Optional[Dict[str, Any]]:
    """Get the persisted annotation payload for a thread/file pair."""
    row = await get_thread_file_annotation_row(thread_id, file_hash)
    if not row:
        return None
    return _serialize_thread_file_annotation_row(row)


async def upsert_thread_file_annotations(
    thread_id: str,
    file_hash: str,
    annotations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Insert or replace the full annotation snapshot for a thread/file pair."""
    annotations_json = json.dumps(annotations or [])
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute(
            """
            INSERT INTO thread_file_annotations (
                thread_id, file_hash, annotations_json, created_at, updated_at
            ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(thread_id, file_hash) DO UPDATE SET
                annotations_json = excluded.annotations_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (thread_id, file_hash, annotations_json),
        )
        await db.commit()

    row = await get_thread_file_annotation_row(thread_id, file_hash)
    if not row:
        raise RuntimeError("Failed to persist annotation snapshot")
    return _serialize_thread_file_annotation_row(row)


async def delete_thread_file_annotations(thread_id: str, file_hash: Optional[str] = None) -> int:
    """Delete persisted annotations for a thread or thread/file pair."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        if file_hash:
            cursor = await db.execute(
                "DELETE FROM thread_file_annotations WHERE thread_id = ? AND file_hash = ?",
                (thread_id, file_hash),
            )
        else:
            cursor = await db.execute(
                "DELETE FROM thread_file_annotations WHERE thread_id = ?",
                (thread_id,),
            )
        await db.commit()
        return cursor.rowcount or 0


# ============ Message Operations ============

async def create_message(
    thread_id: str,
    role: MessageRole,
    content: str,
    context_compact: Optional[str] = None,
    reasoning: Optional[str] = None,
    reasoning_available: bool = False,
    reasoning_format: str = "none",
    web_sources: Optional[List[Dict[str, Any]]] = None,
) -> Message:
    """Create a new message in a thread."""
    message_id = str(uuid.uuid4())
    created_at = datetime.utcnow()
    web_sources_json = json.dumps(web_sources) if web_sources else None

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO messages (
                id, thread_id, role, content, context_compact, reasoning, reasoning_available, reasoning_format, web_sources, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                thread_id,
                role.value,
                content,
                context_compact,
                reasoning,
                int(reasoning_available),
                reasoning_format,
                web_sources_json,
                created_at,
            )
        )
        await db.commit()

    return Message(
        id=message_id,
        thread_id=thread_id,
        role=role,
        content=content,
        context_compact=context_compact,
        reasoning=reasoning,
        reasoning_available=reasoning_available,
        reasoning_format=reasoning_format,
        web_sources=web_sources,
        created_at=created_at
    )


async def get_message(message_id: str) -> Optional[Message]:
    """Get a message by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT id, thread_id, role, content, reasoning, reasoning_available, reasoning_format,
                   context_compact, web_sources, created_at
            FROM messages
            WHERE id = ?
            """,
            (message_id,)
        )
        row = await cursor.fetchone()
        if row:
            return Message(
                id=row["id"],
                thread_id=row["thread_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                context_compact=row["context_compact"],
                reasoning=row["reasoning"],
                reasoning_available=bool(row["reasoning_available"]),
                reasoning_format=row["reasoning_format"] or "none",
                web_sources=_parse_json_list(row["web_sources"]),
                created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
            )
    return None


async def get_thread_messages(
    thread_id: str,
    limit: int = 100,
    offset: int = 0
) -> List[Message]:
    """Get messages for a thread with pagination."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT id, thread_id, role, content, context_compact, reasoning, reasoning_available,
                   reasoning_format, web_sources, created_at
            FROM messages
            WHERE thread_id = ?
            ORDER BY created_at ASC
            LIMIT ? OFFSET ?
        """, (thread_id, limit, offset))
        rows = await cursor.fetchall()
        return [
            Message(
                id=row["id"],
                thread_id=row["thread_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                context_compact=row["context_compact"],
                reasoning=row["reasoning"],
                reasoning_available=bool(row["reasoning_available"]),
                reasoning_format=row["reasoning_format"] or "none",
                web_sources=_parse_json_list(row["web_sources"]),
                created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
            )
            for row in rows
        ]


async def get_recent_messages(thread_id: str, limit: int = 10) -> List[Message]:
    """Get the most recent messages for a thread (for context window)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT id, thread_id, role, content, context_compact, reasoning, reasoning_available,
                   reasoning_format, web_sources, created_at
            FROM messages
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (thread_id, limit))
        rows = await cursor.fetchall()
        # Reverse to get chronological order
        messages = [
            Message(
                id=row["id"],
                thread_id=row["thread_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                context_compact=row["context_compact"],
                reasoning=row["reasoning"],
                reasoning_available=bool(row["reasoning_available"]),
                reasoning_format=row["reasoning_format"] or "none",
                web_sources=_parse_json_list(row["web_sources"]),
                created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
            )
            for row in rows
        ]
        return list(reversed(messages))


async def update_message_context_compact(message_id: str, context_compact: str) -> bool:
    """Update compact context text for a message."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "UPDATE messages SET context_compact = ? WHERE id = ?",
            (context_compact, message_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def delete_message(message_id: str) -> bool:
    """Delete a message by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        await db.commit()
        return cursor.rowcount > 0


async def delete_message_pair(message_id: str) -> List[str]:
    """
    Delete a message and its paired question/answer.
    Returns a list of IDs that were deleted.
    """
    deleted_ids = []
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        # 1. Get the target message
        cursor = await db.execute("SELECT id, thread_id, role, created_at FROM messages WHERE id = ?", (message_id,))
        target = await cursor.fetchone()
        if not target:
            return []
        
        target_role = target["role"]
        target_thread = target["thread_id"]
        target_created = target["created_at"]
        
        deleted_ids.append(message_id)
        
        # 2. Find the candidate pair
        pair_id = None
        if target_role == "assistant":
            # Search for the user message immediately preceding it
            cursor = await db.execute("""
                SELECT id FROM messages 
                WHERE thread_id = ? AND role = 'user' AND created_at <= ? AND id != ?
                ORDER BY created_at DESC LIMIT 1
            """, (target_thread, target_created, message_id))
            pair = await cursor.fetchone()
            if pair:
                pair_id = pair["id"]
        elif target_role == "user":
            # Search for the assistant message immediately following it
            cursor = await db.execute("""
                SELECT id FROM messages 
                WHERE thread_id = ? AND role = 'assistant' AND created_at >= ? AND id != ?
                ORDER BY created_at ASC LIMIT 1
            """, (target_thread, target_created, message_id))
            pair = await cursor.fetchone()
            if pair:
                pair_id = pair["id"]
        
        if pair_id:
            deleted_ids.append(pair_id)
            
        # 3. Perform the deletion
        placeholders = ', '.join(['?'] * len(deleted_ids))
        await db.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", deleted_ids)
        await db.commit()
        
    return deleted_ids


async def get_message_count(thread_id: str) -> int:
    """Get the total number of messages in a thread."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM messages WHERE thread_id = ?",
            (thread_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0


# ============ Thread Stats Operations ============

def _load_documents_meta(raw: Optional[str]) -> Dict[str, Any]:
    """Deserialize the documents_meta JSON column."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


async def _ensure_thread_stats_row(db, thread_id: str) -> None:
    """Insert a thread_stats row if one doesn't exist yet."""
    await db.execute(
        "INSERT OR IGNORE INTO thread_stats (thread_id) VALUES (?)",
        (thread_id,),
    )


async def remove_document_from_stats(thread_id: str, file_hash: str) -> None:
    """Remove a document entry from thread_stats.documents_meta."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT documents_meta FROM thread_stats WHERE thread_id = ?",
            (thread_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return
        docs = _load_documents_meta(row["documents_meta"])
        docs.pop(file_hash, None)
        await db.execute(
            """
            UPDATE thread_stats
            SET documents_meta = ?, last_updated_at = CURRENT_TIMESTAMP
            WHERE thread_id = ?
            """,
            (json.dumps(docs), thread_id),
        )
        await db.commit()


async def upsert_document_in_stats(thread_id: str, file_hash: str, meta: Dict[str, Any]) -> None:
    """Insert or replace a document entry in thread_stats.documents_meta."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_thread_stats_row(db, thread_id)
        cursor = await db.execute(
            "SELECT documents_meta FROM thread_stats WHERE thread_id = ?",
            (thread_id,),
        )
        row = await cursor.fetchone()
        docs = _load_documents_meta(row["documents_meta"] if row else None)
        docs[file_hash] = {
            **docs.get(file_hash, {}),
            **(meta or {}),
        }
        await db.execute(
            """
            UPDATE thread_stats
            SET documents_meta = ?, last_updated_at = CURRENT_TIMESTAMP
            WHERE thread_id = ?
            """,
            (json.dumps(docs), thread_id),
        )
        await db.commit()


async def increment_qa_stats(thread_id: str, qa_chars: int) -> None:
    """
    Increment QA aggregate counters after each answered turn.
    Called on the hot path (every chat answer).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_thread_stats_row(db, thread_id)
        await db.execute(
            """
            UPDATE thread_stats
            SET
                total_qa_pairs  = total_qa_pairs + 1,
                total_qa_chars  = total_qa_chars + ?,
                avg_qa_chars    = CAST(total_qa_chars + ? AS REAL) / (total_qa_pairs + 1),
                last_qa_at      = CURRENT_TIMESTAMP,
                last_updated_at = CURRENT_TIMESTAMP
            WHERE thread_id = ?
            """,
            (qa_chars, qa_chars, thread_id),
        )
        await db.commit()


async def recompute_qa_stats(thread_id: str) -> None:
    """
    Recompute QA stats from the messages table.
    Called after message pair deletion (rare path) to prevent drift.
    Counts assistant messages as QA pairs and sums their content chars.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT COUNT(*) as cnt, COALESCE(SUM(LENGTH(content)), 0) as total_chars
            FROM messages
            WHERE thread_id = ? AND role = 'assistant'
            """,
            (thread_id,),
        )
        row = await cursor.fetchone()
        cnt = row["cnt"] or 0
        total_chars = row["total_chars"] or 0
        avg = (total_chars / cnt) if cnt > 0 else 0.0

        await _ensure_thread_stats_row(db, thread_id)
        await db.execute(
            """
            UPDATE thread_stats
            SET total_qa_pairs  = ?,
                total_qa_chars  = ?,
                avg_qa_chars    = ?,
                last_updated_at = CURRENT_TIMESTAMP
            WHERE thread_id = ?
            """,
            (cnt, total_chars, avg, thread_id),
        )
        await db.commit()


async def get_thread_shape(thread_id: str) -> Dict[str, Any]:
    """
    Return a structured snapshot of the thread's content inventory.
    Used by the prefetch path in chat_service and by the get_thread_shape agent tool.

    Returns:
        {
          "total_qa_pairs": int,
          "total_qa_chars": int,
          "avg_qa_chars": float,
          "last_qa_at": str | None,
          "documents": {
              "<file_hash>": {
                  "file_name": str,
                  "source_type": str,
                  "chunk_count": int,
                  "total_chars": int,
                  "indexing_status": str,
                  "indexed_at": str | None
              }
          }
        }
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT total_qa_pairs, total_qa_chars, avg_qa_chars,
                   last_qa_at, documents_meta
            FROM thread_stats WHERE thread_id = ?
            """,
            (thread_id,),
        )
        row = await cursor.fetchone()

    if not row:
        return {
            "total_qa_pairs": 0,
            "total_qa_chars": 0,
            "avg_qa_chars": 0.0,
            "last_qa_at": None,
            "documents": {},
        }

    return {
        "total_qa_pairs": row["total_qa_pairs"],
        "total_qa_chars": row["total_qa_chars"],
        "avg_qa_chars": round(row["avg_qa_chars"], 1),
        "last_qa_at": row["last_qa_at"],
        "documents": _load_documents_meta(row["documents_meta"]),
    }
