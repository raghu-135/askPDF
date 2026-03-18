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
    source_type TEXT NOT NULL DEFAULT 'pdf'
);

CREATE TABLE IF NOT EXISTS thread_files (
    thread_id TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, file_hash),
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE,
    FOREIGN KEY (file_hash) REFERENCES files(file_hash)
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
    return db


# ============ Thread Operations ============

async def create_thread(name: str, embed_model: str) -> Thread:
    """Create a new thread."""
    thread_id = str(uuid.uuid4())
    created_at = datetime.utcnow()
    
    async with aiosqlite.connect(DB_PATH) as db:
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
        await db.execute(
            "UPDATE threads SET name = ? WHERE id = ?",
            (name, thread_id)
        )
        await db.commit()
    return await get_thread(thread_id)


async def delete_thread(thread_id: str) -> bool:
    """Delete a thread and all associated data."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Delete messages first (cascade should handle this, but being explicit)
        await db.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
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
        # Try to insert, ignore if exists
        await db.execute(
            "INSERT OR IGNORE INTO files (file_hash, file_name, file_path, source_type) VALUES (?, ?, ?, ?)",
            (file_hash, file_name, file_path, source_type)
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


# ============ Thread-File Association Operations ============

async def add_file_to_thread(thread_id: str, file_hash: str) -> bool:
    """Associate a file with a thread."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
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
        cursor = await db.execute(
            "SELECT 1 FROM thread_files WHERE thread_id = ? AND file_hash = ?",
            (thread_id, file_hash)
        )
        row = await cursor.fetchone()
        return row is not None


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


async def upsert_thread_stats_document(
    thread_id: str,
    file_hash: str,
    file_name: str,
    source_type: str = "pdf",
    content_hash: Optional[str] = None,
) -> None:
    """
    Insert or upsert a document entry in thread_stats.documents_meta.
    Called immediately when a file is added to the thread (status=pending).

    If content_hash is provided this is a refresh-reset: chunk_count and
    indexing_status are cleared back to pending so agents see fresh state.
    When no content_hash is given (normal add), existing fields are preserved.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_thread_stats_row(db, thread_id)

        cursor = await db.execute(
            "SELECT documents_meta FROM thread_stats WHERE thread_id = ?",
            (thread_id,),
        )
        row = await cursor.fetchone()
        docs = _load_documents_meta(row["documents_meta"] if row else None)

        existing = docs.get(file_hash, {})
        if content_hash is not None:
            # Refresh-reset: clear stale indexing fields, store new content hash
            docs[file_hash] = {
                "file_name": file_name,
                "source_type": source_type,
                "chunk_count": 0,
                "total_chars": 0,
                "indexing_status": "pending",
                "indexed_at": None,
                "content_hash": content_hash,
            }
        else:
            # Normal add: preserve existing indexed fields if doc was already indexed
            docs[file_hash] = {
                "file_name": file_name,
                "source_type": source_type,
                "chunk_count": existing.get("chunk_count", 0),
                "total_chars": existing.get("total_chars", 0),
                "indexing_status": existing.get("indexing_status", "pending"),
                "indexed_at": existing.get("indexed_at"),
                "content_hash": existing.get("content_hash"),
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


async def update_document_indexing_status(
    thread_id: str,
    file_hash: str,
    status: str,
    chunk_count: int = 0,
    total_chars: int = 0,
    indexed_at: Optional[str] = None,
) -> None:
    """
    Update a document's indexing status and size metadata after background
    indexing completes (status='indexed') or fails (status='failed').
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await _ensure_thread_stats_row(db, thread_id)

        cursor = await db.execute(
            "SELECT documents_meta FROM thread_stats WHERE thread_id = ?",
            (thread_id,),
        )
        row = await cursor.fetchone()
        docs = _load_documents_meta(row["documents_meta"] if row else None)

        entry = docs.get(file_hash, {})
        entry["indexing_status"] = status
        if status == "indexed":
            entry["chunk_count"] = chunk_count
            entry["total_chars"] = total_chars
            entry["indexed_at"] = indexed_at or datetime.utcnow().isoformat()
        docs[file_hash] = entry

        await db.execute(
            """
            UPDATE thread_stats
            SET documents_meta = ?, last_updated_at = CURRENT_TIMESTAMP
            WHERE thread_id = ?
            """,
            (json.dumps(docs), thread_id),
        )
        await db.commit()


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
