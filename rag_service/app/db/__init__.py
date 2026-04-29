"""
app.db - Public API for database operations (PostgreSQL/SQLModel).

This module provides a clean public API for database operations,
using SQLModel with PostgreSQL as the primary database.
"""

# Models and Enums (SQLModel-based)
from app.db.models_sqlmodel import (
    ProcessStatus,
    MessageRole,
    Thread,
    File,
    ThreadFile,
    ThreadFileAnnotation,
    Message,
    ThreadStats,
)

# Connection management (SQLModel/PostgreSQL)
from app.db.connection_sqlmodel import (
    init_db,
    close_db,
    get_session,
    async_session_maker,
    engine,
    test_engine,
)

# Status helpers
from app.db.status import get_scoped_indexing_status

# Constants (from file_repo for backward compatibility)
DEFAULT_SENTENCES_JSON = '{"version": "1.0", "sentences": []}'
DEFAULT_FILE_STATUS = {
    "file_hash": "",
    "parsing": {"status": "unknown"},
    "indexing": {"status": "unknown"},
    "updated_at": None,
}

# Repository instances (singleton pattern - SQLModel versions)
_thread_repo = None
_file_repo = None
_message_repo = None
_thread_file_repo = None
_stats_repo = None


def get_thread_repo():
    """Get the thread repository instance."""
    global _thread_repo
    if _thread_repo is None:
        from app.db.repositories.thread_repo_sqlmodel import ThreadRepository
        _thread_repo = ThreadRepository()
    return _thread_repo


def get_file_repo():
    """Get the file repository instance."""
    global _file_repo
    if _file_repo is None:
        from app.db.repositories.file_repo_sqlmodel import FileRepository
        _file_repo = FileRepository()
    return _file_repo


def get_message_repo():
    """Get the message repository instance."""
    global _message_repo
    if _message_repo is None:
        from app.db.repositories.message_repo_sqlmodel import MessageRepository
        _message_repo = MessageRepository()
    return _message_repo


def get_thread_file_repo():
    """Get the thread-file repository instance."""
    global _thread_file_repo
    if _thread_file_repo is None:
        from app.db.repositories.thread_file_repo_sqlmodel import ThreadFileRepository
        _thread_file_repo = ThreadFileRepository()
    return _thread_file_repo


def get_stats_repo():
    """Get the stats repository instance."""
    global _stats_repo
    if _stats_repo is None:
        from app.db.repositories.stats_repo_sqlmodel import StatsRepository
        _stats_repo = StatsRepository()
    return _stats_repo


# Thread operations
async def create_thread(name: str, embed_model: str):
    """Create a new thread."""
    return await get_thread_repo().create(name, embed_model)


async def get_thread(thread_id: str):
    """Get a thread by ID."""
    return await get_thread_repo().get(thread_id)


async def get_thread_settings(thread_id: str):
    """Get persisted settings for a thread."""
    return await get_thread_repo().get_settings(thread_id)


async def update_thread_settings(thread_id: str, settings: dict):
    """Replace persisted settings for a thread."""
    return await get_thread_repo().update_settings(thread_id, settings)


async def list_threads():
    """List all threads with message counts and file counts."""
    return await get_thread_repo().list_all()


async def update_thread(thread_id: str, name: str):
    """Update a thread's name."""
    return await get_thread_repo().update(thread_id, name)


async def delete_thread(thread_id: str):
    """Delete a thread and all associated data."""
    return await get_thread_repo().delete(thread_id)


# File operations
async def create_or_get_file(file_hash: str, file_name: str, file_path: str = None, source_type: str = "pdf"):
    """Create a new file record or return existing one."""
    return await get_file_repo().create_or_get(file_hash, file_name, file_path, source_type)


async def get_file(file_hash: str):
    """Get a file by hash."""
    return await get_file_repo().get(file_hash)


async def update_file_parsed_sentences(file_hash: str, parsed_data_json: str):
    """Store parsed sentences JSON in the files table."""
    return await get_file_repo().update_parsed_sentences(file_hash, parsed_data_json)


async def get_file_parsed_sentences(file_hash: str):
    """Retrieve parsed sentences JSON from the files table."""
    return await get_file_repo().get_parsed_sentences(file_hash)


async def get_file_status(file_hash: str):
    """Retrieve file_status JSON from the files table."""
    return await get_file_repo().get_status(file_hash)


async def update_file_status(file_hash: str, status_data: dict):
    """Update file_status JSON for a file, merging with existing status."""
    return await get_file_repo().update_status(file_hash, status_data)


async def update_parsing_status(
    file_hash: str,
    status: str,
    started_at: str = None,
    finished_at: str = None,
    error: str = None,
    claim: bool = False,
):
    """Update parsing section of file_status."""
    return await get_file_repo().update_parsing_status(file_hash, status, started_at, finished_at, error, claim)


async def update_indexing_status(
    file_hash: str,
    status: str,
    embedding_model: str = None,
    thread_id: str = None,
    started_at: str = None,
    finished_at: str = None,
    error: str = None,
    chunk_count: int = None,
    total_chars: int = None,
    reused_existing_embeddings: bool = None,
    claim: bool = False,
):
    """Update indexing section of file_status."""
    return await get_file_repo().update_indexing_status(
        file_hash, status, embedding_model, thread_id, started_at, finished_at,
        error, chunk_count, total_chars, reused_existing_embeddings, claim
    )


async def remove_thread_indexing_status(file_hash: str, embedding_model: str, thread_id: str):
    """Remove a thread-scoped indexing entry and recompute the remaining summaries."""
    return await get_file_repo().remove_thread_indexing_status(file_hash, embedding_model, thread_id)


async def delete_file_record(file_hash: str):
    """Delete a file row once all thread associations have been removed."""
    return await get_file_repo().delete(file_hash)


# Thread-file operations
async def add_file_to_thread(thread_id: str, file_hash: str):
    """Associate a file with a thread."""
    return await get_thread_file_repo().add(thread_id, file_hash)


async def get_thread_files(thread_id: str):
    """Get all files associated with a thread."""
    return await get_thread_file_repo().get_files(thread_id)


async def remove_file_from_thread(thread_id: str, file_hash: str):
    """Remove a file association from a thread (does not delete the file record itself)."""
    return await get_thread_file_repo().remove(thread_id, file_hash)


async def is_file_in_thread(thread_id: str, file_hash: str):
    """Check if a file is associated with a thread."""
    return await get_thread_file_repo().is_file_in_thread(thread_id, file_hash)


async def count_threads_with_file_for_model(file_hash: str, embed_model: str, exclude_thread_id: str = None):
    """Count thread associations for a file restricted to a specific embedding model."""
    return await get_thread_file_repo().count_threads_with_file_for_model(file_hash, embed_model, exclude_thread_id)


async def count_threads_with_file(file_hash: str):
    """Count how many threads currently reference a file."""
    return await get_thread_file_repo().count_threads_with_file(file_hash)


async def get_thread_file_annotations(thread_id: str, file_hash: str):
    """Get the persisted annotation payload for a thread/file pair."""
    return await get_thread_file_repo().get_annotations(thread_id, file_hash)


async def upsert_thread_file_annotations(thread_id: str, file_hash: str, annotations: list):
    """Insert or replace the full annotation snapshot for a thread/file pair."""
    return await get_thread_file_repo().upsert_annotations(thread_id, file_hash, annotations)


async def delete_thread_file_annotations(thread_id: str, file_hash: str = None):
    """Delete persisted annotations for a thread or thread/file pair."""
    return await get_thread_file_repo().delete_annotations(thread_id, file_hash)


# Message operations
async def create_message(
    thread_id: str,
    role,
    content: str,
    context_compact: str = None,
    reasoning: str = None,
    reasoning_available: bool = False,
    reasoning_format: str = "none",
    web_sources: list = None,
):
    """Create a new message in a thread."""
    return await get_message_repo().create(
        thread_id, role, content, context_compact, reasoning,
        reasoning_available, reasoning_format, web_sources
    )


async def get_message(message_id: str):
    """Get a message by ID."""
    return await get_message_repo().get(message_id)


async def get_thread_messages(thread_id: str, limit: int = 100, offset: int = 0):
    """Get messages for a thread with pagination."""
    return await get_message_repo().get_thread_messages(thread_id, limit, offset)


async def get_recent_messages(thread_id: str, limit: int = 10):
    """Get the most recent messages for a thread (for context window)."""
    return await get_message_repo().get_recent_messages(thread_id, limit)


async def update_message_context_compact(message_id: str, context_compact: str):
    """Update compact context text for a message."""
    return await get_message_repo().update_context_compact(message_id, context_compact)


async def delete_message(message_id: str):
    """Delete a message by ID."""
    return await get_message_repo().delete(message_id)


async def delete_message_pair(message_id: str):
    """Delete a message and its paired question/answer."""
    return await get_message_repo().delete_pair(message_id)


async def get_message_count(thread_id: str):
    """Get the total number of messages in a thread."""
    return await get_message_repo().get_count(thread_id)


# Stats operations
async def remove_document_from_stats(thread_id: str, file_hash: str):
    """Remove a document entry from thread_stats.documents_meta."""
    return await get_stats_repo().remove_document_from_stats(thread_id, file_hash)


async def upsert_document_in_stats(thread_id: str, file_hash: str, meta: dict):
    """Insert or replace a document entry in thread_stats.documents_meta."""
    return await get_stats_repo().upsert_document_in_stats(thread_id, file_hash, meta)


async def increment_qa_stats(thread_id: str, qa_chars: int):
    """Increment QA aggregate counters after each answered turn."""
    return await get_stats_repo().increment_qa_stats(thread_id, qa_chars)


async def recompute_qa_stats(thread_id: str):
    """Recompute QA stats from the messages table."""
    return await get_stats_repo().recompute_qa_stats(thread_id)


async def get_thread_shape(thread_id: str):
    """Return a structured snapshot of the thread's content inventory."""
    return await get_stats_repo().get_thread_shape(thread_id)


__all__ = [
    # Models
    "ProcessStatus",
    "MessageRole",
    "Thread",
    "File",
    "ThreadFile",
    "ThreadFileAnnotation",
    "Message",
    # Config
    "DB_PATH",
    "init_db",
    "get_db",
    # Status
    "get_scoped_indexing_status",
    # Constants
    "DEFAULT_SENTENCES_JSON",
    "DEFAULT_FILE_STATUS",
    # Thread operations
    "create_thread",
    "get_thread",
    "get_thread_settings",
    "update_thread_settings",
    "list_threads",
    "update_thread",
    "delete_thread",
    # File operations
    "create_or_get_file",
    "get_file",
    "update_file_parsed_sentences",
    "get_file_parsed_sentences",
    "get_file_status",
    "update_file_status",
    "update_parsing_status",
    "update_indexing_status",
    "remove_thread_indexing_status",
    "delete_file_record",
    # Thread-file operations
    "add_file_to_thread",
    "get_thread_files",
    "remove_file_from_thread",
    "is_file_in_thread",
    "count_threads_with_file_for_model",
    "count_threads_with_file",
    "get_thread_file_annotations",
    "upsert_thread_file_annotations",
    "delete_thread_file_annotations",
    # Message operations
    "create_message",
    "get_message",
    "get_thread_messages",
    "get_recent_messages",
    "update_message_context_compact",
    "delete_message",
    "delete_message_pair",
    "get_message_count",
    # Stats operations
    "remove_document_from_stats",
    "upsert_document_in_stats",
    "increment_qa_stats",
    "recompute_qa_stats",
    "get_thread_shape",
]
