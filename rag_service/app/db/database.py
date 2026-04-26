"""
database.py - Compatibility shim for refactored database module.

DEPRECATED: This file exists for backward compatibility during migration.
Please import from app.db instead.

Example:
    OLD: from app.db.database import create_thread
    NEW: from app.db import create_thread
"""

import warnings

# Re-export everything from the new public API
from app.db import (
    # Models
    ProcessStatus,
    MessageRole,
    Thread,
    File,
    ThreadFile,
    ThreadFileAnnotation,
    Message,
    # Config
    DB_PATH,
    init_db,
    get_db,
    # Status
    get_scoped_indexing_status,
    # Thread operations
    create_thread,
    get_thread,
    get_thread_settings,
    update_thread_settings,
    list_threads,
    update_thread,
    delete_thread,
    # File operations
    create_or_get_file,
    get_file,
    update_file_parsed_sentences,
    get_file_parsed_sentences,
    get_file_status,
    update_file_status,
    update_parsing_status,
    update_indexing_status,
    remove_thread_indexing_status,
    delete_file_record,
    # Thread-file operations
    add_file_to_thread,
    get_thread_files,
    remove_file_from_thread,
    is_file_in_thread,
    count_threads_with_file_for_model,
    count_threads_with_file,
    get_thread_file_annotations,
    upsert_thread_file_annotations,
    delete_thread_file_annotations,
    # Message operations
    create_message,
    get_message,
    get_thread_messages,
    get_recent_messages,
    update_message_context_compact,
    delete_message,
    delete_message_pair,
    get_message_count,
    # Stats operations
    remove_document_from_stats,
    upsert_document_in_stats,
    increment_qa_stats,
    recompute_qa_stats,
    get_thread_shape,
)

# Emit deprecation warning
warnings.warn(
    "Importing from app.db.database is deprecated. "
    "Please import from app.db instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

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
