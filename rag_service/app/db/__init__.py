"""
app.db - Public API for database operations (PostgreSQL/SQLModel).

This module provides a clean public API for database operations,
using SQLModel with PostgreSQL as the primary database.
"""

# Models and Enums (SQLModel-based)
from app.db.models_sqlmodel import (
    ProcessStatus,
    MessageRole,
    FileSourceType,
    ChatTurnStatus,
    WorkflowVisibility,
    AgentRunStatus,
    MemoryCandidateStatus,
    MemoryScopeType,
    MemoryStatus,
    MemoryType,
    MemoryVisibility,
    Project,
    Thread,
    File,
    ThreadFile,
    ProjectFile,
    ThreadDocumentAnnotation,
    ChatTurn,
    AgentWorkflow,
    AgentRun,
    Memory,
    MemoryEvent,
    MemoryCandidate,
)
from app.db.enums import (
    EmbeddingReadinessStatus,
    FileStatusSection,
    OperationResultStatus,
    ReasoningFormat,
    ThreadCloneMode,
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
_agent_workflow_repo = None
_project_repo = None
_project_file_repo = None
_memory_repo = None


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


def get_agent_workflow_repo():
    """Get the agent workflow repository instance."""
    global _agent_workflow_repo
    if _agent_workflow_repo is None:
        from app.agent_workflows.repository import AgentWorkflowRepository
        _agent_workflow_repo = AgentWorkflowRepository()
    return _agent_workflow_repo


def get_project_repo():
    """Get the project repository instance."""
    global _project_repo
    if _project_repo is None:
        from app.db.repositories.project_repo_sqlmodel import ProjectRepository
        _project_repo = ProjectRepository()
    return _project_repo


def get_project_file_repo():
    """Get the project-file repository instance."""
    global _project_file_repo
    if _project_file_repo is None:
        from app.db.repositories.project_file_repo_sqlmodel import ProjectFileRepository
        _project_file_repo = ProjectFileRepository()
    return _project_file_repo


def get_memory_repo():
    """Get the memory repository instance."""
    global _memory_repo
    if _memory_repo is None:
        from app.db.repositories.memory_repo_sqlmodel import MemoryRepository
        _memory_repo = MemoryRepository()
    return _memory_repo


# Project operations
async def ensure_default_project():
    """Create the default project and attach orphan threads."""
    return await get_project_repo().ensure_default_project()


async def create_project(name: str, embedding_model: str, description: str = "", settings_json: dict = None):
    """Create a project."""
    return await get_project_repo().create(
        name=name,
        embedding_model=embedding_model,
        description=description,
        settings_json=settings_json,
    )


async def get_project(project_id: str):
    """Get a project by ID."""
    return await get_project_repo().get(project_id)


async def list_projects():
    """List projects."""
    return await get_project_repo().list_all()


async def update_project(project_id: str, name: str = None, description: str = None, settings_json: dict = None):
    """Update a project."""
    return await get_project_repo().update(
        project_id,
        name=name,
        description=description,
        settings_json=settings_json,
    )


async def assign_thread_to_project(thread_id: str, project_id: str):
    """Move a thread into a project."""
    return await get_project_repo().assign_thread(thread_id, project_id)


# Thread operations
async def create_thread(name: str, project_id: str):
    """Create a new thread."""
    return await get_thread_repo().create(name, project_id)


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


async def update_thread_project(thread_id: str, project_id: str):
    """Update a thread's project."""
    return await get_thread_repo().update_project(thread_id, project_id)


async def delete_thread(thread_id: str):
    """Delete a thread and all associated data."""
    return await get_thread_repo().delete(thread_id)


# File operations
async def create_or_get_file(
    file_hash: str,
    file_name: str,
    file_path: str = None,
    source_type: str = FileSourceType.PDF.value,
):
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


async def remove_thread_indexing_status(
    file_hash: str,
    embedding_model: str,
    thread_id: str,
    preserve_model_status: bool = False,
):
    """Remove a thread-scoped indexing entry and recompute the remaining summaries."""
    return await get_file_repo().remove_thread_indexing_status(
        file_hash, embedding_model, thread_id, preserve_model_status
    )


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


async def count_threads_with_file_for_model(file_hash: str, embedding_model: str, exclude_thread_id: str = None):
    """Count thread associations for a file restricted to a specific embedding model."""
    return await get_thread_file_repo().count_threads_with_file_for_model(file_hash, embedding_model, exclude_thread_id)


async def count_threads_with_file(file_hash: str):
    """Count how many threads currently reference a file."""
    return await get_thread_file_repo().count_threads_with_file(file_hash)


async def get_thread_file_association(thread_id: str, file_hash: str):
    """Get the thread-file association row for a single document."""
    return await get_thread_file_repo().get_association(thread_id, file_hash)


async def add_file_to_project(project_id: str, file_hash: str):
    return await get_project_file_repo().add(project_id, file_hash)


async def get_project_files(project_id: str):
    return await get_project_file_repo().get_files(project_id)


async def get_effective_thread_files(thread_id: str):
    return await get_project_file_repo().get_effective_thread_files(thread_id)


async def remove_file_from_project(project_id: str, file_hash: str):
    return await get_project_file_repo().remove(project_id, file_hash)


async def is_file_in_project(project_id: str, file_hash: str):
    return await get_project_file_repo().is_file_in_project(project_id, file_hash)


async def is_file_accessible_to_thread(thread_id: str, file_hash: str):
    return await get_project_file_repo().is_file_accessible_to_thread(thread_id, file_hash)


async def is_file_in_project_thread(project_id: str, file_hash: str):
    return await get_project_file_repo().is_file_in_project_thread(project_id, file_hash)


async def count_projects_with_file(file_hash: str):
    return await get_project_file_repo().count_projects_with_file(file_hash)


async def count_projects_with_file_for_model(file_hash: str, embedding_model: str):
    return await get_project_file_repo().count_projects_with_file_for_model(file_hash, embedding_model)


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
    reasoning_format: str = ReasoningFormat.NONE.value,
    web_sources: list = None,
):
    """Create a new message in a thread."""
    return await get_message_repo().create(
        thread_id, role, content, context_compact, reasoning,
        reasoning_available, reasoning_format, web_sources
    )


async def create_chat_turn(
    thread_id: str,
    question: str,
    answer: str = None,
    rewritten_question: str = None,
    status: str = ChatTurnStatus.COMPLETED.value,
    reasoning: str = "",
    reasoning_available: bool = False,
    reasoning_format: str = ReasoningFormat.NONE.value,
    web_sources: list = None,
    document_sources: list = None,
    used_chat_ids: list = None,
    clarification_options: list = None,
    error: dict = None,
    metadata: dict = None,
    agent_run_id: str = None,
    agent_run_turn_kind: str = None,
    agent_run_sequence: int = None,
    agent_trace_refs_json: dict = None,
):
    """Create one JSONB-backed chat turn."""
    return await get_message_repo().create_turn(
        thread_id=thread_id,
        question=question,
        answer=answer,
        rewritten_question=rewritten_question,
        status=status,
        reasoning=reasoning,
        reasoning_available=reasoning_available,
        reasoning_format=reasoning_format,
        web_sources=web_sources,
        document_sources=document_sources,
        used_chat_ids=used_chat_ids,
        clarification_options=clarification_options,
        error=error,
        metadata=metadata,
        agent_run_id=agent_run_id,
        agent_run_turn_kind=agent_run_turn_kind,
        agent_run_sequence=agent_run_sequence,
        agent_trace_refs_json=agent_trace_refs_json,
    )


async def get_message(message_id: str):
    """Get a message by ID."""
    return await get_message_repo().get(message_id)


async def get_chat_turn(turn_id: str):
    """Get a persisted chat turn by ID."""
    return await get_message_repo().get_turn(turn_id)


async def get_thread_turns(thread_id: str, limit: int = 100, offset: int = 0):
    """Get persisted chat turns for a thread."""
    return await get_message_repo().get_thread_turns(thread_id, limit, offset)


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
    """Remove a document entry from thread document metadata."""
    return await get_stats_repo().remove_document_from_stats(thread_id, file_hash)


async def upsert_document_in_stats(thread_id: str, file_hash: str, meta: dict):
    """Insert or replace a document entry in thread document metadata."""
    return await get_stats_repo().upsert_document_in_stats(thread_id, file_hash, meta)


async def increment_qa_stats(thread_id: str, qa_chars: int):
    """Increment QA aggregate counters after each answered turn."""
    return await get_stats_repo().increment_qa_stats(thread_id, qa_chars)


async def recompute_qa_stats(thread_id: str):
    """Recompute QA stats from chat turns."""
    return await get_stats_repo().recompute_qa_stats(thread_id)


async def get_thread_shape(thread_id: str):
    """Return a structured snapshot of the thread's content inventory."""
    return await get_stats_repo().get_thread_shape(thread_id)


# Memory operations
async def create_memory(**kwargs):
    """Create a canonical durable memory."""
    return await get_memory_repo().create_memory(**kwargs)


async def get_memory(memory_id: str):
    """Get a memory by ID."""
    return await get_memory_repo().get_memory(memory_id)


async def list_memories(**kwargs):
    """List memories."""
    return await get_memory_repo().list_memories(**kwargs)


async def delete_memory(memory_id: str):
    """Hard-delete a durable memory and its audit events."""
    return await get_memory_repo().delete_memory(memory_id)


async def delete_memories_for_scope(**kwargs):
    """Hard-delete all durable memories in a scope."""
    return await get_memory_repo().delete_memories_for_scope(**kwargs)


async def delete_expired_memories(**kwargs):
    """Hard-delete expired durable memories."""
    return await get_memory_repo().delete_expired_memories(**kwargs)


async def list_expired_memories(**kwargs):
    """List expired durable memories."""
    return await get_memory_repo().list_expired_memories(**kwargs)


async def mark_memory_indexing(memory_id: str):
    return await get_memory_repo().mark_memory_indexing(memory_id)


async def mark_memory_indexed(memory_id: str):
    return await get_memory_repo().mark_memory_indexed(memory_id)


async def mark_memory_index_failed(memory_id: str, error: str):
    return await get_memory_repo().mark_memory_index_failed(memory_id, error)


async def list_memories_for_index_retry(**kwargs):
    return await get_memory_repo().list_memories_for_index_retry(**kwargs)


async def create_memory_candidate(**kwargs):
    """Create a memory promotion candidate."""
    return await get_memory_repo().create_candidate(**kwargs)


async def list_memory_candidates(**kwargs):
    """List memory promotion candidates."""
    return await get_memory_repo().list_candidates(**kwargs)


async def resolve_memory_candidate(candidate_id: str, **kwargs):
    """Resolve a memory promotion candidate."""
    return await get_memory_repo().resolve_candidate(candidate_id, **kwargs)


async def delete_memory_candidate(candidate_id: str):
    """Hard-delete a memory promotion candidate."""
    return await get_memory_repo().delete_candidate(candidate_id)


async def delete_memory_candidates_for_thread(thread_id: str):
    """Hard-delete memory promotion candidates related to a thread."""
    return await get_memory_repo().delete_candidates_for_thread(thread_id)


__all__ = [
    # Models
    "ProcessStatus",
    "MessageRole",
    "FileSourceType",
    "ChatTurnStatus",
    "WorkflowVisibility",
    "AgentRunStatus",
    "MemoryCandidateStatus",
    "MemoryScopeType",
    "MemoryStatus",
    "MemoryType",
    "MemoryVisibility",
    "EmbeddingReadinessStatus",
    "FileStatusSection",
    "OperationResultStatus",
    "ReasoningFormat",
    "ThreadCloneMode",
    "Project",
    "Thread",
    "File",
    "ThreadFile",
    "ThreadDocumentAnnotation",
    "ChatTurn",
    "AgentWorkflow",
    "AgentRun",
    "Memory",
    "MemoryEvent",
    "MemoryCandidate",
    # Config
    "init_db",
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
    "get_thread_file_association",
    "get_thread_file_annotations",
    "upsert_thread_file_annotations",
    "delete_thread_file_annotations",
    # Message operations
    "create_chat_turn",
    "create_message",
    "get_message",
    "get_chat_turn",
    "get_thread_turns",
    "get_thread_messages",
    "get_recent_messages",
    "update_message_context_compact",
    "delete_message",
    "delete_message_pair",
    "get_message_count",
    # Agent workflow operations
    "get_agent_workflow_repo",
    # Project operations
    "ensure_default_project",
    "create_project",
    "get_project",
    "list_projects",
    "update_project",
    "assign_thread_to_project",
    "update_thread_project",
    # Stats operations
    "remove_document_from_stats",
    "upsert_document_in_stats",
    "increment_qa_stats",
    "recompute_qa_stats",
    "get_thread_shape",
    # Memory operations
    "create_memory",
    "get_memory",
    "list_memories",
    "delete_memory",
    "delete_memories_for_scope",
    "delete_expired_memories",
    "list_expired_memories",
    "create_memory_candidate",
    "list_memory_candidates",
    "resolve_memory_candidate",
    "delete_memory_candidate",
    "delete_memory_candidates_for_thread",
]
