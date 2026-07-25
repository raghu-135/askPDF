"""
models_sqlmodel.py - SQLModel table definitions for PostgreSQL.

This module contains all SQLModel table classes with proper:
- JSONB handling for flexible data
- Foreign key relationships with cascade behavior
- Indexes for query performance
"""

from datetime import datetime
import uuid
from typing import Dict, Any, List, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from app.db.enums import (
    AgentRunStatus,
    ChatTurnStatus,
    FileSourceType,
    MemoryCandidateStatus,
    MemoryScopeType,
    MemoryStatus,
    MemoryType,
    MemoryVisibility,
    MessageRole,
    ProcessStatus,
    WorkflowVisibility,
)
from app.time_utils import iso_utc_z, utc_now
from sqlmodel import SQLModel, Field, Relationship


# ============================================================================
# Association Table (Many-to-Many: Thread <-> File)
# ============================================================================

class ThreadFile(SQLModel, table=True):
    """Association between threads and files."""
    __tablename__ = "thread_files"
    
    thread_id: str = Field(
        sa_column=Column(String, ForeignKey("threads.id", ondelete="CASCADE"), primary_key=True)
    )
    file_hash: str = Field(
        sa_column=Column(String, ForeignKey("files.file_hash", ondelete="CASCADE"), primary_key=True)
    )
    added_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    annotations: List[Dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSONB, default=list)
    )
    annotations_updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )


# ============================================================================
# Main Tables
# ============================================================================

class Project(SQLModel, table=True):
    """Project container for shared thread context and memory."""
    __tablename__ = "projects"

    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    description: str = ""
    embedding_model: str = Field(index=True)
    settings_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), onupdate=func.now())
    )

    threads: List["Thread"] = Relationship(
        back_populates="project",
        sa_relationship_kwargs={"passive_deletes": True}
    )

    __table_args__ = (
        UniqueConstraint("id", "embedding_model", name="uq_projects_id_embedding_model"),
        CheckConstraint("length(btrim(embedding_model)) > 0", name="ck_projects_embedding_model_nonempty"),
        Index("idx_project_created_at", "created_at"),
    )


class Thread(SQLModel, table=True):
    """Chat thread entity."""
    __tablename__ = "threads"
    
    id: str = Field(primary_key=True)
    project_id: str = Field(
        sa_column=Column(String, index=True, nullable=False)
    )
    name: str = Field(index=True)
    embedding_model: str = Field(index=True)
    is_legacy: bool = Field(default=False, index=True)
    settings: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    thread_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    total_qa_pairs: int = Field(default=0)
    total_qa_chars: int = Field(default=0)
    avg_qa_chars: float = Field(default=0.0)
    last_qa_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )
    documents_meta: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    stats_last_updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), onupdate=func.now())
    )
    
    # Relationships
    project: Optional["Project"] = Relationship(back_populates="threads")
    chat_turns: List["ChatTurn"] = Relationship(
        back_populates="thread",
        sa_relationship_kwargs={"passive_deletes": True, "cascade": "all, delete-orphan"}
    )
    files: List["File"] = Relationship(
        back_populates="threads",
        link_model=ThreadFile,
        sa_relationship_kwargs={"passive_deletes": True}
    )
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="fk_threads_project_id_projects",
            ondelete="RESTRICT",
        ),
        Index("idx_thread_created_at", "created_at"),
    )


class File(SQLModel, table=True):
    """File entity (PDF or web source)."""
    __tablename__ = "files"
    
    file_hash: str = Field(primary_key=True)
    file_name: str = Field(index=True)  # Note: matches existing model, not 'filename'
    file_path: Optional[str] = None
    source_type: str = Field(default=FileSourceType.PDF.value, index=True)
    file_status: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    parsed_sentences_json: Optional[str] = None
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    
    # Relationships
    threads: List["Thread"] = Relationship(
        back_populates="files",
        link_model=ThreadFile,
        sa_relationship_kwargs={"passive_deletes": True}
    )
    # Helper method for safe JSONB mutation
    def set_file_status_key(self, key: str, value: Any) -> None:
        """Set a key in file_status, ensuring change tracking."""
        if self.file_status is None:
            self.file_status = {}
        # Create new dict to ensure SQLAlchemy detects change
        new_status = dict(self.file_status)
        new_status[key] = value
        new_status["updated_at"] = iso_utc_z()
        self.file_status = new_status
    
    __table_args__ = (
        Index("idx_file_source_type", "source_type"),
    )


class ChatTurn(SQLModel, table=True):
    """One persisted chat interaction with flexible JSONB payload."""
    __tablename__ = "chat_turns"

    id: str = Field(primary_key=True)
    thread_id: str = Field(
        sa_column=Column(String, ForeignKey("threads.id", ondelete="CASCADE"), index=True)
    )
    agent_run_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String, ForeignKey("agent_runs.id", ondelete="SET NULL"), index=True)
    )
    agent_run_turn_kind: Optional[str] = Field(default=None)
    agent_run_sequence: Optional[int] = Field(default=None)
    agent_trace_refs_json: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB)
    )
    status: str = Field(default=ChatTurnStatus.COMPLETED.value, index=True)
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), onupdate=func.now())
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )

    # Relationships
    thread: Optional["Thread"] = Relationship(back_populates="chat_turns")

    __table_args__ = (
        Index("idx_chat_turn_thread_created", "thread_id", "created_at"),
        Index("idx_chat_turn_agent_run_sequence", "agent_run_id", "agent_run_sequence"),
    )


class AgentWorkflow(SQLModel, table=True):
    """Agent workflow and its current executable spec."""
    __tablename__ = "agent_workflows"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str = ""
    visibility: str = Field(default=WorkflowVisibility.BUILTIN.value, index=True)
    is_builtin: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default="false"),
    )
    schema_version: int = Field(default=2)
    spec_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    validation_result_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    metadata_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), onupdate=func.now())
    )

    __table_args__ = (
        Index("idx_agent_workflow_builtin", "is_builtin"),
    )

    @property
    def version(self) -> int:
        metadata = self.metadata_json if isinstance(self.metadata_json, dict) else {}
        try:
            return int(metadata.get("version") or self.schema_version or 1)
        except (TypeError, ValueError):
            return 1


class AgentRun(SQLModel, table=True):
    """Execution record for one frozen agent workflow run."""
    __tablename__ = "agent_runs"

    id: str = Field(primary_key=True)
    thread_id: str = Field(
        sa_column=Column(String, ForeignKey("threads.id", ondelete="CASCADE"), index=True)
    )
    user_id: Optional[str] = Field(default=None, index=True)
    workflow_id: str = Field(
        sa_column=Column(String, ForeignKey("agent_workflows.id", ondelete="RESTRICT"), index=True)
    )
    run_metadata_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    resolved_spec_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    status: str = Field(default=AgentRunStatus.RUNNING.value, index=True)
    checkpoint_thread_id: Optional[str] = None
    pending_interrupt_json: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB)
    )
    started_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )
    error_json: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB)
    )
    metrics_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    debug_trace_json: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB)
    )

    __table_args__ = (
        Index("idx_agent_run_thread_started", "thread_id", "started_at"),
    )

    @property
    def workflow_version_id(self) -> Optional[str]:
        metadata = self.run_metadata_json if isinstance(self.run_metadata_json, dict) else {}
        value = metadata.get("workflow_version_id")
        return str(value) if value is not None else None

    @workflow_version_id.setter
    def workflow_version_id(self, value: Optional[str]) -> None:
        metadata = dict(self.run_metadata_json or {})
        if value is None:
            metadata.pop("workflow_version_id", None)
        else:
            metadata["workflow_version_id"] = value
        self.run_metadata_json = metadata

    @property
    def workflow_version(self) -> Optional[int]:
        metadata = self.run_metadata_json if isinstance(self.run_metadata_json, dict) else {}
        value = metadata.get("workflow_version")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @workflow_version.setter
    def workflow_version(self, value: Optional[int]) -> None:
        metadata = dict(self.run_metadata_json or {})
        if value is None:
            metadata.pop("workflow_version", None)
        else:
            metadata["workflow_version"] = int(value)
        self.run_metadata_json = metadata


class Memory(SQLModel, table=True):
    """Canonical durable memory owned by the app, independent of agent framework."""
    __tablename__ = "memories"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    scope_type: str = Field(default=MemoryScopeType.THREAD.value, index=True)
    scope_id: str = Field(index=True)
    memory_type: str = Field(default=MemoryType.SEMANTIC.value, index=True)
    content: str
    summary: str = ""
    embedding_model: str = Field(index=True)
    content_hash: str = Field(index=True)
    index_status: str = Field(default="pending", index=True)
    index_attempts: int = Field(default=0)
    indexed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )
    index_error: Optional[str] = None
    source_refs_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    confidence: float = Field(default=1.0, sa_column=Column(Float, nullable=False, server_default="1"))
    status: str = Field(default=MemoryStatus.ACTIVE.value, index=True)
    visibility: str = Field(default=MemoryVisibility.PRIVATE.value, index=True)
    created_by: Optional[str] = Field(default=None, index=True)
    expires_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )
    fork_origin_json: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB)
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), onupdate=func.now())
    )

    events: List["MemoryEvent"] = Relationship(
        back_populates="memory",
        sa_relationship_kwargs={"passive_deletes": True, "cascade": "all, delete-orphan"}
    )

    __table_args__ = (
        CheckConstraint("scope_type in ('user', 'project', 'thread')", name="ck_memories_scope_type"),
        CheckConstraint("memory_type in ('semantic', 'episodic', 'procedural')", name="ck_memories_memory_type"),
        CheckConstraint("status = 'active'", name="ck_memories_status"),
        CheckConstraint("visibility in ('private', 'project', 'internal')", name="ck_memories_visibility"),
        CheckConstraint("confidence >= 0 and confidence <= 1", name="ck_memories_confidence"),
        CheckConstraint("length(btrim(scope_id)) > 0", name="ck_memories_scope_id_nonempty"),
        CheckConstraint("length(btrim(content)) > 0", name="ck_memories_content_nonempty"),
        CheckConstraint("length(btrim(embedding_model)) > 0", name="ck_memories_embedding_model_nonempty"),
        CheckConstraint("length(btrim(content_hash)) > 0", name="ck_memories_content_hash_nonempty"),
        CheckConstraint("index_status in ('pending', 'indexing', 'indexed', 'failed')", name="ck_memories_index_status"),
        CheckConstraint("index_attempts >= 0", name="ck_memories_index_attempts"),
        Index("idx_memory_scope_status", "scope_type", "scope_id", "status"),
        Index("idx_memory_index_retry", "index_status", "updated_at"),
        Index("idx_memory_created_at", "created_at"),
    )


class MemoryEvent(SQLModel, table=True):
    """Audit event for durable memory changes."""
    __tablename__ = "memory_events"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    memory_id: str = Field(
        sa_column=Column(String, ForeignKey("memories.id", ondelete="CASCADE"), index=True)
    )
    event_type: str = Field(index=True)
    actor_id: Optional[str] = Field(default=None, index=True)
    payload_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )

    memory: Optional["Memory"] = Relationship(back_populates="events")


class MemoryCandidate(SQLModel, table=True):
    """Memory promotion candidate generated from chat/run context."""
    __tablename__ = "memory_candidates"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    source_thread_id: Optional[str] = Field(default=None, index=True)
    source_project_id: Optional[str] = Field(default=None, index=True)
    source_agent_run_id: Optional[str] = Field(default=None, index=True)
    source_turn_id: Optional[str] = Field(default=None, index=True)
    proposed_scope_type: str = Field(default=MemoryScopeType.THREAD.value, index=True)
    proposed_scope_id: str = Field(index=True)
    memory_type: str = Field(default=MemoryType.SEMANTIC.value, index=True)
    content: str
    confidence: float = Field(default=0.0, sa_column=Column(Float, nullable=False, server_default="0"))
    reason: str = ""
    status: str = Field(default=MemoryCandidateStatus.PENDING.value, index=True)
    promoted_memory_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String, ForeignKey("memories.id", ondelete="SET NULL"), index=True)
    )
    resolved_by: Optional[str] = Field(default=None, index=True)
    resolved_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )
    created_by: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), onupdate=func.now())
    )

    __table_args__ = (
        CheckConstraint("proposed_scope_type in ('user', 'project', 'thread')", name="ck_memory_candidates_scope_type"),
        CheckConstraint("memory_type in ('semantic', 'episodic', 'procedural')", name="ck_memory_candidates_memory_type"),
        CheckConstraint("status in ('pending', 'approved', 'rejected', 'auto_approved')", name="ck_memory_candidates_status"),
        CheckConstraint("confidence >= 0 and confidence <= 1", name="ck_memory_candidates_confidence"),
        CheckConstraint("length(btrim(proposed_scope_id)) > 0", name="ck_memory_candidates_scope_id_nonempty"),
        CheckConstraint("length(btrim(content)) > 0", name="ck_memory_candidates_content_nonempty"),
        Index("idx_memory_candidate_scope_status", "proposed_scope_type", "proposed_scope_id", "status"),
        Index("idx_memory_candidate_source_thread", "source_thread_id", "created_at"),
    )
