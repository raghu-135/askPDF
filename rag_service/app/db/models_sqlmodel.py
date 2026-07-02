"""
models_sqlmodel.py - SQLModel table definitions for PostgreSQL.

This module contains all SQLModel table classes with proper:
- JSONB handling for flexible data
- Foreign key relationships with cascade behavior
- Indexes for query performance
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, func, Index, String, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from app.time_utils import iso_utc_z, utc_now
from sqlmodel import SQLModel, Field, Relationship


class ProcessStatus(str, Enum):
    """Status values for processing operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"

    @classmethod
    def is_completed(cls, status: str) -> bool:
        """Check if status is completed."""
        return status == cls.COMPLETED.value

    @classmethod
    def is_failed(cls, status: str) -> bool:
        """Check if status is failed."""
        return status == cls.FAILED.value

    @classmethod
    def is_running(cls, status: str) -> bool:
        """Check if status is running."""
        return status == cls.RUNNING.value


class MessageRole(str, Enum):
    """Role values for chat messages."""
    USER = "user"
    ASSISTANT = "assistant"


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

class Thread(SQLModel, table=True):
    """Chat thread entity."""
    __tablename__ = "threads"
    
    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    embed_model: str = Field(index=True)
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
        Index("idx_thread_created_at", "created_at"),
    )


class File(SQLModel, table=True):
    """File entity (PDF or web source)."""
    __tablename__ = "files"
    
    file_hash: str = Field(primary_key=True)
    file_name: str = Field(index=True)  # Note: matches existing model, not 'filename'
    file_path: Optional[str] = None
    source_type: str = Field(default="pdf", index=True)
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
    status: str = Field(default="completed", index=True)
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
    )


class AgentPatternTemplate(SQLModel, table=True):
    """Versioned agent pattern template family."""
    __tablename__ = "agent_pattern_templates"

    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    description: str = ""
    visibility: str = Field(default="builtin", index=True)
    owner_id: Optional[str] = Field(default=None, index=True)
    current_version_id: Optional[str] = Field(default=None, index=True)
    is_builtin: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default="false"),
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
        Index("idx_agent_pattern_template_builtin", "is_builtin"),
    )


class AgentPatternTemplateVersion(SQLModel, table=True):
    """Immutable spec for a single agent pattern version."""
    __tablename__ = "agent_pattern_template_versions"

    id: str = Field(primary_key=True)
    template_id: str = Field(
        sa_column=Column(String, ForeignKey("agent_pattern_templates.id", ondelete="CASCADE"), index=True)
    )
    version: int = Field(index=True)
    schema_version: int = Field(default=1)
    spec_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    validation_result_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    changelog: Optional[str] = None
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )

    __table_args__ = (
        Index("idx_agent_pattern_template_version_unique", "template_id", "version", unique=True),
    )


class AgentRun(SQLModel, table=True):
    """Execution record for one frozen agent pattern run."""
    __tablename__ = "agent_runs"

    id: str = Field(primary_key=True)
    thread_id: str = Field(
        sa_column=Column(String, ForeignKey("threads.id", ondelete="CASCADE"), index=True)
    )
    user_id: Optional[str] = Field(default=None, index=True)
    template_id: str = Field(
        sa_column=Column(String, ForeignKey("agent_pattern_templates.id", ondelete="RESTRICT"), index=True)
    )
    template_version_id: str = Field(
        sa_column=Column(String, ForeignKey("agent_pattern_template_versions.id", ondelete="RESTRICT"), index=True)
    )
    resolved_spec_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, default=dict)
    )
    status: str = Field(default="running", index=True)
    checkpoint_thread_id: Optional[str] = None
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

    __table_args__ = (
        Index("idx_agent_run_thread_started", "thread_id", "started_at"),
    )
