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
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from app.db.enums import (
    AgentRunStatus,
    ChatTurnStatus,
    FileSourceType,
    MemoryScopeType,
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


class ProjectFile(SQLModel, table=True):
    """Association between projects and shared knowledge files."""
    __tablename__ = "project_files"

    project_id: str = Field(
        sa_column=Column(String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    )
    file_hash: str = Field(
        sa_column=Column(String, ForeignKey("files.file_hash", ondelete="CASCADE"), primary_key=True)
    )
    added_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )

    __table_args__ = (
        Index("idx_project_files_file_hash", "file_hash"),
        Index("idx_project_files_added_at", "added_at"),
    )


class ThreadDocumentAnnotation(SQLModel, table=True):
    """Thread-owned annotation overlay for any document accessible to the thread."""
    __tablename__ = "thread_document_annotations"

    thread_id: str = Field(
        sa_column=Column(String, ForeignKey("threads.id", ondelete="CASCADE"), primary_key=True)
    )
    file_hash: str = Field(
        sa_column=Column(String, ForeignKey("files.file_hash", ondelete="CASCADE"), primary_key=True)
    )
    annotations: List[Dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSONB, default=list)
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )

    __table_args__ = (
        Index("idx_thread_document_annotations_file_hash", "file_hash"),
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
    last_activity_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
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
        Index("idx_project_last_activity_at", "last_activity_at"),
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
            ["project_id", "embedding_model"],
            ["projects.id", "projects.embedding_model"],
            name="fk_threads_project_embedding_model",
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
    framework: str = Field(default="langgraph", index=True)
    builder_id: str = Field(default="langgraph_graph", index=True)
    category: Optional[str] = Field(default=None, index=True)
    schema_version: int = Field(default=1)
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
    framework: str = Field(default="langgraph", index=True)
    builder_id: str = Field(default="langgraph_graph", index=True)
    definition_category: Optional[str] = Field(default=None, index=True)
    runtime_binding_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default=dict),
    )
    runtime_binding_status: str = Field(default="active", index=True)
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
    task_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String, ForeignKey("agent_tasks.id", ondelete="CASCADE"), index=True),
    )
    parent_run_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String, ForeignKey("agent_runs.id", ondelete="RESTRICT"), index=True),
    )
    task_attempt: int = Field(default=1)
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
        CheckConstraint("task_attempt >= 1", name="ck_agent_runs_task_attempt"),
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


class AgentRunEvent(SQLModel, table=True):
    """Canonical, framework-neutral observability event for an agent run."""
    __tablename__ = "agent_run_events"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    agent_run_id: str = Field(
        sa_column=Column(String, ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    )
    event_id: str = Field(index=True)
    sequence: int = Field(default=0)
    attempt: int = Field(default=1)
    kind: str = Field(index=True)
    occurred_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    payload_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default=dict),
    )
    trace_id: Optional[str] = Field(default=None, index=True)
    terminal: bool = Field(default=False, sa_column=Column(Boolean, nullable=False, server_default="false"))
    source_metadata_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default=dict),
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    __table_args__ = (
        UniqueConstraint("agent_run_id", "event_id", name="uq_agent_run_events_run_event"),
        Index("idx_agent_run_events_run_sequence", "agent_run_id", "attempt", "sequence"),
        CheckConstraint("attempt >= 1", name="ck_agent_run_events_attempt"),
        CheckConstraint("sequence >= 0", name="ck_agent_run_events_sequence"),
    )


class AgentTask(SQLModel, table=True):
    """Durable user-facing task that owns one or more agent run attempts."""
    __tablename__ = "agent_tasks"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    thread_id: str = Field(sa_column=Column(String, ForeignKey("threads.id", ondelete="CASCADE"), index=True))
    project_id: Optional[str] = Field(default=None, sa_column=Column(String, ForeignKey("projects.id", ondelete="SET NULL"), index=True))
    user_id: Optional[str] = Field(default=None, index=True)
    workflow_id: str = Field(sa_column=Column(String, ForeignKey("agent_workflows.id", ondelete="RESTRICT"), index=True))
    objective: str = Field(sa_column=Column(Text, nullable=False))
    objective_hash: str = Field(index=True)
    create_idempotency_key: str
    config_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, default=dict))
    status: str = Field(default="created", index=True)
    primary_run_id: Optional[str] = Field(default=None, index=True)
    active_run_id: Optional[str] = Field(default=None, index=True)
    latest_run_attempt: int = 0
    version: int = 1
    lease_owner: Optional[str] = Field(default=None, index=True)
    lease_expires_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), index=True))
    heartbeat_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    completed_todos: int = 0
    total_todos: int = 0
    progress: int = 0
    current_phase: str = "created"
    terminal_reason: Optional[str] = None
    budgets_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, default=dict))
    created_at: datetime = Field(default_factory=utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))
    queued_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    started_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    paused_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    completed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    expires_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), index=True))
    deletion_requested_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), index=True))
    deletion_completed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime = Field(default_factory=utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))

    __table_args__ = (
        CheckConstraint("length(btrim(objective)) > 0", name="ck_agent_tasks_objective_nonempty"),
        CheckConstraint("length(btrim(objective_hash)) > 0", name="ck_agent_tasks_objective_hash_nonempty"),
        CheckConstraint("length(btrim(create_idempotency_key)) > 0", name="ck_agent_tasks_idempotency_nonempty"),
        CheckConstraint("status in ('created','queued','running','pausing','paused','awaiting_approval','cancelling','cancelled','completed','failed','expired')", name="ck_agent_tasks_status"),
        CheckConstraint("version >= 1 and latest_run_attempt >= 0", name="ck_agent_tasks_versions"),
        CheckConstraint("progress between 0 and 100 and completed_todos >= 0 and total_todos >= 0", name="ck_agent_tasks_progress"),
        Index(
            "uq_agent_tasks_owner_idempotency_nullsafe",
            "thread_id",
            text("coalesce(user_id, '')"),
            "create_idempotency_key",
            unique=True,
        ),
        Index("idx_agent_tasks_claim", "status", "lease_expires_at", "queued_at"),
        Index("idx_agent_tasks_thread_created", "thread_id", "created_at"),
    )


class AgentTaskPlanRevision(SQLModel, table=True):
    __tablename__ = "agent_task_plan_revisions"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    task_id: str = Field(sa_column=Column(String, ForeignKey("agent_tasks.id", ondelete="CASCADE"), index=True))
    agent_run_id: str = Field(sa_column=Column(String, ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True))
    revision: int
    planner_visit: int = 1
    reason: str
    objective: str = Field(sa_column=Column(Text, nullable=False))
    completion_criteria_json: List[Any] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False, default=list))
    ordered_todo_ids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False, default=list))
    plan_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, default=dict))
    provenance_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, default=dict))
    content_hash: str
    created_at: datetime = Field(default_factory=utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))

    __table_args__ = (
        CheckConstraint("revision >= 1 and planner_visit >= 1", name="ck_agent_task_plan_revision_values"),
        UniqueConstraint("task_id", "revision", name="uq_agent_task_plan_revision"),
    )


class AgentTaskTodo(SQLModel, table=True):
    __tablename__ = "agent_task_todos"

    id: str = Field(primary_key=True)
    task_id: str = Field(
        sa_column=Column(
            String,
            ForeignKey("agent_tasks.id", ondelete="CASCADE"),
            primary_key=True,
        )
    )
    title: str
    description: str = Field(sa_column=Column(Text, nullable=False))
    completion_criteria: str = Field(sa_column=Column(Text, nullable=False))
    status: str = Field(default="pending", index=True)
    priority: int = 50
    required: bool = True
    dependency_ids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False, default=list))
    profile_id: str
    attempt: int = 0
    max_attempts: int = 2
    progress: int = 0
    version: int = 1
    result_summary: Optional[str] = Field(default=None, sa_column=Column(Text))
    terminal_reason: Optional[str] = None
    current_subagent_run_id: Optional[str] = Field(default=None, index=True)
    evidence_ids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False, default=list))
    artifact_ids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False, default=list))
    created_revision: int = 1
    updated_revision: int = 1
    created_at: datetime = Field(default_factory=utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))
    updated_at: datetime = Field(default_factory=utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))

    __table_args__ = (
        CheckConstraint("status in ('pending','ready','running','blocked','completed','failed','skipped','cancelled')", name="ck_agent_task_todos_status"),
        CheckConstraint("priority between 0 and 100 and progress between 0 and 100", name="ck_agent_task_todos_ranges"),
        CheckConstraint("attempt >= 0 and max_attempts between 1 and 10", name="ck_agent_task_todos_attempts"),
        CheckConstraint("version >= 1 and created_revision >= 1 and updated_revision >= created_revision", name="ck_agent_task_todos_versions"),
        Index("idx_agent_task_todos_schedule", "task_id", "status", "priority"),
    )


class AgentTaskSubagentRun(SQLModel, table=True):
    __tablename__ = "agent_task_subagent_runs"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    task_id: str = Field(sa_column=Column(String, ForeignKey("agent_tasks.id", ondelete="CASCADE"), index=True))
    agent_run_id: str = Field(sa_column=Column(String, ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True))
    todo_id: str = Field(index=True)
    execution_key: str = Field(unique=True, index=True)
    profile_id: str
    plan_revision: int
    attempt: int
    status: str = Field(default="queued", index=True)
    usage_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, default=dict))
    tool_policy_hash: str
    timeout_ms: int
    output_artifact_ids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False, default=list))
    error_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    started_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    completed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    created_at: datetime = Field(default_factory=utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))

    __table_args__ = (
        CheckConstraint("status in ('queued','running','completed','failed','timed_out','cancelled')", name="ck_agent_task_subagent_status"),
        CheckConstraint("attempt >= 1 and plan_revision >= 1 and timeout_ms > 0", name="ck_agent_task_subagent_values"),
        ForeignKeyConstraint(["task_id", "todo_id"], ["agent_task_todos.task_id", "agent_task_todos.id"], ondelete="CASCADE"),
    )


class AgentTaskArtifact(SQLModel, table=True):
    __tablename__ = "agent_task_artifacts"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    task_id: str = Field(sa_column=Column(String, ForeignKey("agent_tasks.id", ondelete="CASCADE"), index=True))
    agent_run_id: str = Field(sa_column=Column(String, ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True))
    todo_id: Optional[str] = Field(default=None, index=True)
    subagent_run_id: Optional[str] = Field(default=None, sa_column=Column(String, ForeignKey("agent_task_subagent_runs.id", ondelete="SET NULL"), index=True))
    ownership_key: str = Field(index=True)
    kind: str = Field(index=True)
    object_key: str = Field(unique=True)
    media_type: str
    byte_size: int
    sha256: str
    version: int = 1
    provenance_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, default=dict))
    source_refs_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, default=dict))
    summary_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, default=dict))
    supersedes_id: Optional[str] = Field(default=None, sa_column=Column(String, ForeignKey("agent_task_artifacts.id", ondelete="SET NULL")))
    validity: str = Field(default="valid", index=True)
    sensitivity: str = Field(default="private")
    retention_until: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), index=True))
    deleted_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    created_at: datetime = Field(default_factory=utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))

    __table_args__ = (
        CheckConstraint("kind in ('tool_output','intermediate_report','context_summary','final_report')", name="ck_agent_task_artifacts_kind"),
        CheckConstraint("validity in ('valid','invalid','deleted') and sensitivity in ('private','sensitive')", name="ck_agent_task_artifacts_state"),
        CheckConstraint("byte_size >= 0 and version >= 1", name="ck_agent_task_artifacts_values"),
        CheckConstraint("length(btrim(ownership_key)) > 0", name="ck_agent_task_artifacts_ownership_key"),
        UniqueConstraint("agent_run_id", "ownership_key", "sha256", "kind", name="uq_agent_task_artifact_content"),
        Index(
            "uq_agent_task_artifacts_final_report",
            "agent_run_id",
            unique=True,
            postgresql_where=text("kind = 'final_report' and validity = 'valid' and deleted_at is null"),
        ),
    )


class AgentTaskEvent(SQLModel, table=True):
    __tablename__ = "agent_task_events"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    task_id: str = Field(sa_column=Column(String, ForeignKey("agent_tasks.id", ondelete="CASCADE"), index=True))
    sequence: int
    event_id: Optional[str] = Field(default=None, index=True)
    event_type: str = Field(index=True)
    actor_type: str
    actor_id: Optional[str] = None
    agent_run_id: Optional[str] = Field(default=None, index=True)
    todo_id: Optional[str] = Field(default=None, index=True)
    subagent_run_id: Optional[str] = Field(default=None, index=True)
    artifact_id: Optional[str] = Field(default=None, index=True)
    payload_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, default=dict))
    policy_hash: Optional[str] = None
    config_hash: Optional[str] = None
    occurred_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    terminal: bool = Field(default=False, sa_column=Column(Boolean, nullable=False, server_default="false"))
    source_metadata_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default=dict),
    )
    created_at: datetime = Field(default_factory=utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))

    __table_args__ = (
        CheckConstraint("sequence >= 1", name="ck_agent_task_events_sequence"),
        UniqueConstraint("task_id", "sequence", name="uq_agent_task_event_sequence"),
        Index("idx_agent_task_events_stream", "task_id", "sequence"),
        Index("idx_agent_task_events_run_stream", "task_id", "agent_run_id", "sequence"),
    )


class AgentTaskCommand(SQLModel, table=True):
    __tablename__ = "agent_task_commands"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    task_id: str = Field(sa_column=Column(String, ForeignKey("agent_tasks.id", ondelete="CASCADE"), index=True))
    action: str
    idempotency_key: str
    expected_version: int
    actor_id: Optional[str] = None
    status: str = Field(default="accepted")
    result_version: Optional[int] = None
    result_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, default=dict))
    created_at: datetime = Field(default_factory=utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))
    completed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))

    __table_args__ = (
        CheckConstraint("action in ('start','pause','resume','cancel','retry','expire','delete','steer')", name="ck_agent_task_commands_action"),
        CheckConstraint("status in ('accepted','completed','rejected') and expected_version >= 1", name="ck_agent_task_commands_state"),
        UniqueConstraint("task_id", "action", "idempotency_key", name="uq_agent_task_command_idempotency"),
    )


class AgentRuntimeOperation(SQLModel, table=True):
    """Product-owned idempotency record for a runtime control operation."""
    __tablename__ = "agent_runtime_operations"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    run_id: str = Field(sa_column=Column(String, ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True))
    operation: str = Field(index=True)
    idempotency_key: str
    request_fingerprint: str
    status: str = Field(default="in_progress", index=True)
    result_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False, default=dict))
    error_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))
    completed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))

    __table_args__ = (
        CheckConstraint("status in ('in_progress', 'completed', 'failed')", name="ck_agent_runtime_operations_status"),
        CheckConstraint("length(btrim(idempotency_key)) > 0", name="ck_agent_runtime_operations_key_nonempty"),
        UniqueConstraint("run_id", "operation", "idempotency_key", name="uq_agent_runtime_operation_idempotency"),
        Index("idx_agent_runtime_operations_run_operation", "run_id", "operation"),
    )


class Memory(SQLModel, table=True):
    """Canonical durable memory owned by the app, independent of agent framework."""
    __tablename__ = "memories"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    scope_type: str = Field(default=MemoryScopeType.THREAD.value, index=True)
    scope_id: str = Field(index=True)
    content: str
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
        sa_column=Column(JSONB, nullable=False, default=dict)
    )
    attributes_json: Dict[str, Any] = Field(
        default_factory=lambda: {
            "kind": "fact",
            "applicability": ["task_specific"],
            "durability": "stable",
        },
        sa_column=Column(JSONB, nullable=False, default=dict),
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), onupdate=func.now())
    )
    semantic_updated_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )

    events: List["MemoryEvent"] = Relationship(
        back_populates="memory",
        sa_relationship_kwargs={"passive_deletes": True, "cascade": "all, delete-orphan"}
    )

    __table_args__ = (
        CheckConstraint("scope_type in ('user', 'project', 'thread')", name="ck_memories_scope_type"),
        CheckConstraint("length(btrim(scope_id)) > 0", name="ck_memories_scope_id_nonempty"),
        CheckConstraint("length(btrim(content)) > 0", name="ck_memories_content_nonempty"),
        CheckConstraint("length(btrim(embedding_model)) > 0", name="ck_memories_embedding_model_nonempty"),
        CheckConstraint("length(btrim(content_hash)) > 0", name="ck_memories_content_hash_nonempty"),
        CheckConstraint("index_status in ('pending', 'indexing', 'indexed', 'failed')", name="ck_memories_index_status"),
        CheckConstraint("index_attempts >= 0", name="ck_memories_index_attempts"),
        CheckConstraint(
            "jsonb_typeof(attributes_json) = 'object' "
            "and attributes_json ->> 'kind' in ('preference', 'profile', 'instruction', 'constraint', 'decision', 'fact') "
            "and jsonb_typeof(attributes_json -> 'applicability') = 'array' "
            "and jsonb_array_length(attributes_json -> 'applicability') > 0 "
            "and (attributes_json -> 'applicability') <@ '[\"all_answers\",\"writing\",\"code\",\"research\",\"project\",\"task_specific\"]'::jsonb "
            "and attributes_json ->> 'durability' in ('stable', 'time_sensitive')",
            name="ck_memories_attributes_json",
        ),
        Index("idx_memory_scope", "scope_type", "scope_id"),
        Index("idx_memory_index_retry", "index_status", "updated_at"),
        Index("idx_memory_created_at", "created_at"),
        Index("idx_memory_semantic_updated_at", "semantic_updated_at"),
    )


class MemoryOverride(SQLModel, table=True):
    """An explicit, user-confirmed replacement relationship between memories."""
    __tablename__ = "memory_overrides"

    overriding_memory_id: str = Field(
        sa_column=Column(String, ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True)
    )
    overridden_memory_id: str = Field(
        sa_column=Column(String, ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True)
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )

    __table_args__ = (
        CheckConstraint(
            "overriding_memory_id <> overridden_memory_id",
            name="ck_memory_overrides_not_self",
        ),
        Index("ix_memory_overrides_target", "overridden_memory_id"),
    )


class GlobalMemoryRepresentation(SQLModel, table=True):
    """Secondary model-specific vector state for canonical Global memory."""
    __tablename__ = "global_memory_representations"

    memory_id: str = Field(
        sa_column=Column(String, ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True)
    )
    embedding_model: str = Field(primary_key=True)
    content_hash: str
    index_status: str = Field(default="pending", index=True)
    index_attempts: int = Field(default=0)
    indexed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    index_error: Optional[str] = None
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )

    __table_args__ = (
        CheckConstraint("length(btrim(embedding_model)) > 0", name="ck_global_memory_rep_model_nonempty"),
        CheckConstraint("length(btrim(content_hash)) > 0", name="ck_global_memory_rep_hash_nonempty"),
        CheckConstraint(
            "index_status in ('pending', 'indexing', 'indexed', 'failed')",
            name="ck_global_memory_rep_status",
        ),
        CheckConstraint("index_attempts >= 0", name="ck_global_memory_rep_attempts"),
        Index("idx_global_memory_rep_retry", "embedding_model", "index_status", "updated_at"),
    )


class MemoryScopeActivity(SQLModel, table=True):
    """Monotonic mutation version for one canonical memory scope."""
    __tablename__ = "memory_scope_activity"

    scope_type: str = Field(primary_key=True)
    scope_id: str = Field(primary_key=True)
    version: int = Field(default=1)
    changed_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )

    __table_args__ = (
        CheckConstraint("scope_type in ('user', 'project', 'thread')", name="ck_memory_scope_activity_type"),
        CheckConstraint("version >= 1", name="ck_memory_scope_activity_version"),
    )


class MemoryReviewState(SQLModel, table=True):
    """Last completed consistency-review versions for a memory review context."""
    __tablename__ = "memory_review_states"

    context_type: str = Field(primary_key=True)
    context_id: str = Field(primary_key=True)
    reviewed_scope_versions_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default=dict),
    )
    last_reviewed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )

    __table_args__ = (
        CheckConstraint("context_type in ('user', 'project', 'thread')", name="ck_memory_review_context_type"),
    )


class MemoryManagerIdempotency(SQLModel, table=True):
    """Durable claim and result record for confirmed memory-manager plans."""
    __tablename__ = "memory_manager_idempotency"

    idempotency_key: str = Field(primary_key=True)
    plan_hash: str = Field(index=True)
    status: str = Field(default="in_progress", index=True)
    result_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, default=dict)
    )
    actor_id: str = Field(default="ui")
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    )

    __table_args__ = (
        CheckConstraint("length(btrim(idempotency_key)) > 0", name="ck_memory_manager_idempotency_key_nonempty"),
        CheckConstraint("length(btrim(plan_hash)) > 0", name="ck_memory_manager_idempotency_plan_hash_nonempty"),
        CheckConstraint("status in ('in_progress', 'committed')", name="ck_memory_manager_idempotency_status"),
    )


class EmbeddingJob(SQLModel, table=True):
    """Durable, deduplicated work item for model-specific vector materialization."""
    __tablename__ = "embedding_jobs"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    resource_type: str = Field(index=True)
    resource_id: str = Field(index=True)
    scope_id: str = Field(index=True)
    embedding_model: str = Field(index=True)
    source_version: str
    status: str = Field(default="pending", index=True)
    attempts: int = Field(default=0)
    error: Optional[str] = None
    available_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    claimed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    completed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )

    __table_args__ = (
        CheckConstraint("resource_type in ('document', 'chat_memory', 'global_memory')", name="ck_embedding_job_resource_type"),
        CheckConstraint("status in ('pending', 'running', 'completed', 'failed')", name="ck_embedding_job_status"),
        CheckConstraint("attempts >= 0", name="ck_embedding_job_attempts"),
        CheckConstraint("length(btrim(resource_id)) > 0", name="ck_embedding_job_resource_id_nonempty"),
        CheckConstraint("length(btrim(scope_id)) > 0", name="ck_embedding_job_scope_id_nonempty"),
        CheckConstraint("length(btrim(embedding_model)) > 0", name="ck_embedding_job_model_nonempty"),
        CheckConstraint("length(btrim(source_version)) > 0", name="ck_embedding_job_source_version_nonempty"),
        UniqueConstraint(
            "resource_type", "resource_id", "scope_id", "embedding_model",
            name="uq_embedding_job_target",
        ),
        Index("idx_embedding_job_claim", "status", "available_at"),
        Index("idx_embedding_job_model_status", "resource_type", "embedding_model", "status"),
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
