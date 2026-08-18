from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent_workflows.workflow_runtime import default_agent_workflow_key
from app.models.llm_server_client import (
    DEFAULT_TOKEN_BUDGET,
    MAX_CUSTOM_INSTRUCTIONS_CHARS,
    REPLANS_LIMIT,
    MAX_SYSTEM_ROLE_CHARS,
)
from app.models.memory_manager_input_budget import MAX_MEMORY_MANAGER_REQUEST_MESSAGES
from app.models.memory_manager_budget import HARD_MAX_CANONICAL_OPERATIONS, HARD_MAX_RELATIONSHIP_TARGETS
from app.models.memory_tools import MemoryAttributes
from app.models.memory_limits import MAX_MEMORY_QUERY_CHARS


MAX_MEMORY_SEARCH_RESULTS = 50


class ThreadCreateRequest(BaseModel):
    """Request body for creating a thread."""

    model_config = ConfigDict(extra="forbid")

    name: str
    project_id: Optional[str] = None


class ThreadUpdateRequest(BaseModel):
    """Request body for updating a thread."""

    name: str


class ThreadBulkDeleteRequest(BaseModel):
    """Request body for deleting multiple threads."""

    thread_ids: List[str] = Field(default_factory=list)

    @field_validator("thread_ids")
    @classmethod
    def validate_thread_ids(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("thread_ids must contain at least one thread ID")
        return value


class ThreadBulkDeleteFailure(BaseModel):
    """Failure information for a thread that could not be deleted."""

    thread_id: str
    error: str


class ThreadBulkDeleteResponse(BaseModel):
    """Response body for deleting multiple threads."""

    deleted_thread_ids: List[str] = Field(default_factory=list)
    not_found_thread_ids: List[str] = Field(default_factory=list)
    failed_thread_ids: List[ThreadBulkDeleteFailure] = Field(default_factory=list)


class ThreadForkRequest(BaseModel):
    """Request body for forking a thread."""

    message_id: Optional[str] = None
    name: Optional[str] = None
    target_project_id: Optional[str] = None
    memory_copy_mode: Optional[str] = None


class ThreadProjectUpdateRequest(BaseModel):
    """Request body for moving a thread into a project."""

    project_id: str


class ProjectCreateRequest(BaseModel):
    """Request body for creating a project."""

    name: str
    embedding_model: str
    description: str = ""
    settings_json: Dict[str, Any] = Field(default_factory=dict)


class ProjectUpdateRequest(BaseModel):
    """Request body for updating a project."""

    name: Optional[str] = None
    description: Optional[str] = None
    settings_json: Optional[Dict[str, Any]] = None


class ProjectCloneRequest(BaseModel):
    """Request body for cloning a project snapshot."""

    model_config = ConfigDict(extra="forbid")

    name: str
    include_threads: bool = False


class MemorySearchRequest(BaseModel):
    """Request body for read-only scoped memory retrieval."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(max_length=MAX_MEMORY_QUERY_CHARS)
    allowed_scopes: Optional[List[str]] = None
    max_results: int = Field(default=10, ge=1, le=MAX_MEMORY_SEARCH_RESULTS)


class MemoryManagerMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MEMORY_QUERY_CHARS)
    choice_id: Optional[str] = Field(default=None, max_length=100)


class MemoryManagerContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_scope_type: Literal["user", "project", "thread"]
    selected_scope_id: str = Field(min_length=1)
    thread_id: Optional[str] = None
    project_id: Optional[str] = None


class MemoryManagerWebSearchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    approved: bool


class MemoryManagerWebSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    title: str = Field(default="Internet Search", max_length=500)
    url: str = Field(default="", max_length=4000)
    query: str = Field(min_length=1, max_length=1000)
    searched_at: str = Field(min_length=1, max_length=100)


class MemoryManagerConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["create", "edit", "conversation_review", "memory_review"]
    context: MemoryManagerContext
    memory_id: Optional[str] = None
    messages: List[MemoryManagerMessage] = Field(
        default_factory=list,
        max_length=MAX_MEMORY_MANAGER_REQUEST_MESSAGES,
    )
    llm_model: str = Field(min_length=1)
    context_window: int = Field(default=DEFAULT_TOKEN_BUDGET, ge=256, le=2_000_000)
    web_search_mode: Literal["off", "ask", "on"] = "off"
    web_search_decision: Optional[MemoryManagerWebSearchDecision] = None
    memory_review_cursor: Optional["MemoryConsistencyReviewCursor"] = None


class MemoryOverrideTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    expected_updated_at: str = Field(min_length=1)


class MemoryChangeOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["create", "update", "delete", "noop"]
    scope_type: Optional[Literal["user", "project", "thread"]] = None
    scope_id: Optional[str] = None
    memory_id: Optional[str] = None
    expected_updated_at: Optional[str] = None
    content: Optional[str] = Field(default=None, max_length=MAX_MEMORY_QUERY_CHARS)
    attributes: Optional[MemoryAttributes] = None
    override_targets: List[MemoryOverrideTarget] = Field(default_factory=list, max_length=20)
    semantic_action: Optional[Literal["create", "update", "delete", "move", "set_overrides"]] = None
    operation_group_id: Optional[str] = Field(default=None, max_length=100)
    move_source_memory_id: Optional[str] = None
    move_destination_memory_id: Optional[str] = None
    web_sources: List[MemoryManagerWebSource] = Field(default_factory=list, max_length=12)


class MemoryReviewCursor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1)
    reviewed_through_turn_id: str = Field(min_length=1)
    reviewed_through_created_at: datetime


class MemoryConsistencyReviewCursor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_type: Literal["user", "project", "thread"]
    context_id: str = Field(min_length=1)
    snapshot_at: datetime
    snapshot_scope_versions: Dict[str, int] = Field(default_factory=dict)
    anchor_position: int = Field(default=0, ge=0)
    reviewed_anchor_count: int = Field(default=0, ge=0)
    remaining_anchor_count: int = Field(default=0, ge=0)


class MemoryChangeApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: MemoryManagerContext
    confirmed: Literal[True]
    operations: List[MemoryChangeOperation] = Field(default_factory=list, max_length=20)
    review_cursor: Optional[MemoryReviewCursor] = None
    memory_review_cursor: Optional[MemoryConsistencyReviewCursor] = None
    actor_id: str = Field(default="ui", min_length=1, max_length=200)


class MemoryManagerOperation(BaseModel):
    """Explicit, browser-held operation in a unified memory plan."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "memory_create",
        "memory_update",
        "memory_delete",
        "memory_move",
        "memory_merge",
        "relationship_replace",
    ]
    memory_id: Optional[str] = None
    source_memory_id: Optional[str] = None
    destination_memory_id: Optional[str] = None
    scope_type: Optional[Literal["user", "project", "thread"]] = None
    scope_id: Optional[str] = None
    target_scope_type: Optional[Literal["user", "project", "thread"]] = None
    target_scope_id: Optional[str] = None
    content: Optional[str] = Field(default=None, max_length=MAX_MEMORY_QUERY_CHARS)
    attributes: Optional[MemoryAttributes] = None
    override_target_ids: List[str] = Field(default_factory=list, max_length=HARD_MAX_RELATIONSHIP_TARGETS)
    override_target_versions: Dict[str, str] = Field(default_factory=dict)
    expected_updated_at: Optional[str] = None
    operation_group_id: Optional[str] = Field(default=None, max_length=100)


class MemoryManagerPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["direct_edit", "conversation_extract", "consistency_review"]
    context: MemoryManagerContext
    messages: List[MemoryManagerMessage] = Field(default_factory=list, max_length=MAX_MEMORY_MANAGER_REQUEST_MESSAGES)
    memory_id: Optional[str] = None
    llm_model: str = Field(min_length=1)
    context_window: int = Field(default=DEFAULT_TOKEN_BUDGET, ge=256, le=2_000_000)
    review_round: int = Field(default=1, ge=1)
    review_cursor: Optional[MemoryReviewCursor] = None
    memory_review_cursor: Optional[MemoryConsistencyReviewCursor] = None
    review_id: Optional[str] = None
    web_search_mode: Literal["off", "ask", "on"] = "off"
    web_search_decision: Optional[MemoryManagerWebSearchDecision] = None


class MemoryManagerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    plan_hash: str
    mode: Literal["direct_edit", "conversation_extract", "consistency_review"]
    context: MemoryManagerContext
    state: Literal["proposal", "clarification", "no_changes", "blocked"] = "no_changes"
    message: str = ""
    choices: List[Dict[str, str]] = Field(default_factory=list)
    embedding_readiness: List[Dict[str, Any]] = Field(default_factory=list)
    pending_web_search: Optional[Dict[str, str]] = None
    web_sources: List[Dict[str, Any]] = Field(default_factory=list)
    consent: Optional[Dict[str, Any]] = None
    operations: List[MemoryManagerOperation] = Field(default_factory=list, max_length=HARD_MAX_CANONICAL_OPERATIONS)
    analysis: List[Dict[str, Any]] = Field(default_factory=list)
    review: Optional[Dict[str, Any]] = None
    memory_review: Optional[Dict[str, Any]] = None
    budget: Dict[str, int] = Field(default_factory=dict)
    review_id: Optional[str] = None
    next_cursor: Optional[Dict[str, Any]] = None
    scope_versions: Dict[str, int] = Field(default_factory=dict)


class MemoryManagerApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: MemoryManagerPlan
    plan_hash: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1, max_length=200)
    confirmed: Literal[True]
    actor_id: str = Field(default="ui", min_length=1, max_length=200)


class ThreadFileRequest(BaseModel):
    """Request body for adding a file to a thread."""

    file_hash: str
    file_name: str
    file_path: Optional[str] = None


class ThreadFileAnnotationsUpdateRequest(BaseModel):
    """Request body for persisting a thread/file annotation snapshot."""

    annotations: List[Dict[str, object]] = Field(default_factory=list)


class ThreadFileAnnotationsResponse(BaseModel):
    """Response body for a persisted thread/file annotation snapshot."""

    thread_id: str
    file_hash: str
    annotations: List[Dict[str, object]] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ThreadChatRequest(BaseModel):
    """Request body for thread-based chat."""

    thread_id: str
    question: str
    llm_model: str
    use_web_search: bool = False
    hitl_web_approval: Optional[bool] = None
    use_reranker: Optional[bool] = None
    bypass_clarification: bool = False
    context_window: int = DEFAULT_TOKEN_BUDGET  # Added context window size
    replans: Optional[int] = Field(default=None, ge=1, le=REPLANS_LIMIT)
    system_role_override: Optional[str] = Field(
        default=None, max_length=MAX_SYSTEM_ROLE_CHARS
    )
    tool_instructions_override: Optional[Dict[str, str]] = None
    custom_instructions_override: Optional[str] = Field(
        default=None, max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS
    )
    client_timezone: Optional[str] = Field(default=None, max_length=100)
    client_locale: Optional[str] = Field(default=None, max_length=50)
    client_now_iso: Optional[str] = Field(default=None, max_length=80)


class ThreadMemorySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_enabled: bool = True
    thread_reads_thread_memory: bool = True
    thread_reads_project_memory: bool = True
    thread_reads_user_memory: bool = False


class ThreadSettingsResponse(BaseModel):
    replans: int = Field(default=1, ge=1, le=REPLANS_LIMIT)
    system_role: str = Field(default="", max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions: Dict[str, str] = Field(default_factory=dict)
    custom_instructions: str = Field(
        default="", max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS
    )
    hitl_web_approval: bool = False
    use_reranker: bool = False
    agent_workflow: Dict[str, str] = Field(default_factory=lambda: {"workflow_id": default_agent_workflow_key()})
    memory: ThreadMemorySettings = Field(default_factory=ThreadMemorySettings)


class ThreadSettingsUpdateRequest(BaseModel):
    replans: Optional[int] = Field(default=None, ge=1, le=REPLANS_LIMIT)
    system_role: Optional[str] = Field(default=None, max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions: Optional[Dict[str, str]] = None
    custom_instructions: Optional[str] = Field(
        default=None, max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS
    )
    hitl_web_approval: Optional[bool] = None
    use_reranker: Optional[bool] = None
    agent_workflow: Optional[Dict[str, str]] = None
    memory: Optional[ThreadMemorySettings] = None


class ToolCatalogEntry(BaseModel):
    id: str
    display_name: str
    description: str
    default_prompt: str


class PromptDefaults(BaseModel):
    replans_limit: int
    context_window: int
    system_role: str
    tool_instructions: Dict[str, str]
    custom_instructions: str
    hitl_web_approval: bool = False
    use_reranker: bool = False


class PromptPreviewRequest(BaseModel):
    context_window: int = DEFAULT_TOKEN_BUDGET
    system_role: Optional[str] = Field(default=None, max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions: Optional[Dict[str, str]] = None
    custom_instructions: Optional[str] = Field(
        default=None, max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS
    )
    use_web_search: bool = False
    agent_workflow: Optional[Dict[str, str]] = None
    agent_workflow_id: Optional[str] = None
    client_timezone: Optional[str] = Field(default=None, max_length=100)
    client_locale: Optional[str] = Field(default=None, max_length=50)
    client_now_iso: Optional[str] = Field(default=None, max_length=80)


class PdfParseRequest(BaseModel):
    """Request body for PDF parsing."""

    file_hash: str
    file_name: str
    backend_url: str


class ProcessPdfRequest(BaseModel):
    """Request body for thread-owned PDF processing."""

    thread_id: str
    file_hash: str
    file_name: str
    backend_url: str
