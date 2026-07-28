from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent_workflows.workflow_runtime import default_agent_workflow_key
from app.models.llm_server_client import (
    DEFAULT_TOKEN_BUDGET,
    MAX_CUSTOM_INSTRUCTIONS_CHARS,
    REPLANS_LIMIT,
    MAX_SYSTEM_ROLE_CHARS,
)


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


class MemoryCreateRequest(BaseModel):
    """Request body for creating a canonical memory."""

    model_config = ConfigDict(extra="forbid")

    scope_type: str
    scope_id: str
    memory_type: str = "semantic"
    content: str
    summary: str = ""
    source_refs_json: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    visibility: str = "private"
    created_by: Optional[str] = None
    expires_at: Optional[datetime] = None


class MemorySearchRequest(BaseModel):
    """Request body for read-only scoped memory retrieval."""

    model_config = ConfigDict(extra="forbid")

    query: str
    allowed_scopes: Optional[List[str]] = None
    max_results: int = Field(default=10, ge=1, le=50)


class MemoryCandidateCreateRequest(BaseModel):
    """Request body for creating a memory promotion candidate."""

    proposed_scope_type: str
    proposed_scope_id: str
    memory_type: str = "semantic"
    content: str
    source_thread_id: Optional[str] = None
    source_project_id: Optional[str] = None
    source_agent_run_id: Optional[str] = None
    source_turn_id: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    created_by: Optional[str] = None


class MemoryCandidateResolveRequest(BaseModel):
    """Request body for resolving a promotion candidate."""

    model_config = ConfigDict(extra="forbid")

    status: str
    actor_id: Optional[str] = None


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
