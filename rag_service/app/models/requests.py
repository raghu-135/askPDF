from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.models.llm_server_client import (
    LOCAL_EMBEDDING_MODEL,
    DEFAULT_TOKEN_BUDGET,
    MAX_CUSTOM_INSTRUCTIONS_CHARS,
    REPLANS_LIMIT,
    MAX_SYSTEM_ROLE_CHARS,
)


class ThreadCreateRequest(BaseModel):
    """Request body for creating a thread."""

    name: str
    embed_model: str = Field(default=LOCAL_EMBEDDING_MODEL)


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
    use_reranker: Optional[bool] = None
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
    use_reranker: bool = True
    agent_pattern: Dict[str, str] = Field(default_factory=lambda: {"template_id": "router_rag_agent"})


class ThreadSettingsUpdateRequest(BaseModel):
    replans: Optional[int] = Field(default=None, ge=1, le=REPLANS_LIMIT)
    system_role: Optional[str] = Field(default=None, max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions: Optional[Dict[str, str]] = None
    custom_instructions: Optional[str] = Field(
        default=None, max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS
    )
    hitl_web_approval: Optional[bool] = None
    use_reranker: Optional[bool] = None
    agent_pattern: Optional[Dict[str, str]] = None


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
    use_reranker: bool = True


class PromptPreviewRequest(BaseModel):
    context_window: int = DEFAULT_TOKEN_BUDGET
    system_role: Optional[str] = Field(default=None, max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions: Optional[Dict[str, str]] = None
    custom_instructions: Optional[str] = Field(
        default=None, max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS
    )
    use_web_search: bool = False
    agent_pattern: Optional[Dict[str, str]] = None
    agent_pattern_id: Optional[str] = None
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
