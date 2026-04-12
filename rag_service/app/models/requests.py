from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.llm_server_client import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOKEN_BUDGET,
    INTENT_AGENT_MAX_ITERATIONS,
    MAX_CUSTOM_INSTRUCTIONS_CHARS,
    MAX_MAX_ITERATIONS,
    MAX_SYSTEM_ROLE_CHARS,
    MIN_MAX_ITERATIONS,
)


class ThreadCreateRequest(BaseModel):
    """Request body for creating a thread."""

    name: str
    embed_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)


class ThreadUpdateRequest(BaseModel):
    """Request body for updating a thread."""

    name: str


class ThreadFileRequest(BaseModel):
    """Request body for adding a file to a thread."""

    file_hash: str
    file_name: str
    file_path: Optional[str] = None


class WebSourceRequest(BaseModel):
    """Request body for indexing a webpage into a thread."""

    url: str


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
    max_iterations: Optional[int] = Field(
        default=None, ge=MIN_MAX_ITERATIONS, le=MAX_MAX_ITERATIONS
    )
    system_role_override: Optional[str] = Field(
        default=None, max_length=MAX_SYSTEM_ROLE_CHARS
    )
    tool_instructions_override: Optional[Dict[str, str]] = None
    custom_instructions_override: Optional[str] = Field(
        default=None, max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS
    )
    use_intent_agent: Optional[bool] = None
    intent_agent_max_iterations: Optional[int] = Field(default=None, ge=1, le=10)
    intent_agent_skip_clarify: Optional[bool] = None
    reasoning_mode: Optional[bool] = None


class ThreadSettingsResponse(BaseModel):
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS, ge=MIN_MAX_ITERATIONS, le=MAX_MAX_ITERATIONS
    )
    system_role: str = Field(default="", max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions: Dict[str, str] = Field(default_factory=dict)
    custom_instructions: str = Field(
        default="", max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS
    )
    use_intent_agent: bool = True
    intent_agent_max_iterations: int = Field(
        default=INTENT_AGENT_MAX_ITERATIONS, ge=1, le=10
    )
    reasoning_mode: bool = True
    use_reranker: bool = True


class ThreadSettingsUpdateRequest(BaseModel):
    max_iterations: Optional[int] = Field(
        default=None, ge=MIN_MAX_ITERATIONS, le=MAX_MAX_ITERATIONS
    )
    system_role: Optional[str] = Field(default=None, max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions: Optional[Dict[str, str]] = None
    custom_instructions: Optional[str] = Field(
        default=None, max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS
    )
    use_intent_agent: Optional[bool] = None
    intent_agent_max_iterations: Optional[int] = Field(default=None, ge=1, le=10)
    reasoning_mode: Optional[bool] = None
    use_reranker: Optional[bool] = None


class ToolCatalogEntry(BaseModel):
    id: str
    display_name: str
    description: str
    default_prompt: str


class PromptDefaults(BaseModel):
    max_iterations: int
    min_max_iterations: int
    max_max_iterations: int
    context_window: int
    system_role: str
    tool_instructions: Dict[str, str]
    custom_instructions: str
    use_intent_agent: bool = True
    intent_agent_max_iterations: int = INTENT_AGENT_MAX_ITERATIONS
    reasoning_mode: bool = True
    use_reranker: bool = True


class PromptPreviewRequest(BaseModel):
    context_window: int = DEFAULT_TOKEN_BUDGET
    system_role: Optional[str] = Field(default=None, max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions: Optional[Dict[str, str]] = None
    custom_instructions: Optional[str] = Field(
        default=None, max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS
    )
    use_web_search: bool = False
    intent_agent_ran: bool = True
    reasoning_mode: bool = True


class RefreshWebSourceRequest(BaseModel):
    content_hash: Optional[str] = None
    confirmed: bool = False


class PdfParseRequest(BaseModel):
    """Request body for PDF parsing."""

    file_hash: str
    file_name: str
    backend_url: str
