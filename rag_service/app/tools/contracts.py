"""Framework-neutral request and service contracts for first-party tools."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from app.agent.tool_contract import ToolResult
from app.tools.context import ToolInvocationContext


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


class InternetSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    reason: str = Field(default="Verify current external information", max_length=500)


class DocumentSearchRequest(QueryRequest):
    max_results: int = Field(default=10, ge=1, le=30)


class DocumentFilter(BaseModel):
    source_types: list[str] = Field(default_factory=list, max_length=10)
    tags: list[str] = Field(default_factory=list, max_length=20)
    pages: list[int] = Field(default_factory=list, max_length=100)


class SearchKnowledgeRequest(QueryRequest):
    level: Literal["document", "section", "chunk"] = "chunk"
    document_id: str | None = Field(default=None, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    section_id: str | None = Field(default=None, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    filters: DocumentFilter = Field(default_factory=DocumentFilter)
    max_results: int = Field(default=10, ge=1, le=30)


class InspectDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    section_id: str | None = Field(default=None, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    cursor: str | None = Field(default=None, max_length=512)
    page_size: int = Field(default=50, ge=1, le=200)


class ReadContextRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    expansion: Literal["chunk", "section", "table"] = "chunk"
    token_budget: int = Field(default=2000, ge=1, le=8000)
    cursor: str | None = Field(default=None, max_length=512)


class TimelineRequest(QueryRequest):
    sources: Literal["all", "conversation", "documents", "web_cache"] = "all"
    order: Literal["relevance", "oldest", "newest"] = "relevance"
    max_results: int = Field(default=10, ge=1, le=30)


class EmptyRequest(BaseModel):
    pass


class ToolServices(Protocol):
    """Dependency seam; concrete services are supplied by the application."""

    async def get_thread_shape(self, thread_id: str) -> dict[str, Any]: ...


Handler = Any
ToolHandler = Any

__all__ = [
    "DocumentSearchRequest", "EmptyRequest",
    "DocumentFilter", "InspectDocumentRequest", "ReadContextRequest", "SearchKnowledgeRequest",
    "QueryRequest", "TimelineRequest", "InternetSearchRequest", "ToolHandler", "ToolInvocationContext",
    "ToolResult", "ToolServices",
]
