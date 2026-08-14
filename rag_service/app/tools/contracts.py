"""Framework-neutral request and service contracts for first-party tools."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from app.agent.tool_contract import ToolResult
from app.tools.context import ToolInvocationContext


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


class DocumentSearchRequest(QueryRequest):
    max_results: int = Field(default=10, ge=1, le=30)


class FocusedDocumentSearchRequest(DocumentSearchRequest):
    file_hash: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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
    "DocumentSearchRequest", "EmptyRequest", "FocusedDocumentSearchRequest",
    "QueryRequest", "TimelineRequest", "ToolHandler", "ToolInvocationContext",
    "ToolResult", "ToolServices",
]
