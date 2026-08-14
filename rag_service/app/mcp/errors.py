"""Typed failures for the mandatory MCP execution boundary."""

from __future__ import annotations

import asyncio
from typing import Any


class MCPUnavailableError(RuntimeError):
    """Raised when a tool cannot be executed because MCP is unavailable."""

    def __init__(
        self,
        tool_name: str,
        *,
        category: str,
        cause: BaseException,
        retryable: bool,
        mcp_request_id: str | None = None,
        tool_call_id: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.category = category
        self.cause = cause
        self.retryable = retryable
        self.mcp_request_id = mcp_request_id
        self.tool_call_id = tool_call_id
        self.run_id = run_id
        self.thread_id = thread_id
        super().__init__(f"MCP {category} failure for tool {tool_name}: {cause}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": "mcp_unavailable",
            "type": type(self).__name__,
            "message": str(self),
            "raw_message": str(self.cause),
            "retryable": self.retryable,
            "tool_name": self.tool_name,
            "category": self.category,
            "mcp_request_id": self.mcp_request_id,
            "tool_call_id": self.tool_call_id,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
        }


def classify_mcp_failure(exc: BaseException) -> tuple[str, bool]:
    if isinstance(exc, BaseExceptionGroup):
        categories = [classify_mcp_failure(child) for child in exc.exceptions]
        if any(category == "timeout" for category, _ in categories):
            return "timeout", True
        if any(category == "connection" for category, _ in categories):
            return "connection", True
        return "protocol", False
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "timeout", True
    if isinstance(exc, (OSError, ConnectionError)):
        return "connection", True
    return "protocol", False
