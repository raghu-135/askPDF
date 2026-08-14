"""Encode and decode trusted runtime metadata for MCP calls."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from app.tools.context import ToolInvocationContext

RUNTIME_CONTEXT_KEY = "com.askpdf/runtime-context"


@dataclass(frozen=True)
class MCPRequestContext:
    """Protocol metadata kept distinct from model-controlled tool arguments."""

    mcp_request_id: str
    tool_call_id: str | None = None
    deadline_at: datetime | None = None
    cancellation_token: Any = None


def encode_context(context: ToolInvocationContext) -> dict[str, Any]:
    return {RUNTIME_CONTEXT_KEY: context.as_dict()}


def decode_context(meta: Mapping[str, Any] | None) -> ToolInvocationContext:
    value = (meta or {}).get(RUNTIME_CONTEXT_KEY, {})
    if not isinstance(value, Mapping):
        value = {}
    return ToolInvocationContext.from_mapping(value)
