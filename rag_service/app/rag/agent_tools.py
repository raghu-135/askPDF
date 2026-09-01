"""Deprecated import surface for retrieval tool callers.

Tool execution lives in ``app.tools`` and is exposed to workflows only through
the MCP adapters.  This module contains aliases solely for external callers
that have not yet moved their imports; it contains no retrieval implementation
and never executes a LangChain-owned tool.
"""

from app.mcp.tool_adapter import create_mcp_tool
from app.tools.contracts import FocusedDocumentSearchRequest as FocusedDocumentSearchInput
from app.tools.contracts import TimelineRequest as ThreadTimelineSearchInput

get_thread_shape = create_mcp_tool("get_thread_shape")
search_documents = create_mcp_tool("search_documents")
search_document_by_id = create_mcp_tool("search_document_by_id")
search_thread_conversation_history = create_mcp_tool("search_thread_conversation_history")
search_durable_memory = create_mcp_tool("search_durable_memory")
search_thread_events = create_mcp_tool("search_thread_events")

__all__ = [
    "FocusedDocumentSearchInput",
    "ThreadTimelineSearchInput",
    "get_thread_shape",
    "search_documents",
    "search_document_by_id",
    "search_thread_conversation_history",
    "search_durable_memory",
    "search_thread_events",
]
