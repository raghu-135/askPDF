"""MCP-backed tool handles used by the application-facing agent surface."""

from app.mcp.tool_adapter import create_mcp_tool
from app.tools.contracts import InspectDocumentRequest, ReadContextRequest, SearchKnowledgeRequest
from app.tools.contracts import TimelineRequest as ThreadTimelineSearchInput

get_thread_shape = create_mcp_tool("get_thread_shape")
search_knowledge = create_mcp_tool("search_knowledge")
inspect_document = create_mcp_tool("inspect_document")
read_context = create_mcp_tool("read_context")
search_thread_conversation_history = create_mcp_tool("search_thread_conversation_history")
search_durable_memory = create_mcp_tool("search_durable_memory")
search_thread_events = create_mcp_tool("search_thread_events")

__all__ = [
    "ThreadTimelineSearchInput",
    "get_thread_shape",
    "search_knowledge",
    "inspect_document",
    "read_context",
    "search_thread_conversation_history",
    "search_durable_memory",
    "search_thread_events",
]
