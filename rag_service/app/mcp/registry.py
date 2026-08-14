"""Authoritative, typed MCP tool definitions."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from app.agent.tool_contract import ToolResult
from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG, get_tool_contract_id
from app.tools.contracts import DocumentSearchRequest, EmptyRequest, FocusedDocumentSearchRequest, QueryRequest, TimelineRequest
from app.tools.context import ToolInvocationContext
from app.tools.retrieval_conversation import search_thread_conversation_history
from app.tools.retrieval_documents import search_document_by_id, search_documents
from app.tools.retrieval_memory import search_durable_memory
from app.tools.retrieval_timeline import search_thread_events
from app.tools.web_search import search_web
from app.tools.wikipedia import WikipediaRequest, invoke_wikipedia
from app.tools.memory_manager import InternetSearchRequest, memory_get, memory_prepare_change, memory_search, internet_search
from app.models.memory_tools import MemoryGetInput, MemoryPrepareChangeInput, MemorySearchInput
from app.tools.external_research import search_external
from app.tools.thread_shape import ThreadShapeRequest, invoke_thread_shape

ToolHandler = Callable[..., Awaitable[ToolResult]]

TOOL_RESULT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["ok", "content", "sources", "artifacts", "warnings", "metrics", "trace"],
    "properties": {
        "ok": {"type": "boolean"},
        "content": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "object"}},
        "artifacts": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "error": {"anyOf": [{"type": "null"}, {"type": "object", "required": ["code", "message", "type", "retryable"], "properties": {"code": {"type": "string"}, "message": {"type": "string"}, "type": {"type": "string"}, "retryable": {"type": "boolean"}}}]},
        "metrics": {"type": "object"},
        "trace": {"type": "object"},
    },
}


@dataclass(frozen=True)
class MCPToolDefinition:
    name: str
    request_model: type[BaseModel]
    handler: ToolHandler
    registry_contract_id: str
    contract_version: str
    server_name: str

    @property
    def model(self) -> type[BaseModel]:
        return self.request_model

    def __iter__(self):
        """Compatibility with the original ``(model, handler)`` test/API shape."""
        yield self.request_model
        yield self.handler

    def __getitem__(self, index: int):
        if index == 0:
            return self.request_model
        if index == 1:
            return self.handler
        raise IndexError(index)


@dataclass(frozen=True)
class MCPServerDefinition:
    name: str
    tool_names: frozenset[str]


def logical_server_groups() -> dict[str, MCPServerDefinition]:
    groups: dict[str, set[str]] = {}
    for name, config in TOOL_FRIENDLY_CONFIG.items():
        if config.get("mcp_enabled", True):
            groups.setdefault(config.get("mcp_server", "first_party_context"), set()).add(name)
    return {name: MCPServerDefinition(name, frozenset(tools)) for name, tools in groups.items()}


def _neutral(name: str, model: type[BaseModel], handler: ToolHandler) -> MCPToolDefinition:
    config = TOOL_FRIENDLY_CONFIG[name]
    return MCPToolDefinition(name, model, handler, config["id"], config.get("contract_version", "1"), config.get("mcp_server", "first_party_context"))


_NEUTRAL: dict[str, MCPToolDefinition] = {
    "get_thread_shape": _neutral("get_thread_shape", ThreadShapeRequest, invoke_thread_shape),
    "search_documents": _neutral("search_documents", DocumentSearchRequest, search_documents),
    "search_document_by_id": _neutral("search_document_by_id", FocusedDocumentSearchRequest, search_document_by_id),
    "search_thread_conversation_history": _neutral("search_thread_conversation_history", DocumentSearchRequest, search_thread_conversation_history),
    "search_durable_memory": _neutral("search_durable_memory", DocumentSearchRequest, search_durable_memory),
    "search_thread_events": _neutral("search_thread_events", TimelineRequest, search_thread_events),
    "search_web": _neutral("search_web", QueryRequest, search_web),
    "wikipedia": _neutral("wikipedia", WikipediaRequest, invoke_wikipedia),
    "memory_search": _neutral("memory_search", MemorySearchInput, memory_search),
    "memory_get": _neutral("memory_get", MemoryGetInput, memory_get),
    "memory_prepare_change": _neutral("memory_prepare_change", MemoryPrepareChangeInput, memory_prepare_change),
    "internet_search": _neutral("internet_search", InternetSearchRequest, internet_search),
}

for _provider_name in ("wikidata", "arxiv", "pub_med", "pubmed", "semanticscholar", "semantic_scholar", "stack_exchange", "yahoo_finance_news"):
    if _provider_name in TOOL_FRIENDLY_CONFIG:
        async def _provider_handler(request: QueryRequest, context: ToolInvocationContext, *, _name: str = _provider_name):
            return await search_external(request, context, tool_name=_name)
        _NEUTRAL[_provider_name] = _neutral(_provider_name, QueryRequest, _provider_handler)


MCP_TOOL_DEFINITIONS: dict[str, MCPToolDefinition] = dict(_NEUTRAL)


def enabled_definitions() -> dict[str, MCPToolDefinition]:
    return dict(MCP_TOOL_DEFINITIONS)


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema() if hasattr(model, "model_json_schema") else model.schema()


def descriptor(name: str, definition: MCPToolDefinition | type[BaseModel]) -> dict[str, Any]:
    model = definition.request_model if isinstance(definition, MCPToolDefinition) else definition
    config = TOOL_FRIENDLY_CONFIG[name]
    return {"name": name, "description": config["description"], "inputSchema": _schema(model), "outputSchema": TOOL_RESULT_OUTPUT_SCHEMA, "_meta": {"com.askpdf/contract-id": config["id"], "com.askpdf/contract-version": config.get("contract_version", "1"), "com.askpdf/server": config.get("mcp_server")}}


def validate_registry() -> None:
    registry_names = {
        name for name, config in TOOL_FRIENDLY_CONFIG.items()
        if config.get("mcp_enabled", True)
    }
    missing = sorted(registry_names - set(MCP_TOOL_DEFINITIONS))
    if missing:
        raise RuntimeError(f"MCP registry has no framework-neutral handler for: {', '.join(missing)}")
    orphaned = sorted(set(MCP_TOOL_DEFINITIONS) - registry_names)
    if orphaned:
        raise RuntimeError(f"MCP registry contains tools not enabled in the authoritative registry: {', '.join(orphaned)}")
    for name, definition in MCP_TOOL_DEFINITIONS.items():
        config = TOOL_FRIENDLY_CONFIG.get(name)
        if not config:
            raise RuntimeError(f"MCP tool {name!r} is missing from TOOL_FRIENDLY_CONFIG")
        if config.get("mcp_enabled") and not callable(definition.handler):
            raise RuntimeError(f"MCP tool {name!r} has no handler")
        if config.get("mcp_tool", name) != name:
            raise RuntimeError(f"MCP tool name mismatch for {name!r}")
        if hasattr(definition.handler, "ainvoke"):
            raise RuntimeError(f"MCP tool {name!r} resolves to a framework tool instead of a handler")
        if definition.registry_contract_id != config["id"]:
            raise RuntimeError(f"MCP contract mismatch for {name!r}")
        _schema(definition.request_model)
        if not TOOL_RESULT_OUTPUT_SCHEMA.get("required"):
            raise RuntimeError(f"MCP tool {name!r} has no output schema")
    groups = logical_server_groups()
    grouped = [tool for group in groups.values() for tool in group.tool_names]
    if len(grouped) != len(set(grouped)):
        raise RuntimeError("MCP tool appears in multiple logical server groups")
    unknown = sorted(set(grouped) - set(MCP_TOOL_DEFINITIONS))
    if unknown:
        raise RuntimeError(f"Logical MCP server group contains unknown tools: {', '.join(unknown)}")


def validate_mcp_invocation(name: str, context: ToolInvocationContext) -> None:
    """Validate the protocol/domain boundary, not framework authorization.

    Caller-node and capability authorization is performed by the trusted
    workflow adapter before dispatch. Direct framework-neutral callers may
    legitimately omit those fields until execution-grant authentication is
    introduced.
    """
    if name not in TOOL_FRIENDLY_CONFIG:
        raise ValueError(f"Unknown tool: {name}")
    if name not in MCP_TOOL_DEFINITIONS:
        raise ValueError(f"MCP handler is unavailable for tool {name}")
    if not get_tool_contract_id(name):
        raise ValueError(f"Tool {name} has no contract ID")
