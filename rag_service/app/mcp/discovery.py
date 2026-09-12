"""Client-side MCP tool discovery and descriptor validation."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
from app.tools.contracts import DocumentSearchRequest, FocusedDocumentSearchRequest, InternetSearchRequest, QueryRequest, TimelineRequest
from app.tools.thread_shape import ThreadShapeRequest
from app.tools.wikipedia import WikipediaRequest
from app.models.memory_tools import MemoryGetInput, MemoryPrepareChangeInput, MemorySearchInput


@dataclass(frozen=True)
class MCPDiscoveredTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    contract_id: str
    contract_version: str


_DISCOVERY_CACHE: dict[tuple[object, str], MCPDiscoveredTool] = {}


_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "get_thread_shape": ThreadShapeRequest,
    "search_documents": DocumentSearchRequest,
    "search_document_by_id": FocusedDocumentSearchRequest,
    "search_thread_conversation_history": DocumentSearchRequest,
    "search_durable_memory": DocumentSearchRequest,
    "search_thread_events": TimelineRequest,
    "search_web": QueryRequest,
    "wikipedia": WikipediaRequest,
    "memory_search": MemorySearchInput,
    "memory_get": MemoryGetInput,
    "memory_prepare_change": MemoryPrepareChangeInput,
    "internet_search": InternetSearchRequest,
}
for _name in ("wikidata", "arxiv", "pub_med", "pubmed", "semanticscholar", "semantic_scholar", "stack_exchange", "yahoo_finance_news"):
    _REQUEST_MODELS[_name] = QueryRequest


def request_model_for_tool(name: str) -> type[BaseModel]:
    return _REQUEST_MODELS[name]


def _cache_key(client: Any, name: str) -> tuple[object, str]:
    """Use stable transport identity while keeping test/fake clients isolated."""
    identity = getattr(client, "descriptor_cache_key", None)
    if identity is None:
        identity = id(client)
    return identity, name


def clear_discovery_cache() -> None:
    _DISCOVERY_CACHE.clear()


async def discover_tool(client: Any, name: str) -> MCPDiscoveredTool:
    cache_key = _cache_key(client, name)
    cached = _DISCOVERY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    listed = await client.request("tools/list", {})
    if not isinstance(listed, dict) or not isinstance(listed.get("tools"), list):
        raise RuntimeError("MCP discovery returned an invalid tools/list response")
    item = next((value for value in listed.get("tools", []) if value.get("name") == name), None)
    if item is None:
        raise RuntimeError(f"MCP discovery did not advertise tool {name!r}")
    metadata = item.get("_meta") or {}
    config = TOOL_FRIENDLY_CONFIG.get(name) or {}
    contract_id = metadata.get("com.askpdf/contract-id")
    version = metadata.get("com.askpdf/contract-version", "1")
    if contract_id != config.get("id") or version != config.get("contract_version", "1"):
        raise RuntimeError(f"MCP descriptor contract mismatch for {name!r}")
    if not isinstance(item.get("description"), str) or not item["description"].strip():
        raise RuntimeError(f"MCP descriptor is missing a description for {name!r}")
    if not isinstance(item.get("inputSchema"), dict) or not item["inputSchema"]:
        raise RuntimeError(f"MCP descriptor is missing inputSchema for {name!r}")
    if not isinstance(item.get("outputSchema"), dict) or not item["outputSchema"]:
        raise RuntimeError(f"MCP descriptor is missing outputSchema for {name!r}")
    discovered = MCPDiscoveredTool(
        name=name,
        description=str(item.get("description") or ""),
        input_schema=dict(item.get("inputSchema") or {}),
        output_schema=dict(item.get("outputSchema") or {}),
        contract_id=str(contract_id),
        contract_version=str(version),
    )
    _DISCOVERY_CACHE[cache_key] = discovered
    return discovered
