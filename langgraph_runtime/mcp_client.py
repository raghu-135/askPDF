"""LangChain tool wrappers backed only by the control-plane MCP HTTP endpoint."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from uuid import uuid4

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict, Field

from langgraph_runtime.agent.tool_registry import TOOL_FRIENDLY_CONFIG


class MCPUnavailableError(RuntimeError):
    def __init__(self, tool_name: str, *, cause: BaseException | None = None, retryable: bool = True, **_: Any) -> None:
        super().__init__(f"MCP tool {tool_name!r} is unavailable")
        self.tool_name = tool_name
        self.cause = cause
        self.retryable = retryable

    def as_dict(self) -> dict[str, Any]:
        """Return the bounded runtime-local error shape consumed by workflows."""

        cause = str(self.cause or "MCP request failed")[:700]
        return {
            "code": "mcp_unavailable",
            "type": type(self).__name__,
            "message": str(self)[:700],
            "raw_message": cause,
            "retryable": self.retryable,
            "tool_name": self.tool_name,
        }


class _OpenArguments(BaseModel):
    model_config = ConfigDict(extra="allow")


class _QueryArguments(_OpenArguments):
    query: str = Field(min_length=1, max_length=4000)


class _DocumentSearchArguments(_QueryArguments):
    max_results: int = Field(default=10, ge=1, le=30)


class _FocusedDocumentSearchArguments(_DocumentSearchArguments):
    file_hash: str = Field(min_length=1, max_length=256)


class _TimelineArguments(_QueryArguments):
    sources: str = "all"
    order: str = "relevance"
    max_results: int = Field(default=10, ge=1, le=30)


_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    "search_documents": _DocumentSearchArguments,
    "search_document_by_id": _FocusedDocumentSearchArguments,
    "search_thread_conversation_history": _DocumentSearchArguments,
    "search_durable_memory": _DocumentSearchArguments,
    "search_thread_events": _TimelineArguments,
    "search_web": _QueryArguments,
    "wikipedia": _QueryArguments,
    "wikidata": _QueryArguments,
    "arxiv": _QueryArguments,
    "pub_med": _QueryArguments,
    "pubmed": _QueryArguments,
    "semanticscholar": _QueryArguments,
    "semantic_scholar": _QueryArguments,
    "stack_exchange": _QueryArguments,
    "yahoo_finance_news": _QueryArguments,
}


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return {"query": value}
    if value is None:
        return {}
    return dict(value)


async def _call(name: str, arguments: dict[str, Any], config: RunnableConfig | None) -> str:
    url = os.getenv("MCP_LOOPBACK_URL", "").strip()
    if not url:
        raise MCPUnavailableError(name, cause=RuntimeError("MCP_LOOPBACK_URL is required"))
    configurable = dict((config or {}).get("configurable") or {})
    token = str(configurable.get("mcp_execution_context_token") or "")
    if not token:
        raise MCPUnavailableError(name, cause=RuntimeError("MCP execution grant is missing"), retryable=False)
    timeout = float(os.getenv("MCP_REQUEST_TIMEOUT_SECONDS", "120"))
    metadata = {
        "mcp_request_id": str(configurable.get("mcp_request_id") or f"mcp-{uuid4().hex}"),
    }
    arguments = {**arguments, "_askpdf_context_token": token}
    try:
        async with httpx.AsyncClient(
            timeout=timeout, headers={"x-askpdf-execution-context": token}
        ) as client:
            async with streamable_http_client(url, http_client=client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await asyncio.wait_for(session.initialize(), timeout=timeout)
                    listed = await asyncio.wait_for(session.list_tools(), timeout=timeout)
                    if name not in {tool.name for tool in listed.tools}:
                        raise RuntimeError(f"MCP tool {name!r} was not advertised")
                    result = await asyncio.wait_for(
                        session.call_tool(name, arguments, meta=metadata), timeout=timeout
                    )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise MCPUnavailableError(name, cause=exc) from exc
    text = "".join(getattr(item, "text", "") for item in result.content or [] if getattr(item, "type", None) == "text")
    structured = result.structuredContent if isinstance(result.structuredContent, dict) else {}
    payload = {
        "ok": not bool(result.isError),
        "content": structured.get("content", text),
        "sources": structured.get("sources", []),
        "artifacts": structured.get("artifacts", {}),
        "warnings": structured.get("warnings", []),
        "metrics": structured.get("metrics", {}),
        "trace": structured.get("trace", {}),
    }
    if structured.get("error") is not None:
        payload["error"] = structured["error"]
    return json.dumps(payload, ensure_ascii=False)


def create_mcp_langchain_tool(tool_name: str, request_model: type[Any] | None = None) -> BaseTool:
    request_model = request_model or _ARGUMENT_MODELS.get(tool_name, _OpenArguments)

    async def invoke(*args: Any, config: RunnableConfig = None, **kwargs: Any) -> str:
        arguments = dict(kwargs)
        if args:
            arguments.update(_arguments(args[0]))
        return await _call(tool_name, arguments, config)

    metadata = TOOL_FRIENDLY_CONFIG.get(tool_name) or {}
    return StructuredTool.from_function(
        coroutine=invoke,
        name=tool_name,
        description=str(metadata.get("description") or tool_name),
        args_schema=request_model,
    )


def classify_mcp_failure(exc: BaseException) -> tuple[str, bool]:
    return ("timeout", True) if isinstance(exc, TimeoutError) else ("unavailable", True)
