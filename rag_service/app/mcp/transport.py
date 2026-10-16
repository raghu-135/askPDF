"""Official MCP SDK transports used by workflow adapters."""

from __future__ import annotations

import os
import asyncio
from typing import Any, Mapping

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session



def _timeout_seconds() -> float:
    from app.mcp.config import mcp_request_timeout_seconds
    return mcp_request_timeout_seconds()


async def _bounded(awaitable):
    task = asyncio.create_task(awaitable)
    try:
        return await asyncio.wait_for(asyncio.shield(task), _timeout_seconds())
    except asyncio.TimeoutError as exc:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise TimeoutError("MCP request timed out") from exc
    except asyncio.CancelledError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


def _contains_timeout(error: BaseException) -> bool:
    return isinstance(error, TimeoutError) or (
        isinstance(error, BaseExceptionGroup)
        and any(_contains_timeout(child) for child in error.exceptions)
    )


def _first_runtime_error(error: BaseException) -> RuntimeError | None:
    if isinstance(error, RuntimeError):
        return error
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            found = _first_runtime_error(child)
            if found is not None:
                return found
    return None


def _content_text(result: types.CallToolResult) -> str:
    return "".join(
        getattr(item, "text", "")
        for item in (result.content or [])
        if getattr(item, "type", None) == "text"
    )


def _result_dict(result: types.CallToolResult) -> dict[str, Any]:
    structured = result.structuredContent
    return {
        "content": [{"type": "text", "text": _content_text(result)}],
        "structuredContent": structured,
        "isError": result.isError,
    }


class InProcessMCPClient:
    """SDK ClientSession over SDK memory streams."""

    descriptor_cache_key = "in_process"

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        from app.mcp.server import get_sdk_server

        params = dict(params or {})
        unknown_tool: str | None = None
        try:
            async with create_connected_server_and_client_session(get_sdk_server()) as session:
                if method == "initialize":
                    result = await _bounded(session.initialize())
                    return result.model_dump(by_alias=True, exclude_none=True)
                if method == "tools/list":
                    result = await _bounded(session.list_tools())
                    return {"tools": [item.model_dump(by_alias=True, exclude_none=True) for item in result.tools]}
                if method == "tools/call":
                    listed = await _bounded(session.list_tools())
                    known = {item.name for item in listed.tools}
                    if str(params.get("name") or "") not in known:
                        unknown_tool = str(params.get("name") or "")
                    if unknown_tool:
                        result = None
                    else:
                        metadata = params.get("_meta") or params.get("meta") or {}
                        result = await _bounded(session.call_tool(
                            str(params.get("name") or ""),
                            dict(params.get("arguments") or {}),
                            meta=metadata if isinstance(metadata, dict) else None,
                        ))
                    if result is not None:
                        return _result_dict(result)
                if method == "notifications/initialized":
                    return {}
                if not unknown_tool:
                    raise RuntimeError(f"Unsupported MCP method: {method}")
        except BaseExceptionGroup as exc:
            if _contains_timeout(exc):
                raise TimeoutError("MCP request timed out") from exc
            runtime_error = _first_runtime_error(exc)
            if runtime_error is not None:
                raise runtime_error from exc
            raise
        if unknown_tool:
            raise RuntimeError(f"Unknown tool: {unknown_tool}")


class LoopbackHTTPMCPClient:
    """SDK streamable HTTP ClientSession for the internal endpoint."""

    def __init__(self, url: str | None = None, http_client: Any = None):
        self.url = url or os.getenv("MCP_LOOPBACK_URL", "")
        self.http_client = http_client

    @property
    def descriptor_cache_key(self) -> str:
        return f"loopback_http:{self.url}"

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        from app.http_clients import get_http_client
        http_client = self.http_client or get_http_client("mcp")
        try:
            async with streamable_http_client(self.url, http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    if method == "initialize":
                        result = await _bounded(session.initialize())
                        return result.model_dump(by_alias=True, exclude_none=True)
                    if method == "tools/list":
                        result = await _bounded(session.list_tools())
                        return {"tools": [item.model_dump(by_alias=True, exclude_none=True) for item in result.tools]}
                    if method == "tools/call":
                        metadata = params.get("_meta") or params.get("meta") or {}
                        result = await _bounded(session.call_tool(
                            str(params.get("name") or ""),
                            dict(params.get("arguments") or {}),
                            meta=metadata if isinstance(metadata, dict) else None,
                        ))
                        return _result_dict(result)
                    raise RuntimeError(f"Unsupported MCP method: {method}")
        except BaseExceptionGroup as exc:
            if _contains_timeout(exc):
                raise TimeoutError("MCP request timed out") from exc
            runtime_error = _first_runtime_error(exc)
            if runtime_error is not None:
                raise runtime_error from exc
            raise


def get_mcp_client() -> InProcessMCPClient | LoopbackHTTPMCPClient:
    from app.mcp.config import mcp_transport, validate_mcp_configuration
    validate_mcp_configuration()
    return LoopbackHTTPMCPClient() if mcp_transport() == "loopback_http" else InProcessMCPClient()
