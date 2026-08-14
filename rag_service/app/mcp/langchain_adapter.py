"""Generic LangChain-to-MCP compatibility adapters."""

import json
import logging
import asyncio
from uuid import uuid4
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
from app.mcp.context_codec import RUNTIME_CONTEXT_KEY, encode_context
from app.mcp.discovery import discover_tool, request_model_for_tool
from app.mcp.errors import MCPUnavailableError, classify_mcp_failure
from app.mcp.telemetry import inject_trace_context
from app.mcp.transport import get_mcp_client
from app.tools.context import ToolInvocationContext

logger = logging.getLogger(__name__)


async def _await_mcp_call(awaitable, cancellation_checker=None, cancellation_scope_id=None):
    task = asyncio.create_task(awaitable)
    if cancellation_checker is None:
        return await task
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=0.1)
            if done:
                break
            if await cancellation_checker():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                from app.tools.background_tasks import cancel_background_tasks
                await cancel_background_tasks(cancellation_scope_id)
                raise asyncio.CancelledError
        return await task
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            from app.tools.background_tasks import cancel_background_tasks
            await cancel_background_tasks(cancellation_scope_id)
        raise


def context_from_config(config: RunnableConfig | None, tool_call_id: str | None = None) -> ToolInvocationContext:
    configurable = config.get("configurable", {}) if config else {}
    metadata = config.get("metadata", {}) if config else {}
    return ToolInvocationContext.from_mapping({
        **metadata,
        **configurable,
        "thread_id": configurable.get("app_thread_id") or configurable.get("thread_id"),
        "tool_call_id": tool_call_id or configurable.get("tool_call_id"),
        "run_id": configurable.get("run_id") or configurable.get("agent_run_id") or metadata.get("run_id"),
        "caller_node": configurable.get("caller_node") or metadata.get("caller_node"),
        "caller_node_type": configurable.get("caller_node_type") or metadata.get("caller_node_type"),
        "caller_capabilities": configurable.get("caller_capabilities") or metadata.get("caller_capabilities"),
        "route": configurable.get("route") or metadata.get("route"),
        "traceparent": metadata.get("traceparent") or configurable.get("traceparent"),
        "tracestate": metadata.get("tracestate") or configurable.get("tracestate"),
        "tool_call_id": configurable.get("tool_call_id") or metadata.get("tool_call_id") or tool_call_id,
        "mcp_request_id": configurable.get("mcp_request_id") or metadata.get("mcp_request_id"),
    })


def _serialized_tool_result(result: dict[str, Any], text: str) -> str:
    structured = result.get("structuredContent") or {}
    payload: dict[str, Any] = {
        "ok": not bool(result.get("isError", False)),
        "content": structured.get("content", text),
        "sources": structured.get("sources", []),
        "artifacts": structured.get("artifacts", {}),
        "warnings": structured.get("warnings", []),
        "metrics": structured.get("metrics", {}),
        "trace": structured.get("trace", {}),
    }
    for key in ("transport", "mcp_mode", "mcp_server", "mcp_contract_version"):
        if structured.get(key) is not None:
            payload[key] = structured[key]
    if structured.get("error") is not None:
        payload["error"] = structured["error"]
    artifacts = payload["artifacts"] if isinstance(payload["artifacts"], dict) else {}
    for key, legacy_key in {
        "document_sources": "__document_sources__",
        "web_sources": "__web_sources__",
        "used_chat_ids": "__used_chat_ids__",
        "timeline_events": "__timeline_events__",
    }.items():
        if key in artifacts:
            payload[legacy_key] = artifacts[key]
    if payload["warnings"]:
        payload["__warnings__"] = payload["warnings"]
    if artifacts:
        payload["__artifacts__"] = artifacts
    return json.dumps(payload, ensure_ascii=False)


async def call_mcp_tool(name: str, arguments: dict[str, Any], config: RunnableConfig | None = None) -> str:
    context = context_from_config(config)
    # The SDK owns the JSON-RPC counter for a ClientSession.  In-process
    # calls may intentionally use short-lived sessions, so do not allow the
    # SDK's per-session numeric ID to become our cross-layer correlation ID.
    # Generate one at the trusted adapter boundary instead.  A caller-supplied
    # ID is preserved for nested/retried calls that already have correlation.
    if not context.mcp_request_id:
        context = ToolInvocationContext.from_mapping({
            **context.as_dict(),
            "mcp_request_id": f"mcp-{uuid4().hex}",
        })
    checker = ((config or {}).get("configurable") or {}).get("cancellation_checker") if config else None
    try:
        client = get_mcp_client()
        discovered = await _await_mcp_call(discover_tool(client, name), checker, context.cancellation_scope_id)
        if not discovered.output_schema.get("required"):
            raise RuntimeError(f"MCP tool {name!r} has no advertised output schema")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        category, retryable = classify_mcp_failure(exc)
        raise MCPUnavailableError(
            name,
            category=category,
            cause=exc,
            retryable=retryable,
            mcp_request_id=context.mcp_request_id,
            tool_call_id=context.tool_call_id,
            run_id=context.run_id,
            thread_id=context.thread_id,
        ) from exc
    metadata = encode_context(context)
    inject_trace_context(metadata[RUNTIME_CONTEXT_KEY])
    try:
        result = await _await_mcp_call(client.request("tools/call", {
            "name": name,
            "arguments": arguments,
            "_meta": metadata,
        }), checker, context.cancellation_scope_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        category, retryable = classify_mcp_failure(exc)
        raise MCPUnavailableError(
            name,
            category=category,
            cause=exc,
            retryable=retryable,
            mcp_request_id=context.mcp_request_id,
            tool_call_id=context.tool_call_id,
            run_id=context.run_id,
            thread_id=context.thread_id,
        ) from exc
    try:
        text = "".join(item.get("text", "") for item in result.get("content", []) if item.get("type") == "text")
        structured = result.get("structuredContent")
        required = {"ok", "content", "sources", "artifacts", "warnings", "metrics", "trace"}
        if not isinstance(structured, dict):
            if result.get("isError"):
                return json.dumps({
                    "ok": False,
                    "content": text,
                    "sources": [],
                    "artifacts": {},
                    "warnings": [],
                    "metrics": {},
                    "trace": {},
                    "error": {
                        "code": "mcp_protocol_error",
                        "message": "MCP returned an error without structuredContent",
                        "type": "MCPProtocolError",
                        "retryable": False,
                    },
                }, ensure_ascii=False)
            raise RuntimeError(f"MCP tool {name!r} returned no structuredContent")
        missing = sorted(required - set(structured))
        if missing:
            raise RuntimeError(f"MCP tool {name!r} returned malformed structuredContent; missing: {', '.join(missing)}")
        if not isinstance(structured.get("ok"), bool):
            raise RuntimeError(f"MCP tool {name!r} returned malformed structuredContent; ok must be boolean")
        is_error = bool(result.get("isError", False))
        if structured["ok"] != (not is_error):
            raise RuntimeError(
                f"MCP tool {name!r} returned contradictory success/error envelope: "
                f"structuredContent.ok={structured['ok']!r}, isError={is_error!r}"
            )
        return _serialized_tool_result(result, text)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        category, retryable = classify_mcp_failure(exc)
        raise MCPUnavailableError(
            name,
            category=category,
            cause=exc,
            retryable=retryable,
            mcp_request_id=context.mcp_request_id,
            tool_call_id=context.tool_call_id,
            run_id=context.run_id,
            thread_id=context.thread_id,
        ) from exc


def _arguments_from_input(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return {"query": value}
    if value is None:
        return {}
    return dict(value)


def create_mcp_langchain_tool(
    tool_name: str,
    request_model: type[Any] | None = None,
) -> BaseTool:
    """Create a caller-compatibility wrapper backed exclusively by MCP."""
    if request_model is None:
        request_model = request_model_for_tool(tool_name)

    async def invoke(*args: Any, config: RunnableConfig = None, **kwargs: Any) -> str:
        arguments = dict(kwargs)
        if args:
            arguments.update(_arguments_from_input(args[0]))
        mcp_result = await call_mcp_tool(tool_name, arguments, config)
        return mcp_result

    return StructuredTool.from_function(
        coroutine=invoke,
        name=tool_name,
        description=TOOL_FRIENDLY_CONFIG[tool_name]["description"],
        args_schema=request_model,
    )


def create_wikipedia_tool() -> BaseTool:
    return create_mcp_langchain_tool("wikipedia")


def create_thread_shape_tool() -> BaseTool:
    return create_mcp_langchain_tool("get_thread_shape")
