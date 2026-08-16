"""Official MCP SDK server for askPDF first-party tools."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.lowlevel import Server
from mcp import types
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Mount

from app.mcp.config import mcp_mode, mcp_transport, validate_mcp_configuration
from app.mcp.context_codec import decode_context
from app.mcp.registry import (
    MCP_TOOL_DEFINITIONS,
    TOOL_RESULT_OUTPUT_SCHEMA,
    descriptor,
    enabled_definitions,
    validate_mcp_invocation,
    validate_registry,
)
from app.mcp.telemetry import extracted_trace_context, tool_span

logger = logging.getLogger(__name__)


def _schema(model: type[Any]) -> dict[str, Any]:
    return model.model_json_schema() if hasattr(model, "model_json_schema") else model.schema()


class MCPServer:
    """Registry-backed SDK server; handlers never cross back into LangChain."""

    protocol_version = "2025-06-18"

    def __init__(self) -> None:
        validate_mcp_configuration()
        validate_registry()
        self.sdk = Server("askpdf-first-party", version="1")
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.sdk.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name=name,
                    description=TOOL_FRIENDLY_CONFIG[name]["description"],
                    inputSchema=_schema(definition.request_model),
                    outputSchema=TOOL_RESULT_OUTPUT_SCHEMA,
                    _meta={
                        "com.askpdf/contract-id": TOOL_FRIENDLY_CONFIG[name]["id"],
                        "com.askpdf/contract-version": TOOL_FRIENDLY_CONFIG[name].get("contract_version", "1"),
                        "com.askpdf/server": TOOL_FRIENDLY_CONFIG[name].get("mcp_server"),
                    },
                )
                for name, definition in enabled_definitions().items()
            ]

        @self.sdk.call_tool(validate_input=True)
        async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
            definition = MCP_TOOL_DEFINITIONS.get(name)
            if definition is None or name not in enabled_definitions():
                raise ValueError(f"Unknown tool: {name}")
            request_context = self.sdk.request_context
            meta = request_context.meta
            if hasattr(meta, "model_dump"):
                meta = meta.model_dump(exclude_none=True)
            elif not isinstance(meta, dict):
                meta = dict(meta or {})
            context = decode_context(meta)
            if not context.mcp_request_id:
                context = context.__class__.from_mapping({
                    **context.as_dict(), "mcp_request_id": str(request_context.request_id),
                })
            validate_mcp_invocation(name, context)
            config = TOOL_FRIENDLY_CONFIG[name]
            logger.info(
                "MCP tool call start tool=%s thread_id=%s run_id=%s tool_call_id=%s mcp_request_id=%s",
                name, context.thread_id, context.run_id, context.tool_call_id, context.mcp_request_id,
            )
            async with extracted_trace_context(meta):
                async with tool_span(
                    "askpdf.mcp.tool", tool_name=name, contract_id=config["id"],
                    thread_id=context.thread_id,
                ):
                    request_model = definition.request_model
                    request = request_model.model_validate(arguments)
                    result = await definition.handler(request, context)
            structured = result.structured(
                contract_id=config["id"],
                contract_version=config.get("contract_version", "1"),
            )
            trace = structured.setdefault("trace", {})
            trace.setdefault("mcp_request_id", context.mcp_request_id)
            trace.setdefault("tool_call_id", context.tool_call_id)
            structured.update({
                "mcp_server": config.get("mcp_server"),
                "mcp_contract_version": config.get("contract_version", "1"),
                "transport": mcp_transport(),
                "mcp_mode": mcp_mode(),
            })
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=result.content)],
                structuredContent=structured,
                isError=not result.ok or result.error is not None,
            )


def get_http_app() -> Any:
    """Return the SDK streamable-HTTP app backed by the low-level Server."""
    validate_mcp_configuration()
    manager = StreamableHTTPSessionManager(
        app=get_sdk_server(),
        json_response=True,
        stateless=True,
        security_settings=TransportSecuritySettings(
            allowed_hosts=[
                "localhost",
                "127.0.0.1",
                "rag-service",
                "rag-service:8000",
                "host.docker.internal",
            ],
        ),
    )

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        async with manager.run():
            yield

    return Starlette(
        routes=[Mount("/", app=manager.handle_request)],
        lifespan=lifespan,
    )


TOOL_FRIENDLY_CONFIG = __import__(
    "app.agent.tool_registry", fromlist=["TOOL_FRIENDLY_CONFIG"]
).TOOL_FRIENDLY_CONFIG

_server: MCPServer | None = None


def get_server() -> MCPServer:
    global _server
    if _server is None:
        _server = MCPServer()
    return _server


def get_sdk_server() -> Server:
    return get_server().sdk
