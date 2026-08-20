"""Official MCP SDK server for askPDF first-party tools."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from mcp.server.lowlevel import Server
from mcp import types
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Mount

from app.mcp.config import mcp_mode, mcp_transport, validate_mcp_configuration
from app.mcp.context_codec import decode_context
from app.mcp.execution_context_token import (
    TOKEN_ARGUMENT,
    TOKEN_HEADER,
    ExecutionContextTokenError,
    decode_execution_context_token,
    verified_token_run_id,
)
from app.mcp.registry import (
    MCP_TOOL_DEFINITIONS,
    TOOL_RESULT_OUTPUT_SCHEMA,
    descriptor,
    enabled_definitions,
    validate_mcp_invocation,
    validate_registry,
)
from app.mcp.telemetry import extracted_trace_context, tool_span
from app.mcp.tool_audit import persist_tool_audit

logger = logging.getLogger(__name__)
_transport_execution_token: ContextVar[str | None] = ContextVar(
    "askpdf_mcp_execution_token", default=None,
)


def _schema(model: type[Any]) -> dict[str, Any]:
    return model.model_json_schema() if hasattr(model, "model_json_schema") else model.schema()


class MCPServer:
    """Registry-backed SDK server; handlers never cross back into LangChain."""

    protocol_version = "2025-06-18"

    def __init__(self, *, allowed_tools: frozenset[str] | None = None, require_execution_token: bool = False) -> None:
        validate_mcp_configuration()
        validate_registry()
        self.allowed_tools = allowed_tools
        self.require_execution_token = require_execution_token
        self.sdk = Server("askpdf-first-party", version="1")
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.sdk.list_tools()
        async def list_tools() -> list[types.Tool]:
            if self.require_execution_token:
                execution_token = _transport_execution_token.get()
                if not execution_token:
                    logger.warning("Hermes MCP discovery rejected reason=missing")
                    raise ValueError("Hermes MCP execution context is required")
                try:
                    decode_execution_context_token(str(execution_token))
                except ExecutionContextTokenError as exc:
                    logger.warning("Hermes MCP discovery rejected reason=%s", exc.reason)
                    raise ValueError("Invalid Hermes MCP execution context") from exc
            return [
                types.Tool(
                    name=name,
                    description=TOOL_FRIENDLY_CONFIG[name]["description"],
                    inputSchema=self._input_schema(definition.request_model),
                    outputSchema=TOOL_RESULT_OUTPUT_SCHEMA,
                    _meta={
                        "com.askpdf/contract-id": TOOL_FRIENDLY_CONFIG[name]["id"],
                        "com.askpdf/contract-version": TOOL_FRIENDLY_CONFIG[name].get("contract_version", "1"),
                        "com.askpdf/server": TOOL_FRIENDLY_CONFIG[name].get("mcp_server"),
                    },
                )
                for name, definition in enabled_definitions().items()
                if self.allowed_tools is None or name in self.allowed_tools
            ]

        @self.sdk.call_tool(validate_input=True)
        async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
            definition = MCP_TOOL_DEFINITIONS.get(name)
            if definition is None or name not in enabled_definitions() or (
                self.allowed_tools is not None and name not in self.allowed_tools
            ):
                raise ValueError(f"Unknown tool: {name}")
            arguments = dict(arguments or {})
            argument_token = arguments.pop(TOKEN_ARGUMENT, None)
            execution_token = _transport_execution_token.get() if self.require_execution_token else argument_token
            if self.require_execution_token and not execution_token:
                logger.warning("Hermes MCP tool rejected tool=%s reason=missing", name)
                raise ValueError("Hermes MCP execution context is required")
            request_context = self.sdk.request_context
            meta = request_context.meta
            if hasattr(meta, "model_dump"):
                meta = meta.model_dump(exclude_none=True)
            elif not isinstance(meta, dict):
                meta = dict(meta or {})
            context = decode_context(meta)
            if execution_token:
                try:
                    context = decode_execution_context_token(str(execution_token), tool_name=name)
                except ExecutionContextTokenError as exc:
                    logger.warning("Hermes MCP tool rejected tool=%s reason=%s", name, exc.reason)
                    rejected_run_id = verified_token_run_id(str(execution_token))
                    if rejected_run_id:
                        await persist_tool_audit(
                            run_id=rejected_run_id,
                            request_id=str(request_context.request_id),
                            phase="failed",
                            tool_name=name,
                            payload={
                                "failure_stage": "execution_context",
                                "token_rejection_reason": exc.reason,
                                "error": {"code": "mcp_execution_context_rejected", "retryable": exc.reason == "expired"},
                            },
                        )
                    raise ValueError("Invalid Hermes MCP execution context") from exc
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
            audit_request_id = str(context.mcp_request_id or request_context.request_id)
            await persist_tool_audit(
                run_id=str(context.run_id or ""), request_id=audit_request_id,
                phase="started", tool_name=name,
                payload={"argument_names": sorted(arguments)},
            )
            async with extracted_trace_context(meta):
                async with tool_span(
                    "askpdf.mcp.tool", tool_name=name, contract_id=config["id"],
                    thread_id=context.thread_id,
                ):
                    try:
                        request_model = definition.request_model
                        request = request_model.model_validate(arguments)
                        result = await definition.handler(request, context)
                    except Exception as exc:
                        missing_fields = []
                        invalid_fields = []
                        if hasattr(exc, "errors"):
                            for issue in exc.errors():
                                field = ".".join(str(value) for value in issue.get("loc") or [])
                                if issue.get("type") == "missing":
                                    missing_fields.append(field)
                                elif field:
                                    invalid_fields.append(field)
                        await persist_tool_audit(
                            run_id=str(context.run_id or ""), request_id=audit_request_id,
                            phase="failed", tool_name=name,
                            payload={
                                "failure_stage": "arguments" if missing_fields or invalid_fields else "handler",
                                "missing_arguments": sorted(missing_fields),
                                "invalid_arguments": sorted(invalid_fields),
                                "error": {"code": "tool_arguments_invalid" if missing_fields or invalid_fields else "tool_execution_failed", "retryable": True},
                            },
                        )
                        raise
            await persist_tool_audit(
                run_id=str(context.run_id or ""), request_id=audit_request_id,
                phase="completed" if result.ok and result.error is None else "failed",
                tool_name=name, result=result,
                payload={"failure_stage": "handler"} if not result.ok or result.error is not None else None,
            )
            structured = result.structured(
                contract_id=config["id"],
                contract_version=config.get("contract_version", "1"),
            )
            trace = structured.setdefault("trace", {})
            trace.setdefault("mcp_request_id", context.mcp_request_id)
            trace.setdefault("tool_call_id", context.tool_call_id)
            structured["result_count"] = len(result.sources)
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

    def _input_schema(self, model: type[Any]) -> dict[str, Any]:
        schema = dict(_schema(model))
        properties = dict(schema.get("properties") or {})
        if not self.require_execution_token:
            properties[TOKEN_ARGUMENT] = {
                "type": "string",
                "description": "Opaque askPDF execution context supplied by an authorized runtime.",
            }
        schema["properties"] = properties
        return schema


def get_http_app(*, allowed_tools: frozenset[str] | None = None, require_execution_token: bool = False) -> Any:
    """Return the SDK streamable-HTTP app backed by the low-level Server."""
    validate_mcp_configuration()

    server = MCPServer(allowed_tools=allowed_tools, require_execution_token=require_execution_token)

    def create_manager() -> StreamableHTTPSessionManager:
        return StreamableHTTPSessionManager(
            app=server.sdk,
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

    # The SDK manager is single-use: its run() context cannot be entered a
    # second time. Starlette applications may nevertheless be started and
    # stopped repeatedly by tests, reloaders, and embedded service hosts.
    # Keep the mounted ASGI endpoint stable while replacing the manager for
    # each application lifespan.
    manager: StreamableHTTPSessionManager | None = None

    async def handle_request(scope: Any, receive: Any, send: Any) -> None:
        if manager is None:
            raise RuntimeError("MCP HTTP application is not running")
        token = None
        if scope.get("type") == "http":
            token = next((value.decode("latin-1") for key, value in scope.get("headers") or [] if key.decode("latin-1").lower() == TOKEN_HEADER), None)
        marker = _transport_execution_token.set(token)
        try:
            await manager.handle_request(scope, receive, send)
        finally:
            _transport_execution_token.reset(marker)

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        nonlocal manager
        manager = create_manager()
        try:
            async with manager.run():
                yield
        finally:
            manager = None

    return Starlette(
        routes=[Mount("/", app=handle_request)],
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
