"""askPDF compatibility hooks for the pinned Hermes gateway revision.

The pinned gateway discovers MCP servers at process startup. askPDF creates
isolated profiles after startup, so a profile-scoped /v1/toolsets request must
discover that profile's MCP server before the run is admitted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
from typing import Any


PINNED_REVISION = "bdd0a79c6a0ebc2344d5d6913c70bd89fa59c894"
logger = logging.getLogger("askpdf.hermes_compat")


def _registered_tools(server_names: list[str]) -> list[str]:
    from tools import mcp_tool

    result: list[str] = []
    with mcp_tool._lock:
        for name in server_names:
            server = mcp_tool._servers.get(name)
            result.extend(list(getattr(server, "_registered_tool_names", ()) or ()))
            result.extend(list(mcp_tool._lazy_server_tool_names.get(name, ()) or ()))
    return sorted(set(result))


def _context_header_digests(servers: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Return non-secret proof of the exact headers loaded by pinned Hermes."""
    result: dict[str, str] = {}
    for name, server in servers.items():
        headers = server.get("headers") if isinstance(server, dict) else None
        if not isinstance(headers, dict):
            continue
        token = next(
            (str(value) for key, value in headers.items() if str(key).lower() == "x-askpdf-execution-context"),
            "",
        )
        if token:
            result[str(name)] = hashlib.sha256(token.encode()).hexdigest()
    return result


def _retire_servers(server_names: list[str]) -> None:
    """Shut down only this profile's MCP servers and discard their headers."""
    from agent.async_utils import safe_schedule_threadsafe
    from tools import mcp_tool
    from tools.registry import registry

    with mcp_tool._lock:
        servers = [mcp_tool._servers.pop(name) for name in server_names if name in mcp_tool._servers]
        cached_tools = [
            tool_name
            for name in server_names
            for tool_name in (mcp_tool._lazy_server_tool_names.pop(name, ()) or ())
        ]
        loop = mcp_tool._mcp_loop
    for tool_name in cached_tools:
        registry.deregister(tool_name)
    if not servers or loop is None or not loop.is_running():
        return

    async def shutdown() -> None:
        await asyncio.gather(*(server.shutdown() for server in servers), return_exceptions=True)

    future = safe_schedule_threadsafe(
        shutdown(), loop, logger=logger,
        log_message="askPDF profile MCP retirement failed to schedule",
    )
    if future is not None:
        try:
            future.result(timeout=15)
        except BaseException as exc:
            logger.warning("askPDF profile MCP retirement failed: %s", type(exc).__name__)


def _install() -> None:
    from aiohttp import web
    from gateway.platforms.api_server import APIServerAdapter
    from hermes_cli.config import load_config
    from tools.mcp_tool import _load_mcp_config, register_mcp_servers

    original_toolsets = APIServerAdapter._handle_toolsets
    original_events = APIServerAdapter._handle_run_events

    async def handle_toolsets(self: Any, request: Any) -> Any:
        config = load_config()
        # The generic config loader expands against process env. The MCP
        # loader is the pinned revision's profile-aware interpolation path.
        servers = dict(_load_mcp_config() or {})
        server_names = sorted(str(name) for name in servers)
        if servers:
            await asyncio.to_thread(register_mcp_servers, servers)
        registered_tools = _registered_tools(server_names)
        response = await original_toolsets(self, request)
        if response.status != 200:
            return response
        try:
            payload = json.loads(response.body)
        except (TypeError, ValueError, json.JSONDecodeError):
            return response
        metadata = dict(config.get("askpdf_runtime_profile") or {})
        payload["askpdf_runtime_profile"] = {
            "name": str(metadata.get("name") or ""),
            "config_fingerprint": str(metadata.get("config_fingerprint") or ""),
            "mcp_server_names": server_names,
            "registered_tools": registered_tools,
            "mcp_context_header_sha256": _context_header_digests(servers),
        }
        return web.json_response(payload, status=response.status)

    async def handle_run_events(self: Any, request: Any) -> Any:
        config = load_config()
        server_names = sorted(str(name) for name in (config.get("mcp_servers") or {}))
        try:
            return await original_events(self, request)
        finally:
            if server_names:
                await asyncio.to_thread(_retire_servers, server_names)

    APIServerAdapter._handle_toolsets = handle_toolsets
    APIServerAdapter._handle_run_events = handle_run_events
    logger.info("Installed askPDF Hermes compatibility hook for %s", PINNED_REVISION)


if (
    os.getenv("ASKPDF_HERMES_COMPAT_ENABLED") == "1"
    and os.getenv("HERMES_UPSTREAM_REVISION", PINNED_REVISION) == PINNED_REVISION
    and "gateway" in sys.argv
):
    _install()
