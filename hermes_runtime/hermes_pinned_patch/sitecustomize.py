"""askPDF pinned-revision hooks for the Hermes gateway.

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
from typing import Any, Mapping


PINNED_REVISION = "2bd378ed1b8d6bf2192cb26e0a55ff3091e6bd19"
logger = logging.getLogger("askpdf.hermes_pinned_patch")


def _apply_initial_tool_requirement(agent: Any, api_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Consume askPDF's profile-local first-turn tool requirement once."""
    if not bool(getattr(agent, "_askpdf_require_initial_tool", False)):
        return api_kwargs
    agent._askpdf_require_initial_tool = False
    if api_kwargs.get("tools"):
        api_kwargs["tool_choice"] = "required"
    return api_kwargs


def _registered_tools(server_names: list[str], *, registered: list[str] | None = None) -> list[str]:
    if registered:
        return sorted({str(value) for value in registered})
    from tools.mcp_tool_registration import _existing_tool_names

    names = {str(value) for value in _existing_tool_names()}
    if server_names:
        prefixes = tuple(f"mcp__{name}__" for name in server_names)
        names = {value for value in names if value.startswith(prefixes)}
    return sorted(names)


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
    from tools import mcp_tool
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    scope = mcp_tool._mcp_registry_scope()
    expected = {str(name) for name in server_names if name}
    if not expected:
        return
    shutdown_mcp_servers(scope=scope, names=expected, timeout=15.0)


def _install() -> None:
    from aiohttp import web
    from gateway.platforms.api_server import APIServerAdapter
    from hermes_cli.config import load_config
    from tools.mcp_tool_config import _load_mcp_config
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools import mcp_tool
    from tools import mcp_tool_handlers
    from tools.approval import request_tool_approval
    from approval_bridge import wrap_mcp_handler
    from agent import chat_completion_helpers
    from agent import turn_usage
    from gateway.platforms import api_server_runs
    from gateway.run_turn_runner import _settle_api_runtime_lock
    from openrouter_provider_usage import (
        accumulate_agent_provider_cost,
        enrich_terminal_usage,
        merge_openrouter_usage_request,
    )

    original_toolsets = APIServerAdapter._handle_toolsets
    original_sweep = APIServerAdapter._sweep_orphaned_runs_once
    original_create_agent = APIServerAdapter._create_agent
    original_build_api_kwargs = chat_completion_helpers.build_api_kwargs
    original_record_response_usage = turn_usage.record_response_usage
    original_run_usage = api_server_runs._run_usage
    original_settle_api_runtime_lock = _settle_api_runtime_lock
    original_make_tool_handler = mcp_tool_handlers._make_tool_handler

    def settle_api_runtime_lock(agent: Any, result: Any, captured: dict[str, Any]) -> None:
        outcome = original_settle_api_runtime_lock(agent, result, captured)
        usage = captured.get("usage")
        if isinstance(usage, Mapping):
            captured["usage"] = enrich_terminal_usage(dict(usage), agent)
        return outcome

    def make_tool_handler(server_name: str, tool_name: str, tool_timeout: float, *, native_images: bool = False):
        handler = original_make_tool_handler(
            server_name, tool_name, tool_timeout, native_images=native_images,
        )
        if not server_name.startswith("askpdf_"):
            return handler
        return wrap_mcp_handler(
            handler,
            request_tool_approval,
            tool_name=tool_name,
        )

    mcp_tool_handlers._make_tool_handler = make_tool_handler

    def create_agent(self: Any, *args: Any, **kwargs: Any) -> Any:
        model_options = kwargs.get("model_options")
        require_initial_tool = bool(
            isinstance(model_options, dict)
            and model_options.get("askpdf_require_initial_tool") is True
        )
        agent = original_create_agent(self, *args, **kwargs)
        agent._askpdf_require_initial_tool = require_initial_tool
        return agent

    def build_api_kwargs(agent: Any, api_messages: list[Any], tools_for_api: list[Any] | None = None) -> dict[str, Any]:
        from runtime_protocol.provider_tool_schema import provider_compatible_function_tools

        kwargs = original_build_api_kwargs(agent, api_messages, tools_for_api=tools_for_api)
        kwargs = _apply_initial_tool_requirement(agent, kwargs)
        if kwargs.get("tools"):
            kwargs["tools"] = provider_compatible_function_tools(kwargs["tools"])
        return merge_openrouter_usage_request(
            kwargs,
            base_url=str(getattr(agent, "base_url", "") or ""),
            model=str(getattr(agent, "model", "") or ""),
            provider=str(getattr(agent, "provider", "") or ""),
        )

    def record_response_usage(agent: Any, response: Any, *args: Any, **kwargs: Any) -> Any:
        outcome = original_record_response_usage(agent, response, *args, **kwargs)
        accumulate_agent_provider_cost(agent, response)
        return outcome

    def run_usage(agent: Any) -> dict[str, Any]:
        usage = enrich_terminal_usage(original_run_usage(agent), agent)
        return usage

    # ``turn_response_check`` binds ``record_response_usage`` at import time; rebind after
    # we replace ``turn_usage.record_response_usage`` so streaming turns accumulate cost.
    try:
        import agent.turn_response_check as turn_response_check

        turn_response_check.record_response_usage = record_response_usage
    except Exception:
        logger.debug("Could not rebind turn_response_check.record_response_usage", exc_info=True)

    async def handle_toolsets(self: Any, request: Any) -> Any:
        config = load_config()
        # The generic config loader expands against process env. The MCP
        # loader is the pinned revision's profile-aware interpolation path.
        servers = dict(_load_mcp_config() or {})
        server_names = sorted(str(name) for name in servers)
        registered: list[str] = []
        if servers:
            registered = list(await asyncio.to_thread(register_mcp_servers, servers) or ())
        response = await original_toolsets(self, request)
        if response.status != 200:
            return response
        # The pinned gateway discovers/registers MCP tools while serving the
        # toolsets request. Read registration state after the original handler
        # has completed so activation metadata reflects the actual toolset.
        registered_tools = _registered_tools(server_names, registered=registered)
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

    async def handle_run_events(self: Any, request: Any, *, _api_server: Any = None) -> Any:
        from event_stream import serve_run_events
        from gateway.platforms import api_server as gateway_api_server
        from gateway.platforms.api_server import _sse_frame
        if _api_server is None:
            _api_server = gateway_api_server
        config = load_config()
        server_names = sorted(str(name) for name in (config.get("mcp_servers") or {}))

        async def retire_after_terminal() -> None:
            if server_names:
                await asyncio.to_thread(_retire_servers, server_names)

        return await serve_run_events(
            self,
            request,
            web=web,
            encode_frame=_sse_frame,
            on_terminal=retire_after_terminal,
            _api_server=_api_server,
        )

    def sweep_orphaned_runs(self: Any, now: float | None = None) -> None:
        from event_stream import sweep_run_events
        sweep_run_events(self, original_sweep, now)

    APIServerAdapter._handle_toolsets = handle_toolsets
    APIServerAdapter._handle_run_events = handle_run_events
    APIServerAdapter._sweep_orphaned_runs_once = sweep_orphaned_runs
    APIServerAdapter._create_agent = create_agent
    chat_completion_helpers.build_api_kwargs = build_api_kwargs
    turn_usage.record_response_usage = record_response_usage
    api_server_runs._run_usage = run_usage
    import gateway.run_turn_runner as run_turn_runner

    run_turn_runner._settle_api_runtime_lock = settle_api_runtime_lock
    logger.info("Installed askPDF pinned Hermes patch for %s", PINNED_REVISION)


if (
    os.getenv("ASKPDF_HERMES_COMPAT_ENABLED") == "1"
    and os.getenv("HERMES_UPSTREAM_REVISION", PINNED_REVISION) == PINNED_REVISION
):
    _install()
