"""Hermes MCP registry wire names (matches upstream ``tools.mcp_tool_schema``)."""

from __future__ import annotations

import hashlib
import re

MCP_TOOL_NAME_PREFIX = "mcp__"
_MCP_TOOL_NAME_MAX_LENGTH = 64
_MCP_TOOL_NAME_HASH_LENGTH = 8


def sanitize_mcp_name_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))


def mcp_prefixed_tool_name(server_name: str, tool_name: str) -> str:
    full_name = (
        f"{MCP_TOOL_NAME_PREFIX}{sanitize_mcp_name_component(server_name)}__"
        f"{sanitize_mcp_name_component(tool_name)}"
    )
    if len(full_name) <= _MCP_TOOL_NAME_MAX_LENGTH:
        return full_name
    suffix = "_" + hashlib.sha256(full_name.encode("utf-8")).hexdigest()[:_MCP_TOOL_NAME_HASH_LENGTH]
    return full_name[: _MCP_TOOL_NAME_MAX_LENGTH - len(suffix)] + suffix
