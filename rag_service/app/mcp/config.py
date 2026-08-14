"""Configuration for the mandatory first-party MCP boundary."""

import os
import math
from urllib.parse import urlparse


def mcp_request_timeout_seconds() -> float:
    raw = os.getenv("MCP_REQUEST_TIMEOUT_SECONDS", "30").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid MCP_REQUEST_TIMEOUT_SECONDS={raw!r}; expected a positive number of seconds"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(
            f"Invalid MCP_REQUEST_TIMEOUT_SECONDS={raw!r}; expected a positive number of seconds"
        )
    return value


def mcp_transport() -> str:
    value = os.getenv("MCP_TRANSPORT", "in_process").strip().lower()
    if value not in {"in_process", "loopback_http"}:
        raise RuntimeError(
            f"Invalid MCP_TRANSPORT={value!r}; expected 'in_process' or 'loopback_http'"
        )
    return value


def validate_mcp_configuration() -> None:
    transport = mcp_transport()
    mcp_request_timeout_seconds()
    if transport == "loopback_http":
        value = os.getenv("MCP_LOOPBACK_URL", "").strip()
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError(
                "MCP_LOOPBACK_URL must be an absolute http:// or https:// URL when MCP_TRANSPORT=loopback_http"
            )
    enabled = os.getenv("MCP_ENABLED")
    if enabled is not None and enabled.strip().lower() in {"0", "false", "no", "off"}:
        raise RuntimeError(
            "MCP-only execution does not support MCP_ENABLED=false; remove the setting"
        )
    mode = os.getenv("MCP_TOOL_MODE", "").strip().lower()
    if mode in {"legacy", "shadow"}:
        raise RuntimeError(
            f"MCP-only execution does not support MCP_TOOL_MODE={mode!r}; remove the setting"
        )
    if mode and mode != "mcp":
        raise RuntimeError(f"Invalid MCP_TOOL_MODE={mode!r}; remove the setting")


def mcp_mode() -> str:
    """Return the fixed execution mode for structured trace compatibility."""
    return "mcp"
