"""Decode the canonical MCP adapter envelope for application consumers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecodedMCPResult:
    """Outer MCP result plus the JSON payload emitted by a tool handler."""

    envelope: dict[str, Any]
    payload: dict[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.envelope.get("ok", True))

    @property
    def warnings(self) -> list[Any]:
        return list(self.envelope.get("warnings") or [])

    @property
    def error(self) -> Any:
        return self.envelope.get("error")


def decode_mcp_result(raw: Any) -> DecodedMCPResult:
    """Decode one adapter result without losing its canonical envelope.

    Tool implementations commonly serialize their domain payload as the
    envelope's ``content`` string.  Plain text remains a payload under the
    ``content`` key, while malformed JSON is never allowed to erase warnings
    or errors from the outer MCP result.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {
                "ok": False,
                "content": raw,
                "error": {
                    "code": "mcp_protocol_error",
                    "message": "MCP result was not a canonical structured envelope",
                    "type": "MCPProtocolError",
                    "retryable": False,
                },
            }
    if not isinstance(raw, dict):
        raw = {
            "ok": False,
            "content": str(raw or ""),
            "error": {
                "code": "mcp_protocol_error",
                "message": "MCP result was not an object envelope",
                "type": "MCPProtocolError",
                "retryable": False,
            },
        }
    envelope = dict(raw)
    envelope_fields = {"ok", "warnings", "error", "artifacts", "sources", "metrics", "trace"}
    required_fields = {"ok", "content", "sources", "artifacts", "warnings", "metrics", "trace"}
    if not envelope_fields.intersection(envelope):
        # Some SDK versions expose a CallToolResult's text content but omit
        # structuredContent on the client model.  In that case the adapter
        # returns the handler's JSON text directly.  Treat it as the domain
        # payload while synthesizing only the missing outer envelope.
        return DecodedMCPResult(
            envelope={
                "ok": False,
                "content": json.dumps(raw, ensure_ascii=False),
                "error": {
                    "code": "mcp_protocol_error",
                    "message": "MCP result was missing the canonical envelope fields",
                    "type": "MCPProtocolError",
                    "retryable": False,
                },
            },
            payload=dict(raw),
        )
    missing = sorted(required_fields - set(envelope))
    if missing and envelope_fields.intersection(envelope):
        envelope["ok"] = False
        envelope.setdefault("error", {
            "code": "mcp_protocol_error",
            "message": f"MCP result was missing canonical fields: {', '.join(missing)}",
            "type": "MCPProtocolError",
            "retryable": False,
        })
    content = envelope.get("content")
    payload: dict[str, Any]
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            decoded = None
        payload = dict(decoded) if isinstance(decoded, dict) else {"content": content}
    elif isinstance(content, dict):
        payload = dict(content)
    else:
        payload = {"content": content if content is not None else ""}
    return DecodedMCPResult(envelope=envelope, payload=payload)
