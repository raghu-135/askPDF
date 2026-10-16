"""Bridge MCP human-gated tools to the pinned Hermes native approval API."""

import base64
import hashlib
import json
from typing import Any, Callable, Mapping


def _stable_invocation_id(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()

APPROVAL_RULE_PREFIX = "askpdf_tool:"


def decode_approval_request(event: dict[str, Any]) -> dict[str, Any] | None:
    key = str(event.get("pattern_key") or "")
    prefix = "plugin_rule:" + APPROVAL_RULE_PREFIX
    if not key.startswith(prefix):
        return None
    encoded = key[len(prefix):]
    request = json.loads(base64.urlsafe_b64decode(encoded))
    if not isinstance(request, dict) or not isinstance(request.get("proposed_tool"), dict):
        raise ValueError("Invalid Hermes tool approval request")
    return request


def wrap_mcp_handler(
    handler: Callable[..., str],
    request_approval: Callable[..., dict[str, Any]],
    *,
    tool_name: str | None = None,
) -> Callable[..., str]:
    def invoke(arguments: dict[str, Any], **kwargs: Any) -> str:
        payload_args = {
            key: value for key, value in dict(arguments or {}).items()
            if key != "_askpdf_invocation_id"
        }
        invocation_id = _stable_invocation_id({"tool": tool_name or "", "arguments": payload_args})
        arguments = {**payload_args, "_askpdf_invocation_id": invocation_id}
        raw = handler(arguments, **kwargs)
        payload = json.loads(raw)
        structured = payload.get("structuredContent") or {}
        request = (structured.get("artifacts") or {}).get("approval_request")
        if not isinstance(request, dict):
            return raw
        encoded = base64.urlsafe_b64encode(json.dumps(request, separators=(",", ":")).encode()).decode()
        decision = request_approval(
            str(request["proposed_tool"]["name"]),
            str(request.get("prompt") or "Approve this tool call"),
            rule_key=APPROVAL_RULE_PREFIX + encoded,
        )
        if not decision.get("approved"):
            return json.dumps({"result": "Tool skipped: human approval was denied or timed out. No external action was performed."})
        # The product persists the decision before acknowledging native
        # approval. MCP verifies it again; a native allowance alone is not
        # sufficient permission to execute the tool.
        return handler(arguments, **kwargs)
    return invoke
