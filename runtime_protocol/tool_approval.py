"""Framework-neutral policy and requests for human-gated tool invocation.

Runtimes own suspension and continuation. This module owns the meaning of a
decision, independently of the tool implementation or the agent framework.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

LANGGRAPH_TOOL_APPROVAL_RESPONSE = "run.resume"
HERMES_TOOL_APPROVAL_RESPONSE = "run.approval.respond"


class ApprovalMode(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ApprovalScope(str, Enum):
    INVOCATION = "invocation"
    RUN = "run"
    TASK = "task"


@dataclass(frozen=True)
class ToolApprovalPolicy:
    mode: ApprovalMode
    scope: ApprovalScope = ApprovalScope.RUN

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ToolApprovalPolicy:
        policy = cls(ApprovalMode(value["mode"]), ApprovalScope(value.get("scope", "run")))
        if policy.scope is ApprovalScope.INVOCATION:
            raise ValueError("The reusable approval scope must be run or task")
        return policy

    def to_dict(self) -> dict[str, str]:
        return {"mode": self.mode.value, "scope": self.scope.value}


def tool_approval_response_operation(runtime: str) -> str:
    """Map a runtime identity to the native continuation used after a tool gate."""
    if str(runtime or "").strip().lower() == "hermes":
        return HERMES_TOOL_APPROVAL_RESPONSE
    return LANGGRAPH_TOOL_APPROVAL_RESPONSE


def stable_invocation_id(identity: Mapping[str, Any]) -> str:
    """Deterministic invocation identity for MCP retries after a human pause."""
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def invocation_digest(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Bind approval to exact JSON arguments, with stable object-key order."""
    encoded = json.dumps(
        {"tool": tool_name, "arguments": dict(arguments)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def tool_approval_request(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    policy: ToolApprovalPolicy,
    caller: str,
    response_operation: str,
) -> dict[str, Any]:
    """One public approval contract for every runtime and every tool."""
    return {
        "type": "tool_approval",
        "kind": "approval",
        "mode": "approval",
        "phase": "before",
        "node_id": caller,
        "title": f"Approve {tool_name}?",
        "prompt": "Review this tool call before it runs.",
        "allowed_actions": ["approve", "approve_for_scope", "continue_without"],
        "default_action": "continue_without",
        "reject_behavior": "resume",
        "approval_scope_kind": policy.scope.value,
        "checkpoint_resume": True,
        "response_operation": response_operation,
        "response_schema": {},
        "proposed_tool": {
            "name": tool_name,
            "caller_node": caller,
            "arguments": dict(arguments),
            "argument_hash": invocation_digest(tool_name, arguments),
        },
    }


def decision_scope(action: str, policy: ToolApprovalPolicy) -> ApprovalScope | None:
    """Approval once never implicitly becomes a reusable grant."""
    if action == "approve":
        return ApprovalScope.INVOCATION
    if action == "approve_for_scope":
        return policy.scope
    if action in {"continue_without", "reject"}:
        return None
    raise ValueError(f"Unsupported tool approval action: {action!r}")


def effective_mode(policy: ToolApprovalPolicy, *, grant: str | None = None) -> ApprovalMode:
    """An explicit denial wins over any previously issued allowance."""
    if policy.mode is ApprovalMode.DENY or grant == "denied":
        return ApprovalMode.DENY
    if grant == "allowed":
        return ApprovalMode.ALLOW
    return policy.mode
