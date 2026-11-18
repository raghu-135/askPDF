"""Delegated children must inherit parent web-search approval settings."""

from __future__ import annotations

import pytest

from app.services import tool_approval
from app.tools.context import ToolInvocationContext
from runtime_protocol.delegation_inheritance import merge_delegate_inherited_tool_settings
from runtime_protocol.tool_approval import ApprovalMode


def test_delegated_child_config_compiles_ask_web_policy():
    inherited = merge_delegate_inherited_tool_settings({"web_search_mode": "ask", "use_web_search": True})
    child_config = {
        "use_web_search": True,
        "allowed_tool_ids": ["live_web_recon"],
        **inherited,
    }
    policies = tool_approval.invocation_policies(child_config)
    assert policies["search_web"]["mode"] == ApprovalMode.ASK.value


@pytest.mark.asyncio
async def test_delegated_child_web_search_under_ask_interrupts_before_execution(monkeypatch):
    inherited = merge_delegate_inherited_tool_settings({"web_search_mode": "ask", "use_web_search": True})
    child_config = {
        "use_web_search": True,
        "allowed_tool_ids": ["live_web_recon"],
        **inherited,
    }
    policies = tool_approval.invocation_policies(child_config)
    context = ToolInvocationContext(
        run_id="child-run",
        thread_id="thread-1",
        extensions={"tool_approval_policy": policies},
    )
    mode, pending = await tool_approval.check_tool_approval(
        "search_web",
        {"query": "latest benchmark"},
        context,
        invocation_id="call-delegated-1",
    )
    assert mode is ApprovalMode.ASK
    assert isinstance(pending, dict)
    assert pending.get("tool_name") == "search_web"