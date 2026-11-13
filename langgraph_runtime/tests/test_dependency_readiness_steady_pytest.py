from __future__ import annotations

import time

import pytest

from langgraph_runtime.dependencies import DependencyMonitor


def test_unavailable_configured_ignores_stale_snapshot_overlay_when_checks_are_recent(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_DEPENDENCY_JITTER_RATIO", "0")
    monkeypatch.delenv("LLM_API_URL", raising=False)
    monitor = DependencyMonitor()
    monitor._snapshots["mcp"] = {
        "state": "available",
        "capability_ids": ["search_documents"],
        "reason": None,
        "_last_success_epoch": time.time() - monitor.stale_after - 5,
        "_last_checked_epoch": time.time(),
    }
    overlay = monitor.snapshot()["mcp"]
    assert overlay["state"] == "available"
    assert monitor.unavailable_configured() is None


def test_stale_overlay_marks_degraded_when_checks_and_success_both_age_out(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_DEPENDENCY_JITTER_RATIO", "0")
    monkeypatch.delenv("LLM_API_URL", raising=False)
    monitor = DependencyMonitor()
    now = time.time()
    monitor._snapshots["mcp"] = {
        "state": "available",
        "capability_ids": ["search_documents"],
        "reason": None,
        "_last_success_epoch": now - monitor.stale_after - 5,
        "_last_checked_epoch": now - monitor.stale_after - 5,
    }
    assert monitor.snapshot()["mcp"]["state"] == "degraded"
    assert monitor.unavailable_configured() is None


@pytest.mark.asyncio
async def test_refresh_marks_degraded_on_probe_failure_while_retaining_capabilities(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_DEPENDENCY_JITTER_RATIO", "0")
    monkeypatch.delenv("LLM_API_URL", raising=False)
    monkeypatch.setenv("MCP_LOOPBACK_URL", "http://mcp.internal/mcp/")
    monitor = DependencyMonitor()
    monitor._snapshots["mcp"] = {
        "state": "available",
        "capability_ids": ["search_documents"],
        "_last_success_epoch": time.time(),
    }

    async def failing_probe(*_args, **_kwargs):
        return {"ok": False, "reason": "ConnectError"}

    monkeypatch.setattr("langgraph_runtime.dependencies.probe_mcp", failing_probe)
    await monitor.refresh()
    assert monitor._snapshots["mcp"]["state"] == "degraded"
    assert monitor._snapshots["mcp"]["capability_ids"] == ["search_documents"]
    failure = monitor.unavailable_configured()
    assert failure is not None
    assert failure["dependency"] == "mcp"
