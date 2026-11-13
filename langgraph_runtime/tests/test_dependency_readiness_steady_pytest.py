from __future__ import annotations

import time

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
