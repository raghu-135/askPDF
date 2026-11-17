"""Helpers when delegated child runs finish or fail after human review."""

from __future__ import annotations

from typing import Any

from app.db import AgentRunStatus


async def release_delegated_parent_if_stalled(repository: Any, child_run: Any) -> None:
    """Mark a delegated parent complete when a child terminal left the parent in-flight."""

    parent_run_id = str(getattr(child_run, "parent_run_id", "") or "").strip()
    if not parent_run_id:
        return
    parent_run = await repository.get_run(parent_run_id)
    if parent_run is None:
        return
    parent_status = str(getattr(parent_run, "status", "") or "")
    if parent_status != AgentRunStatus.RUNNING.value:
        return
    metrics = dict(getattr(parent_run, "metrics_json", None) or {})
    metrics["delegated_child_run_id"] = str(getattr(child_run, "id", "") or "")
    await repository.complete_run(
        parent_run_id,
        status=AgentRunStatus.COMPLETED.value,
        metrics_json=metrics,
    )