"""Durable, framework-neutral observability for Hermes MCP execution."""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from app.agent_workflows.repository import AgentWorkflowRepository
from app.runtime.contracts import AgentRuntimeEvent


logger = logging.getLogger(__name__)


def _evidence_type(result: Any) -> str | None:
    artifacts = dict(getattr(result, "artifacts", None) or {})
    if artifacts.get("document_sources"):
        return "document"
    if artifacts.get("web_sources"):
        return "web"
    if artifacts.get("memory_refs"):
        return "memory"
    return None


async def persist_tool_audit(
    *,
    run_id: str,
    request_id: str,
    phase: str,
    tool_name: str,
    payload: Mapping[str, Any] | None = None,
    result: Any = None,
) -> None:
    if not run_id or not request_id:
        return
    data = {"tool_name": tool_name, "mcp_request_id": request_id, "source": "askpdf_mcp", **dict(payload or {})}
    if result is not None:
        sources = list(getattr(result, "sources", None) or [])
        data.update({
            "ok": bool(getattr(result, "ok", False)) and getattr(result, "error", None) is None,
            "result_count": len(sources),
            "evidence_type": _evidence_type(result),
            "warnings": [str(value) for value in getattr(result, "warnings", None) or []],
            "duration_ms": float(getattr(getattr(result, "metrics", None), "elapsed_ms", 0) or 0),
        })
        error = getattr(result, "error", None)
        if error is not None:
            data["error"] = {"code": str(getattr(error, "code", "tool_failed")), "retryable": bool(getattr(error, "retryable", True))}
    event = AgentRuntimeEvent(
        event_id=f"mcp:{request_id}:{phase}",
        run_id=run_id,
        sequence=int(time.time() * 1000) % 2_000_000_000,
        kind=f"tool.{phase}",
        payload=data,
    )
    try:
        await AgentWorkflowRepository().append_run_event(run_id, event)
    except Exception:
        logger.exception("Unable to persist MCP tool audit event run_id=%s tool=%s phase=%s", run_id, tool_name, phase)
