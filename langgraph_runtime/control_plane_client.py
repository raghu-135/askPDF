"""HTTP client for LangGraph runtime → control plane internal orchestration."""

from __future__ import annotations

import os
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx
from langchain_core.runnables import RunnableConfig

from runtime_protocol.errors import RuntimeError as AgentRuntimeError


def control_plane_base_url() -> str:
    explicit = os.environ.get("CONTROL_PLANE_INTERNAL_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    loopback = os.environ.get("MCP_LOOPBACK_URL", "http://127.0.0.1:8000/internal/mcp/").strip()
    parsed = urlparse(loopback)
    if not parsed.scheme or not parsed.netloc:
        return "http://127.0.0.1:8000"
    return f"{parsed.scheme}://{parsed.netloc}"


def _execution_token(config: RunnableConfig | None) -> str:
    configurable = (config or {}).get("configurable") or {}
    token = configurable.get("mcp_execution_context_token")
    if not isinstance(token, str) or not token.strip():
        raise AgentRuntimeError(
            "delegate_context_missing",
            "Workflow delegation requires an MCP execution context token",
            retryable=False,
        )
    return token.strip()


async def delegate_sync(
    *,
    parent_run_id: str,
    thread_id: str,
    child_workflow_id: str,
    question: str,
    pre_fetch_bundle: Mapping[str, Any] | None,
    config: RunnableConfig,
    llm_model: str | None = None,
    use_web_search: bool | None = None,
    use_reranker: bool | None = None,
    system_role: str | None = None,
    custom_instructions: str | None = None,
    tool_instructions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run a child workflow synchronously via the control plane."""

    url = f"{control_plane_base_url()}/internal/agent-runs/delegate/sync"
    headers = {
        "Content-Type": "application/json",
        "X-AskPDF-Execution-Context": _execution_token(config),
    }
    payload: dict[str, Any] = {
        "parent_run_id": parent_run_id,
        "thread_id": thread_id,
        "child_workflow_id": child_workflow_id,
        "question": question,
        "pre_fetch_bundle": dict(pre_fetch_bundle or {}),
        "llm_model": llm_model,
        "use_web_search": use_web_search,
        "use_reranker": use_reranker,
        "system_role": system_role or "",
        "custom_instructions": custom_instructions or "",
        "tool_instructions": dict(tool_instructions or {}),
    }
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
    if response.status_code >= 400:
        detail = response.text[:2000]
        raise AgentRuntimeError(
            "delegate_sync_failed",
            f"Child workflow delegation failed ({response.status_code})",
            retryable=response.status_code >= 500,
            details={"body": detail, "child_workflow_id": child_workflow_id},
        )
    data = response.json()
    if not isinstance(data, dict):
        raise AgentRuntimeError("delegate_sync_invalid", "Delegate sync returned a non-object", retryable=False)
    return data
