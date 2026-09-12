"""Framework-neutral MCP handlers used by the memory curator."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.tool_contract import ToolResult, make_tool_error_result, tool_started
from app.models.memory_tools import (
    MemoryChangeIntent,
    MemoryGetInput,
    MemoryPrepareChangeInput,
    MemorySearchInput,
    MemoryToolContext,
)
from app.services.memory_tool_service import get_memory_tool, prepare_memory_change, search_memory_tool
from app.services.web_search_service import DEFAULT_WEB_SEARCH_RESULTS, WEB_SEARCH_CAPABILITY, search_internet
from app.tools.context import ToolInvocationContext
from app.tools.contracts import InternetSearchRequest


def _manager_context(context: ToolInvocationContext) -> MemoryToolContext:
    raw = (context.extensions or {}).get("memory_tool_context")
    if not isinstance(raw, dict):
        raise ValueError("memory manager context is required")
    return MemoryToolContext.model_validate(raw)


def _result(tool: str, payload: Any, *, warnings: list[str] | None = None) -> ToolResult:
    text = json.dumps(payload, ensure_ascii=True)
    return ToolResult(content=text, artifacts={"memory_manager_tool": tool}, warnings=warnings or [])


def _sanitize_prepare_intents(request: MemoryPrepareChangeInput, manager: dict[str, Any]) -> tuple[list[MemoryChangeIntent], list[str]]:
    """Apply curator policy at the MCP boundary for every caller path."""
    scope_ids = {str(value) for value in manager.get("scope_ids") or ()}
    conversation_review = manager.get("curator_mode") == "conversation_review"
    intents: list[MemoryChangeIntent] = []
    warnings: list[str] = []
    for intent in request.intents:
        targets = [target for target in (intent.override_target_ids or []) if target not in scope_ids]
        if len(targets) != len(intent.override_target_ids or []):
            warnings.append("Ignored a workspace scope identifier used as an override memory ID.")
        normalized = intent.model_copy(update={"override_target_ids": targets})
        if conversation_review:
            if normalized.action == "noop":
                intents.append(normalized)
                continue
            if normalized.action != "create":
                raise ValueError("Conversation review can only create new Thread memories")
            if normalized.scope_type not in {None, "thread"}:
                raise ValueError("Conversation review cannot create Project or Global memory")
            if normalized.memory_id or normalized.target_scope_type:
                raise ValueError("Conversation review cannot modify or move existing memory")
            normalized = normalized.model_copy(update={"scope_type": "thread"})
        intents.append(normalized)
    return intents, warnings


async def memory_search(request: MemorySearchInput, context: ToolInvocationContext, *, services: Any = None) -> ToolResult:
    payload = await search_memory_tool(_manager_context(context), request)
    return _result("memory_search", payload)


async def memory_get(request: MemoryGetInput, context: ToolInvocationContext, *, services: Any = None) -> ToolResult:
    payload = await get_memory_tool(_manager_context(context), request)
    return _result("memory_get", payload)


async def memory_prepare_change(request: MemoryPrepareChangeInput, context: ToolInvocationContext, *, services: Any = None) -> ToolResult:
    manager = context.extensions or {}
    started = tool_started()
    try:
        intents, warnings = _sanitize_prepare_intents(request, manager)
    except ValueError as exc:
        return make_tool_error_result(
            tool_name="memory_prepare_change",
            error=exc,
            context=context,
            started=started,
            user_message=str(exc),
        )
    payload = await prepare_memory_change(_manager_context(context), MemoryPrepareChangeInput(intents=intents))
    payload["canonical_intents"] = [intent.model_dump(mode="json") for intent in intents]
    if warnings:
        payload["input_warnings"] = warnings
    return _result("memory_prepare_change", payload, warnings=warnings)


async def internet_search(request: InternetSearchRequest, context: ToolInvocationContext, *, services: Any = None) -> ToolResult:
    manager = context.extensions or {}
    capabilities = set(manager.get("capabilities") or context.caller_capabilities)
    if WEB_SEARCH_CAPABILITY not in capabilities:
        return _result("internet_search", {"status": "denied", "message": "Web search capability is unavailable."})
    mode = manager.get("web_search_mode", "off")
    decision = manager.get("web_search_decision") or {}
    try:
        web_call_count = int(manager.get("web_call_count") or 0)
        web_call_limit = int(manager.get("web_call_limit") or 2)
    except (TypeError, ValueError):
        web_call_count, web_call_limit = 0, 2
    if web_call_count >= web_call_limit:
        return _result("internet_search", {"status": "limit_reached", "message": "The web-search limit for this curator run has been reached."})
    if mode == "off":
        return _result("internet_search", {"status": "disabled", "message": "Internet search is off."})
    if mode == "ask" and not (decision.get("approved") and decision.get("query") == request.query):
        if decision.get("query") == request.query and decision.get("approved") is False:
            return _result("internet_search", {"status": "denied", "message": "The user declined this search."})
        return _result("internet_search", {"status": "approval_required", "query": request.query, "reason": request.reason})
    result = await search_internet(request.query, max_results=DEFAULT_WEB_SEARCH_RESULTS)
    return _result("internet_search", result)
