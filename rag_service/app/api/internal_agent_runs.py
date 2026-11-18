"""Internal control-plane endpoints for LangGraph child delegation."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Mapping, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.mcp.execution_context_token import ExecutionContextTokenError, decode_execution_context_token
from app.product_orchestration.service import AgentRunService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["internal-agent-runs"])


class DelegateSyncRequest(BaseModel):
    parent_run_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    child_workflow_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    pre_fetch_bundle: Dict[str, Any] = Field(default_factory=dict)
    llm_model: Optional[str] = None
    use_web_search: Optional[bool] = None
    use_browser_access: Optional[bool] = None
    browser_access_mode: Optional[str] = None
    use_reranker: Optional[bool] = None
    system_role: str = ""
    custom_instructions: str = ""
    tool_instructions: Dict[str, str] = Field(default_factory=dict)
    context_window: Optional[int] = Field(default=None, ge=256, le=2_000_000)
    context_budget: Optional[Dict[str, float]] = None
    inherited_tool_settings: Dict[str, Any] = Field(default_factory=dict)


def _validate_execution_context(
    token: str | None,
    *,
    parent_run_id: str,
    thread_id: str,
) -> Mapping[str, Any]:
    if not token or not token.strip():
        raise HTTPException(status_code=401, detail={"code": "mcp_execution_context_rejected"})
    try:
        context = decode_execution_context_token(token.strip())
    except ExecutionContextTokenError as exc:
        logger.warning("Delegate sync rejected token reason=%s", exc.reason)
        raise HTTPException(status_code=401, detail={"code": "mcp_execution_context_rejected"}) from exc
    if context.run_id != parent_run_id or context.thread_id != thread_id:
        raise HTTPException(status_code=401, detail={"code": "mcp_execution_context_rejected"})
    return {
        "embedding_model": context.embedding_model,
        "context_window": context.context_window,
        "use_web_search": context.use_web_search,
        "use_browser_access": context.use_browser_access,
        "use_reranker": context.use_reranker,
        "llm_model": (context.extensions or {}).get("llm_model"),
    }


@router.post("/agent-runs/delegate/sync")
async def delegate_sync(
    body: DelegateSyncRequest,
    execution_context: str | None = Header(default=None, alias="X-AskPDF-Execution-Context"),
):
    context_fields = _validate_execution_context(
        execution_context,
        parent_run_id=body.parent_run_id,
        thread_id=body.thread_id,
    )
    service = AgentRunService()
    try:
        return await service.run_delegate_sync(
            parent_run_id=body.parent_run_id,
            thread_id=body.thread_id,
            child_workflow_id=body.child_workflow_id,
            question=body.question,
            pre_fetch_bundle=body.pre_fetch_bundle,
            llm_model=body.llm_model or context_fields.get("llm_model"),
            embedding_model=str(context_fields.get("embedding_model") or ""),
            use_web_search=body.use_web_search if body.use_web_search is not None else bool(context_fields.get("use_web_search")),
            use_browser_access=(
                body.use_browser_access
                if body.use_browser_access is not None
                else bool(context_fields.get("use_browser_access"))
            ),
            browser_access_mode=body.browser_access_mode,
            use_reranker=body.use_reranker if body.use_reranker is not None else context_fields.get("use_reranker"),
            system_role=body.system_role,
            custom_instructions=body.custom_instructions,
            tool_instructions=body.tool_instructions,
            context_window=body.context_window or context_fields.get("context_window"),
            context_budget=body.context_budget,
            mcp_execution_context_token=execution_context.strip() if execution_context else None,
            inherited_tool_settings=body.inherited_tool_settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "delegate_invalid", "message": str(exc)}) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail={"code": "delegate_failed", "message": str(exc)}) from exc
