"""Framework-neutral publish_canvas implementation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.agent.tool_contract import make_tool_error_result, make_tool_result, tool_started
from app.db import get_thread_files
from app.models.canvas import (
    CanvasAdmissionError,
    CanvasCreateRequest,
    CanvasSpec,
    admit_canvas_spec,
    parse_canvas_spec,
)
from app.models.canvas_draft import CanvasCompileError, CanvasPublishDraft
from app.services.canvas_compiler import compile_canvas_draft
from app.services.canvas_service import CanvasService, CanvasValidationError, resolve_publish_supersedes_id
from app.tools.context import ToolInvocationContext


class PublishCanvasRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layout: Optional[str] = None
    title: Optional[str] = Field(default=None, max_length=160)
    summary: Optional[str] = Field(default=None, max_length=400)
    claims: Optional[list[dict[str, Any]]] = None
    spec: Optional[Any] = None
    supersedes_id: Optional[str] = Field(default=None, max_length=64)
    idempotency_key: Optional[str] = Field(default=None, min_length=8, max_length=128)

    @model_validator(mode="after")
    def validate_agent_or_spec(self) -> "PublishCanvasRequest":
        if self.spec is not None:
            return self
        if not self.layout or not self.title or not self.claims:
            raise ValueError("publish_canvas requires layout, title, and claims (or a full spec for API callers)")
        return self


def _publish_idempotency_key(
    *,
    requested: str | None,
    agent_run_id: str | None,
    supersedes_id: str | None,
    spec: CanvasSpec,
) -> str:
    if requested:
        return requested
    if agent_run_id:
        suffix = supersedes_id or "create"
        return f"publish_canvas:{agent_run_id}:{suffix}"
    body = json.dumps(spec.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
    return "publish_canvas:" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:96]


async def _thread_file_hashes(thread_id: str) -> set[str]:
    files = await get_thread_files(thread_id)
    return {str(getattr(item, "file_hash", "") or "") for item in files} - {""}


async def invoke_publish_canvas(request: PublishCanvasRequest, context: ToolInvocationContext):
    started = tool_started()
    if not context.thread_id:
        return make_tool_result(
            tool_name="publish_canvas",
            content="No thread context found.",
            context=context,
            started=started,
            ok=False,
            warnings=["missing_thread_id"],
        )

    thread_file_hashes = await _thread_file_hashes(context.thread_id)
    spec: CanvasSpec | None = None

    if request.spec is not None:
        try:
            spec = request.spec if isinstance(request.spec, CanvasSpec) else parse_canvas_spec(request.spec)
        except (ValidationError, ValueError) as exc:
            return make_tool_error_result(
                tool_name="publish_canvas",
                error=exc,
                context=context,
                started=started,
                code="publish_canvas_invalid_component",
                user_message=f"Invalid canvas component: {exc}",
            )
    else:
        try:
            draft = CanvasPublishDraft.model_validate({
                "layout": request.layout,
                "title": request.title,
                "summary": request.summary,
                "claims": request.claims,
            })
        except ValidationError as exc:
            return make_tool_error_result(
                tool_name="publish_canvas",
                error=exc,
                context=context,
                started=started,
                code="publish_canvas_invalid_draft",
                user_message=f"Invalid canvas draft: {exc}",
            )
        try:
            spec = await compile_canvas_draft(draft, thread_file_hashes=thread_file_hashes)
        except CanvasCompileError as exc:
            return make_tool_error_result(
                tool_name="publish_canvas",
                error=exc,
                context=context,
                started=started,
                code=f"publish_canvas_{exc.code}",
                user_message=str(exc),
            )

    assert spec is not None
    try:
        admit_canvas_spec(spec, thread_file_hashes=thread_file_hashes)
    except CanvasAdmissionError as exc:
        return make_tool_error_result(
            tool_name="publish_canvas",
            error=exc,
            context=context,
            started=started,
            code=f"publish_canvas_{exc.code}",
            user_message=str(exc),
        )

    service = CanvasService()
    supersedes_warnings: list[str] = []
    resolved_supersedes_id = request.supersedes_id
    try:
        resolved_supersedes_id, supersedes_warnings = await resolve_publish_supersedes_id(
            service._canvases,
            context.thread_id,
            request.supersedes_id,
        )
    except CanvasValidationError as exc:
        return make_tool_error_result(
            tool_name="publish_canvas",
            error=exc,
            context=context,
            started=started,
            code="publish_canvas_persist_failed",
            user_message=str(exc),
        )
    try:
        created = await service.create(
            context.thread_id,
            CanvasCreateRequest(
                spec=spec,
                supersedes_id=resolved_supersedes_id,
                idempotency_key=_publish_idempotency_key(
                    requested=request.idempotency_key,
                    agent_run_id=context.run_id,
                    supersedes_id=resolved_supersedes_id,
                    spec=spec,
                ),
            ),
            agent_run_id=context.run_id,
        )
    except CanvasValidationError as exc:
        return make_tool_error_result(
            tool_name="publish_canvas",
            error=exc,
            context=context,
            started=started,
            code="publish_canvas_persist_failed",
            user_message=str(exc),
        )
    title = created.get("title") or spec.title
    return make_tool_result(
        tool_name="publish_canvas",
        content=f"Published research canvas {created['id']}: {title}",
        context=context,
        started=started,
        artifacts={"canvas": created},
        warnings=supersedes_warnings,
    )
