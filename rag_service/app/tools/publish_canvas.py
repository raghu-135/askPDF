"""Framework-neutral publish_canvas implementation."""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from app.agent.tool_contract import make_tool_error_result, make_tool_result, tool_started
from app.db import get_thread_files
from app.models.canvas import (
    CanvasAdmissionError,
    CanvasCreateRequest,
    CanvasSpec,
    admit_canvas_spec,
    parse_canvas_spec,
)
from app.services.canvas_service import CanvasService, CanvasValidationError
from app.tools.context import ToolInvocationContext


class PublishCanvasRequest(BaseModel):
    spec: CanvasSpec
    supersedes_id: Optional[str] = Field(default=None, max_length=64)
    idempotency_key: Optional[str] = Field(default=None, min_length=8, max_length=128)


def _idempotency_key(spec: CanvasSpec, requested: str | None) -> str:
    if requested:
        return requested
    body = json.dumps(spec.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
    return "publish_canvas:" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:96]


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
    files = await get_thread_files(context.thread_id)
    thread_file_hashes = {str(getattr(item, "file_hash", "") or "") for item in files} - {""}
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
    try:
        created = await CanvasService().create(
            context.thread_id,
            CanvasCreateRequest(
                spec=spec,
                supersedes_id=request.supersedes_id,
                idempotency_key=_idempotency_key(spec, request.idempotency_key),
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
    )
