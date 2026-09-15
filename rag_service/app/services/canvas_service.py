from __future__ import annotations

from typing import Any, Optional

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from app.db.models_sqlmodel import ThreadCanvas
from app.db.repositories.canvas_repo_sqlmodel import CanvasRepository
from app.db.repositories.message_repo_sqlmodel import MessageRepository
from app.db.repositories.thread_repo_sqlmodel import ThreadRepository
from app.models.canvas import CanvasCreateRequest, parse_canvas_spec
from app.time_utils import iso_utc_z


class CanvasNotFoundError(LookupError):
    pass


class CanvasValidationError(ValueError):
    pass


def serialize_canvas(canvas: ThreadCanvas, *, current: bool) -> dict[str, Any]:
    return {
        "id": canvas.id,
        "thread_id": canvas.thread_id,
        "chat_turn_id": canvas.chat_turn_id,
        "title": canvas.title,
        "spec": canvas.spec_json,
        "supersedes_id": canvas.supersedes_id,
        "created_at": iso_utc_z(canvas.created_at),
        "current": current,
    }


class CanvasService:
    def __init__(
        self,
        *,
        canvases: CanvasRepository | None = None,
        threads: ThreadRepository | None = None,
        messages: MessageRepository | None = None,
    ):
        self._canvases = canvases or CanvasRepository()
        self._threads = threads or ThreadRepository()
        self._messages = messages or MessageRepository()

    async def _require_thread(self, thread_id: str) -> None:
        thread = await self._threads.get(thread_id)
        if thread is None:
            raise CanvasNotFoundError("Thread not found")

    async def create(self, thread_id: str, request: CanvasCreateRequest) -> dict[str, Any]:
        await self._require_thread(thread_id)
        spec = request.spec
        if request.idempotency_key:
            existing = await self._canvases.get_by_idempotency(thread_id, request.idempotency_key)
            if existing is not None:
                current_ids = {item.id for item in await self._canvases.list_for_thread(thread_id)}
                return serialize_canvas(existing, current=existing.id in current_ids)

        chat_turn_id = request.chat_turn_id
        if chat_turn_id:
            turn = await self._messages.get_turn(chat_turn_id)
            if turn is None or turn.thread_id != thread_id:
                raise CanvasValidationError("chat_turn_id must belong to this thread")

        if request.supersedes_id:
            previous = await self._canvases.get(request.supersedes_id)
            if previous is None or previous.thread_id != thread_id:
                raise CanvasValidationError("supersedes_id must belong to this thread")
            successors = [
                item
                for item in await self._canvases.list_for_thread(thread_id, current_only=False)
                if item.supersedes_id == request.supersedes_id
            ]
            if successors:
                raise CanvasValidationError("that canvas already has a successor")

        canvas = ThreadCanvas(
            thread_id=thread_id,
            chat_turn_id=chat_turn_id,
            title=spec.title,
            spec_json=spec.model_dump(mode="json"),
            supersedes_id=request.supersedes_id,
            idempotency_key=request.idempotency_key,
        )
        try:
            saved = await self._canvases.create(canvas)
        except IntegrityError as exc:
            if request.idempotency_key:
                existing = await self._canvases.get_by_idempotency(thread_id, request.idempotency_key)
                if existing is not None:
                    return serialize_canvas(existing, current=True)
            raise CanvasValidationError("canvas could not be stored") from exc
        return serialize_canvas(saved, current=True)

    async def get(self, thread_id: str, canvas_id: str) -> dict[str, Any]:
        await self._require_thread(thread_id)
        canvas = await self._canvases.get(canvas_id)
        if canvas is None or canvas.thread_id != thread_id:
            raise CanvasNotFoundError("Canvas not found")
        current_ids = {item.id for item in await self._canvases.list_for_thread(thread_id)}
        return serialize_canvas(canvas, current=canvas.id in current_ids)

    async def list_for_thread(self, thread_id: str, *, current_only: bool = True) -> list[dict[str, Any]]:
        await self._require_thread(thread_id)
        rows = await self._canvases.list_for_thread(thread_id, current_only=current_only)
        current_ids = {item.id for item in rows} if current_only else {
            item.id for item in await self._canvases.list_for_thread(thread_id)
        }
        return [serialize_canvas(row, current=row.id in current_ids) for row in rows]

    async def refs_by_turn(self, thread_id: str) -> dict[str, dict[str, str]]:
        refs: dict[str, dict[str, str]] = {}
        for item in await self._canvases.list_for_thread(thread_id):
            if item.chat_turn_id and item.chat_turn_id not in refs:
                refs[item.chat_turn_id] = {"id": item.id, "title": item.title}
        return refs


def validate_canvas_payload(payload: Any) -> CanvasCreateRequest:
    try:
        if isinstance(payload, dict) and "spec" not in payload:
            return CanvasCreateRequest(spec=parse_canvas_spec(payload))
        return CanvasCreateRequest.model_validate(payload)
    except ValidationError as exc:
        raise CanvasValidationError(str(exc)) from exc
