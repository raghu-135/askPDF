from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Optional

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from app.db.models_sqlmodel import AgentTaskArtifact
from app.db.repositories.canvas_repo_sqlmodel import CanvasRepository
from app.db.repositories.message_repo_sqlmodel import MessageRepository
from app.db.repositories.thread_repo_sqlmodel import ThreadRepository
from app.models.canvas import (
    RESEARCH_CANVAS_ARTIFACT_KIND,
    RESEARCH_CANVAS_MEDIA_TYPE,
    CanvasCreateRequest,
    parse_canvas_spec,
)
from app.services.content_store import get_content_store, thread_artifact_content_key
from app.services.task_artifact_service import MAX_SINGLE_ARTIFACT_BYTES
from app.time_utils import iso_utc_z


class CanvasNotFoundError(LookupError):
    pass


class CanvasValidationError(ValueError):
    pass


def _canvas_title(artifact: AgentTaskArtifact, spec: dict[str, Any] | None = None) -> str:
    summary = artifact.summary_json or {}
    title = str(summary.get("title") or "").strip()
    if title:
        return title
    if spec:
        return str(spec.get("title") or "").strip()
    return "Research canvas"


def _ownership_key(*, chat_turn_id: Optional[str]) -> str:
    if chat_turn_id:
        return f"chat_turn:{chat_turn_id}"
    return "thread"


async def serialize_canvas(artifact: AgentTaskArtifact, *, current: bool) -> dict[str, Any]:
    spec = parse_canvas_spec(json.loads((await get_content_store().read(artifact.object_key)).decode("utf-8")))
    payload = spec.model_dump(mode="json")
    return {
        "id": artifact.id,
        "thread_id": artifact.thread_id,
        "chat_turn_id": artifact.chat_turn_id,
        "title": _canvas_title(artifact, payload),
        "spec": payload,
        "supersedes_id": artifact.supersedes_id,
        "created_at": iso_utc_z(artifact.created_at),
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

    async def create(
        self,
        thread_id: str,
        request: CanvasCreateRequest,
        *,
        agent_run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        await self._require_thread(thread_id)
        spec = request.spec
        if request.idempotency_key:
            existing = await self._canvases.get_by_idempotency(thread_id, request.idempotency_key)
            if existing is not None:
                current_ids = {item.id for item in await self._canvases.list_for_thread(thread_id)}
                return await serialize_canvas(existing, current=existing.id in current_ids)

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

        body = json.dumps(spec.model_dump(mode="json"), separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(body) > MAX_SINGLE_ARTIFACT_BYTES:
            raise CanvasValidationError("canvas exceeds the 10 MB per-object limit")
        artifact_id = str(uuid.uuid4())
        object_key = thread_artifact_content_key(thread_id, artifact_id)
        digest = hashlib.sha256(body).hexdigest()
        store = get_content_store()
        await store.put(object_key, body, expected_sha256=digest)
        artifact = AgentTaskArtifact(
            id=artifact_id,
            task_id=None,
            agent_run_id=agent_run_id,
            thread_id=thread_id,
            chat_turn_id=chat_turn_id,
            idempotency_key=request.idempotency_key,
            ownership_key=_ownership_key(chat_turn_id=chat_turn_id),
            kind=RESEARCH_CANVAS_ARTIFACT_KIND,
            object_key=object_key,
            media_type=RESEARCH_CANVAS_MEDIA_TYPE,
            byte_size=len(body),
            sha256=digest,
            provenance_json={
                **({"chat_turn_id": chat_turn_id} if chat_turn_id else {}),
                **({"agent_run_id": agent_run_id} if agent_run_id else {}),
            },
            summary_json={"title": spec.title, "schema_version": spec.schema_version},
            supersedes_id=request.supersedes_id,
            retention_until=None,
        )
        try:
            saved = await self._canvases.create(artifact)
        except IntegrityError as exc:
            await store.delete(object_key)
            if request.idempotency_key:
                existing = await self._canvases.get_by_idempotency(thread_id, request.idempotency_key)
                if existing is not None:
                    return await serialize_canvas(existing, current=True)
            raise CanvasValidationError("canvas could not be stored") from exc
        except Exception:
            await store.delete(object_key)
            raise
        return await serialize_canvas(saved, current=True)

    async def get(self, thread_id: str, canvas_id: str) -> dict[str, Any]:
        await self._require_thread(thread_id)
        canvas = await self._canvases.get(canvas_id)
        if canvas is None or canvas.thread_id != thread_id:
            raise CanvasNotFoundError("Canvas not found")
        current_ids = {item.id for item in await self._canvases.list_for_thread(thread_id)}
        try:
            return await serialize_canvas(canvas, current=canvas.id in current_ids)
        except FileNotFoundError as exc:
            raise CanvasNotFoundError("Canvas content is unavailable") from exc

    async def list_for_thread(self, thread_id: str, *, current_only: bool = True) -> list[dict[str, Any]]:
        await self._require_thread(thread_id)
        rows = await self._canvases.list_for_thread(thread_id, current_only=current_only)
        current_ids = {item.id for item in rows} if current_only else {
            item.id for item in await self._canvases.list_for_thread(thread_id)
        }
        payloads = []
        for row in rows:
            try:
                payloads.append(await serialize_canvas(row, current=row.id in current_ids))
            except FileNotFoundError:
                continue
        return payloads

    async def refs_by_turn(self, thread_id: str) -> dict[str, dict[str, str]]:
        refs: dict[str, dict[str, str]] = {}
        canvases = await self._canvases.list_for_thread(thread_id)
        run_ids = [item.agent_run_id for item in canvases if item.agent_run_id and not item.chat_turn_id]
        turns_by_run: dict[str, str] = {}
        for agent_run_id in run_ids:
            turn = await self._messages.get_turn_by_agent_run_id(thread_id, agent_run_id)
            if turn is not None:
                turns_by_run[agent_run_id] = turn.id
        for item in canvases:
            turn_id = item.chat_turn_id or turns_by_run.get(item.agent_run_id or "")
            if turn_id and turn_id not in refs:
                refs[turn_id] = {"id": item.id, "title": _canvas_title(item)}
        return refs


def validate_canvas_payload(payload: Any) -> CanvasCreateRequest:
    try:
        if isinstance(payload, dict) and "spec" not in payload:
            return CanvasCreateRequest(spec=parse_canvas_spec(payload))
        return CanvasCreateRequest.model_validate(payload)
    except ValidationError as exc:
        raise CanvasValidationError(str(exc)) from exc
