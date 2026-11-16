from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any, Optional

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from app.db.models_sqlmodel import AgentTaskArtifact
from app.db.repositories.canvas_repo_sqlmodel import CanvasRepository
from app.db.repositories.message_repo_sqlmodel import MessageRepository
from app.db.repositories.thread_repo_sqlmodel import ThreadRepository
from app.db import get_file_parsed_sentences
from app.models.canvas import (
    RESEARCH_CANVAS_ARTIFACT_KIND,
    RESEARCH_CANVAS_MEDIA_TYPE,
    CanvasAdmissionError,
    CanvasCreateRequest,
    QuoteBlock,
    iter_canvas_quote_file_hashes,
    parse_canvas_spec,
    resolve_quote_blocks,
)
from app.services.content_store import get_content_store, thread_artifact_content_key
from app.services.task_artifact_service import MAX_SINGLE_ARTIFACT_BYTES
from app.time_utils import iso_utc_z
from runtime_protocol.canvas_reference import extract_canvas_references
from runtime_protocol.canvas_revision_context import build_revision_snapshot


class CanvasNotFoundError(LookupError):
    pass


class CanvasValidationError(ValueError):
    pass


_CANVAS_ARTIFACT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_FILE_HASH_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


def is_canvas_artifact_id(value: str) -> bool:
    return bool(_CANVAS_ARTIFACT_ID_RE.fullmatch(str(value or "").strip()))


async def resolve_canvas_lineage_head(
    canvases: CanvasRepository,
    canvas_id: str,
    *,
    max_hops: int = 32,
) -> str:
    """Follow supersede links forward until there is no newer successor."""

    current_id = str(canvas_id or "").strip()
    if not current_id:
        return current_id
    visited: set[str] = set()
    for _ in range(max_hops):
        if current_id in visited:
            break
        visited.add(current_id)
        successor = await canvases.get_successor(current_id)
        if successor is None:
            break
        current_id = successor.id
    return current_id


async def load_quoted_canvas_revisions(thread_id: str, text: str) -> list[dict[str, Any]]:
    """Load full revision baselines for canvas ids quoted in the user message."""

    refs = extract_canvas_references(text)
    if not refs:
        return []
    service = CanvasService()
    seen: set[str] = set()
    revisions: list[dict[str, Any]] = []
    for ref in refs:
        normalized = str(ref or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            record = await service.get(thread_id, ref, resolve_current=True)
        except CanvasNotFoundError:
            continue
        spec = record.get("spec") if isinstance(record.get("spec"), dict) else {}
        quoted_id = str(record.get("requested_id") or record.get("id") or ref)
        lineage_head_id = str(record.get("id") or ref)
        revisions.append(
            build_revision_snapshot(
                canvas_id=quoted_id,
                lineage_head_id=lineage_head_id,
                title=str(record.get("title") or ""),
                spec=spec,
            )
        )
    return revisions


async def resolve_quoted_canvas_supersedes_id(
    canvases: CanvasRepository,
    thread_id: str,
    text: str,
) -> tuple[str | None, list[str]]:
    current_rows = await canvases.list_for_thread(thread_id, current_only=True)
    current_ids = {row.id for row in current_rows}
    refs = extract_canvas_references(text)

    async def head_resolver(canvas_id: str) -> str | None:
        artifact = await canvases.get(canvas_id)
        if artifact is None or artifact.thread_id != thread_id:
            return None
        return await resolve_canvas_lineage_head(canvases, canvas_id)

    picked: str | None = None
    warnings: list[str] = []
    for canvas_id in reversed(refs):
        head = await head_resolver(canvas_id)
        if head is None:
            continue
        if head in current_ids:
            picked = head
            if head != canvas_id:
                warnings.append("quoted_canvas_resolved_to_current_head")
            break
    if not picked and refs:
        warnings.append("quoted_canvas_not_current")
    return picked, warnings


async def resolve_publish_supersedes_id(
    canvases: CanvasRepository,
    thread_id: str,
    supersedes_id: str | None,
) -> tuple[str | None, list[str]]:
    """Drop mistaken supersedes_id values; keep strict checks for cross-thread canvas ids."""

    warnings: list[str] = []
    raw = str(supersedes_id or "").strip()
    if not raw:
        return None, warnings
    if _FILE_HASH_RE.fullmatch(raw):
        warnings.append("supersedes_id_ignored_file_hash")
        return None, warnings
    if not is_canvas_artifact_id(raw):
        warnings.append("supersedes_id_ignored_invalid_format")
        return None, warnings
    previous = await canvases.get(raw)
    if previous is None or previous.thread_id != thread_id:
        warnings.append("supersedes_id_ignored_not_on_thread")
        return None, warnings
    current_rows = await canvases.list_for_thread(thread_id, current_only=True)
    current_ids = {row.id for row in current_rows}
    if raw not in current_ids:
        raise CanvasValidationError("supersedes_id must reference a current canvas on this thread")
    return raw, warnings


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
        task_id: Optional[str] = None,
    ) -> dict[str, Any]:
        await self._require_thread(thread_id)
        spec = request.spec
        quote_hashes = sorted(set(iter_canvas_quote_file_hashes(spec)))
        if quote_hashes:
            sentences_by_file = {
                file_hash: await get_file_parsed_sentences(file_hash)
                for file_hash in quote_hashes
            }
            try:
                spec = resolve_quote_blocks(spec, sentences_by_file=sentences_by_file)
            except CanvasAdmissionError as exc:
                raise CanvasValidationError(str(exc)) from exc
            for block in (
                item
                for section in spec.sections
                for item in section.blocks
                if isinstance(item, QuoteBlock)
            ):
                if not block.text:
                    raise CanvasValidationError(
                        f"quote sentence_id {block.sentence_id} could not be resolved for {block.file_hash}"
                    )
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
            task_id=task_id,
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

    async def get(
        self,
        thread_id: str,
        canvas_id: str,
        *,
        resolve_current: bool = True,
    ) -> dict[str, Any]:
        await self._require_thread(thread_id)
        canvas = await self._canvases.get(canvas_id)
        if canvas is None or canvas.thread_id != thread_id:
            raise CanvasNotFoundError("Canvas not found")
        current_ids = {item.id for item in await self._canvases.list_for_thread(thread_id)}
        requested_id = canvas.id
        head_id = await resolve_canvas_lineage_head(self._canvases, requested_id)
        if resolve_current and head_id != requested_id:
            head = await self._canvases.get(head_id)
            if head is not None and head.thread_id == thread_id:
                canvas = head
        try:
            payload = await serialize_canvas(canvas, current=canvas.id in current_ids)
            if requested_id != canvas.id:
                payload["requested_id"] = requested_id
            return payload
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

    async def link_published_canvases_to_turn(
        self,
        thread_id: str,
        *,
        agent_run_id: str,
        chat_turn_id: str,
    ) -> int:
        if not str(agent_run_id or "").strip() or not str(chat_turn_id or "").strip():
            return 0
        return await self._canvases.assign_chat_turn_for_agent_run(
            thread_id,
            agent_run_id=str(agent_run_id).strip(),
            chat_turn_id=str(chat_turn_id).strip(),
        )

    async def refs_by_turn(self, thread_id: str) -> dict[str, dict[str, str]]:
        refs: dict[str, dict[str, str]] = {}
        canvases = await self._canvases.list_for_thread(thread_id, current_only=False)
        run_ids = {str(item.agent_run_id) for item in canvases if item.agent_run_id}
        turns_by_run: dict[str, str] = {}
        for agent_run_id in run_ids:
            turn = await self._messages.get_turn_by_agent_run_id(thread_id, agent_run_id)
            if turn is not None:
                turns_by_run[str(agent_run_id)] = turn.id
        for item in sorted(canvases, key=lambda row: (row.created_at or row.id, row.id)):
            turn_id = item.chat_turn_id or turns_by_run.get(str(item.agent_run_id or ""))
            if not turn_id:
                continue
            title = _canvas_title(item)
            entry = {"id": str(item.id), "title": title}
            if turn_id in refs:
                refs[turn_id] = entry
            else:
                refs[turn_id] = entry
        return refs


def validate_canvas_payload(payload: Any) -> CanvasCreateRequest:
    try:
        if isinstance(payload, dict) and "spec" not in payload:
            return CanvasCreateRequest(spec=parse_canvas_spec(payload))
        return CanvasCreateRequest.model_validate(payload)
    except ValidationError as exc:
        raise CanvasValidationError(str(exc)) from exc
