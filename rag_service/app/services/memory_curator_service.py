"""Ephemeral LLM-assisted curation over canonical durable memory."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import and_, delete, func, or_
from sqlalchemy.future import select

from app.agent_workflows.runtime_invocation import safe_json_object
from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import ChatTurnStatus, MemoryScopeType, MemoryType, MemoryVisibility
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import ChatTurn, Memory, MemoryEvent, Project, Thread
from app.db.repositories.memory_repo_sqlmodel import MemoryRepository
from app.db.project_activity import touch_project_activity
from app.db.vector import get_vector_db
from app.models.llm_server_client import check_chat_model_ready, get_embedding_model, get_llm
from app.models.requests import (
    MemoryCuratorApplyRequest,
    MemoryCuratorContext,
    MemoryCuratorOperation,
    MemoryCuratorRespondRequest,
)
from app.models.retry import invoke_with_retry
from app.services.embedding_model_service import (
    require_embedding_model_ready,
    resolve_scope_embedding_model,
)
from app.services.memory_policy import (
    LOCAL_USER_MEMORY_SCOPE_ID,
    normalize_project_memory_settings,
    normalize_thread_memory_settings,
)
from app.services.memory_service import (
    _rank_fuse_memory_hits,
    index_memory_record,
    memory_content_hash,
)
from app.time_utils import iso_utc_z, utc_now


logger = logging.getLogger(__name__)

MAX_CURATOR_MEMORIES = 40
MAX_REVIEW_TURNS = 20
VALID_MEMORY_TYPES = {item.value for item in MemoryType}
VALID_ACTIONS = {"create", "update", "delete", "noop"}


class MemoryCuratorError(ValueError):
    code = "memory_curator_error"


class MemoryCuratorNotFoundError(MemoryCuratorError):
    code = "memory_not_found"


class MemoryCuratorModelUnavailableError(MemoryCuratorError):
    code = "llm_model_unavailable"


class MemoryChangedError(MemoryCuratorError):
    code = "memory_changed"

    def __init__(self, memories: Sequence[Memory]):
        super().__init__("Memory changed after the curator proposal was prepared")
        self.memories = list(memories)


def memory_payload(memory: Memory) -> Dict[str, Any]:
    return {
        "id": memory.id,
        "scope_type": memory.scope_type,
        "scope_id": memory.scope_id,
        "memory_type": memory.memory_type,
        "content": memory.content,
        "summary": memory.summary,
        "embedding_model": memory.embedding_model,
        "content_hash": memory.content_hash,
        "index_status": memory.index_status,
        "index_attempts": memory.index_attempts,
        "indexed_at": iso_utc_z(memory.indexed_at) if memory.indexed_at else None,
        "index_error": memory.index_error,
        "source_refs_json": memory.source_refs_json or {},
        "confidence": memory.confidence,
        "status": memory.status,
        "visibility": memory.visibility,
        "created_by": memory.created_by,
        "expires_at": iso_utc_z(memory.expires_at) if memory.expires_at else None,
        "fork_origin_json": memory.fork_origin_json,
        "created_at": iso_utc_z(memory.created_at) if memory.created_at else None,
        "updated_at": iso_utc_z(memory.updated_at) if memory.updated_at else None,
    }


async def _resolve_visible_scopes(
    context: MemoryCuratorContext,
) -> tuple[List[Dict[str, str]], Thread | None, Project | None]:
    selected_type = context.selected_scope_type
    selected_id = (
        LOCAL_USER_MEMORY_SCOPE_ID
        if selected_type == MemoryScopeType.USER.value
        else context.selected_scope_id.strip()
    )
    async with async_session_maker() as session:
        thread = await session.get(Thread, context.thread_id) if context.thread_id else None
        if context.thread_id and thread is None:
            raise MemoryCuratorNotFoundError("Thread not found")
        project_id = context.project_id or (thread.project_id if thread else None)
        project = await session.get(Project, project_id) if project_id else None
        if project_id and project is None:
            raise MemoryCuratorNotFoundError("Project not found")
        if thread and project and thread.project_id != project.id:
            raise MemoryCuratorError("Thread does not belong to the selected project")

    scopes: List[Dict[str, str]] = []
    if thread is not None:
        scopes.append({"scope_type": MemoryScopeType.THREAD.value, "scope_id": thread.id})
    if project is not None:
        scopes.append({"scope_type": MemoryScopeType.PROJECT.value, "scope_id": project.id})
    scopes.append({
        "scope_type": MemoryScopeType.USER.value,
        "scope_id": LOCAL_USER_MEMORY_SCOPE_ID,
    })
    deduped = {
        (scope["scope_type"], scope["scope_id"]): scope
        for scope in scopes
    }
    if (selected_type, selected_id) not in deduped:
        raise MemoryCuratorError("Selected memory scope is not available in this workspace")
    return list(deduped.values()), thread, project


def _turn_text(turn: ChatTurn) -> Dict[str, Any]:
    payload = turn.payload if isinstance(turn.payload, dict) else {}
    return {
        "id": turn.id,
        "question": str(payload.get("question") or ""),
        "answer": str(payload.get("answer") or ""),
        "created_at": iso_utc_z(turn.created_at),
    }


async def _review_batch(thread: Thread | None) -> Dict[str, Any] | None:
    if thread is None:
        return None
    metadata = thread.thread_metadata if isinstance(thread.thread_metadata, dict) else {}
    curator_meta = metadata.get("memory_curator") if isinstance(metadata.get("memory_curator"), dict) else {}
    cursor_at_raw = curator_meta.get("reviewed_through_created_at")
    cursor_id = str(curator_meta.get("reviewed_through_turn_id") or "")
    cursor_at = None
    if cursor_at_raw:
        try:
            cursor_at = datetime.fromisoformat(str(cursor_at_raw).replace("Z", "+00:00"))
        except ValueError:
            cursor_at = None

    base_filters = [
        ChatTurn.thread_id == thread.id,
        ChatTurn.status == ChatTurnStatus.COMPLETED.value,
        ChatTurn.payload["question"].astext.isnot(None),
        ChatTurn.payload["question"].astext != "",
        ChatTurn.payload["answer"].astext.isnot(None),
        ChatTurn.payload["answer"].astext != "",
    ]
    async with async_session_maker() as session:
        if cursor_at is None:
            result = await session.execute(
                select(ChatTurn)
                .where(*base_filters)
                .order_by(ChatTurn.created_at.desc(), ChatTurn.id.desc())
                .limit(MAX_REVIEW_TURNS)
            )
            turns = list(reversed(result.scalars().all()))
            remaining = 0
        else:
            after_cursor = or_(
                ChatTurn.created_at > cursor_at,
                and_(ChatTurn.created_at == cursor_at, ChatTurn.id > cursor_id),
            )
            result = await session.execute(
                select(ChatTurn)
                .where(*base_filters, after_cursor)
                .order_by(ChatTurn.created_at.asc(), ChatTurn.id.asc())
                .limit(MAX_REVIEW_TURNS)
            )
            turns = list(result.scalars().all())
            if turns:
                last = turns[-1]
                remaining = int((await session.execute(
                    select(func.count(ChatTurn.id)).where(
                        *base_filters,
                        or_(
                            ChatTurn.created_at > last.created_at,
                            and_(ChatTurn.created_at == last.created_at, ChatTurn.id > last.id),
                        ),
                    )
                )).scalar() or 0)
            else:
                remaining = 0
    reviewed_through = turns[-1] if turns else None
    return {
        "turns": [_turn_text(turn) for turn in turns],
        "reviewed_count": len(turns),
        "remaining_count": remaining,
        "cursor": (
            {
                "thread_id": thread.id,
                "reviewed_through_turn_id": reviewed_through.id,
                "reviewed_through_created_at": iso_utc_z(reviewed_through.created_at),
            }
            if reviewed_through else None
        ),
    }


async def _recent_scope_memories(scopes: Sequence[Dict[str, str]]) -> List[Memory]:
    clauses = [
        and_(Memory.scope_type == scope["scope_type"], Memory.scope_id == scope["scope_id"])
        for scope in scopes
    ]
    if not clauses:
        return []
    async with async_session_maker() as session:
        result = await session.execute(
            select(Memory)
            .where(
                or_(*clauses),
                Memory.status == "active",
                or_(Memory.expires_at.is_(None), Memory.expires_at > utc_now()),
            )
            .order_by(Memory.updated_at.desc().nullslast(), Memory.created_at.desc())
            .limit(MAX_CURATOR_MEMORIES)
        )
        return list(result.scalars().all())


async def _curator_memory_context(
    scopes: Sequence[Dict[str, str]],
    query: str,
    selected_memory_id: str | None,
) -> tuple[List[Memory], List[Dict[str, Any]]]:
    recent = await _recent_scope_memories(scopes)
    by_id = {memory.id: memory for memory in recent}
    if selected_memory_id and selected_memory_id not in by_id:
        async with async_session_maker() as session:
            selected = await session.get(Memory, selected_memory_id)
        if selected is None:
            raise MemoryCuratorNotFoundError("Memory not found")
        allowed = {(scope["scope_type"], scope["scope_id"]) for scope in scopes}
        if (selected.scope_type, selected.scope_id) not in allowed:
            raise MemoryCuratorError("Selected memory is outside the curator workspace")
        by_id[selected.id] = selected

    readiness: List[Dict[str, Any]] = []
    ranked_groups: List[tuple[str, List[Dict[str, Any]]]] = []
    scopes_by_model: Dict[str, List[Dict[str, str]]] = {}
    for scope in scopes:
        model = await resolve_scope_embedding_model(scope["scope_type"], scope["scope_id"])
        scopes_by_model.setdefault(model, []).append(scope)
    if query.strip():
        for model, model_scopes in scopes_by_model.items():
            try:
                await require_embedding_model_ready(model)
                readiness.append({
                    "embedding_model": model,
                    "scopes": model_scopes,
                    "ready": True,
                    "degraded": False,
                })
                vector = await invoke_with_retry(get_embedding_model(model).aembed_query, query)
                hits = await get_vector_db().search_memory(
                    query_vector=vector,
                    embedding_model=model,
                    scope_filters=model_scopes,
                    limit=MAX_CURATOR_MEMORIES,
                    query_text=query,
                )
                ranked_groups.append((model, hits))
            except Exception as exc:
                readiness.append({
                    "embedding_model": model,
                    "scopes": model_scopes,
                    "ready": False,
                    "degraded": True,
                    "reason": str(exc)[:300],
                })

    ordered: List[Memory] = []
    if selected_memory_id and selected_memory_id in by_id:
        ordered.append(by_id[selected_memory_id])
    normalized_query = " ".join(query.casefold().split())
    for memory in recent:
        normalized_content = " ".join(memory.content.casefold().split())
        if normalized_query and normalized_query == normalized_content and memory not in ordered:
            ordered.append(memory)
    for hit in _rank_fuse_memory_hits(ranked_groups):
        memory_id = str(hit.get("memory_id") or "")
        memory = by_id.get(memory_id)
        if memory is None and memory_id:
            async with async_session_maker() as session:
                memory = await session.get(Memory, memory_id)
            if memory is not None:
                by_id[memory.id] = memory
        if memory is not None and memory not in ordered:
            ordered.append(memory)
    for memory in recent:
        if memory not in ordered:
            ordered.append(memory)
    return ordered[:MAX_CURATOR_MEMORIES], readiness


def _bounded_context_payload(
    memories: Sequence[Memory],
    *,
    context_window: int,
) -> List[Dict[str, Any]]:
    char_budget = max(1, int(context_window * 4 * 0.25))
    used = 0
    result = []
    for memory in memories:
        payload = {
            "id": memory.id,
            "scope_type": memory.scope_type,
            "scope_id": memory.scope_id,
            "memory_type": memory.memory_type,
            "content": memory.content,
            "summary": memory.summary,
            "updated_at": iso_utc_z(memory.updated_at or memory.created_at),
        }
        remaining = char_budget - used
        encoded = json.dumps(payload, ensure_ascii=True)
        if len(encoded) > remaining:
            overhead = len(encoded) - len(memory.content)
            available_content = remaining - overhead
            if available_content <= 0:
                break
            payload["content"] = memory.content[:available_content]
            encoded = json.dumps(payload, ensure_ascii=True)
        if used + len(encoded) > char_budget:
            break
        result.append(payload)
        used += len(encoded)
    return result


def _consent_snapshot(thread: Thread | None, project: Project | None) -> Dict[str, Any]:
    thread_memory = normalize_thread_memory_settings(thread.settings if thread else {})
    project_memory = normalize_project_memory_settings(project.settings_json if project else {})
    return {
        "administration_available": True,
        "thread_reads_project_memory": (
            thread_memory["thread_reads_project_memory"] if thread else None
        ),
        "project_reads_user_memory": (
            project_memory["project_reads_user_memory"] if project else None
        ),
        "thread_reads_user_memory": (
            thread_memory["thread_reads_user_memory"] if thread else None
        ),
        "effective_user_recall": (
            project_memory["project_reads_user_memory"]
            and thread_memory["thread_reads_user_memory"]
            if thread and project else None
        ),
    }


def _normalize_choice(raw: Any, index: int) -> Dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    label = str(raw.get("label") or "").strip()
    if not label:
        return None
    return {
        "id": str(raw.get("id") or f"choice-{index + 1}"),
        "label": label[:300],
        "description": str(raw.get("description") or "")[:1000],
        "user_message": str(raw.get("user_message") or label)[:2000],
    }


def _normalize_operation(
    raw: Any,
    *,
    memory_lookup: Dict[str, Memory],
    visible_scopes: set[tuple[str, str]],
) -> Dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action") or "")
    if action not in VALID_ACTIONS:
        return None
    if action == "noop":
        return {"action": "noop"}
    memory_id = str(raw.get("memory_id") or "").strip() or None
    existing = memory_lookup.get(memory_id or "")
    scope_type = str(raw.get("scope_type") or (existing.scope_type if existing else ""))
    scope_id = str(raw.get("scope_id") or (existing.scope_id if existing else "")).strip()
    if scope_type == MemoryScopeType.USER.value:
        scope_id = LOCAL_USER_MEMORY_SCOPE_ID
    if (scope_type, scope_id) not in visible_scopes:
        return None
    if action in {"update", "delete"} and existing is None:
        return None
    operation: Dict[str, Any] = {
        "action": action,
        "scope_type": scope_type,
        "scope_id": scope_id,
    }
    if existing is not None:
        operation.update({
            "memory_id": existing.id,
            "expected_updated_at": iso_utc_z(existing.updated_at or existing.created_at),
        })
    if action in {"create", "update"}:
        content = str(raw.get("content") or "").strip()
        memory_type = str(raw.get("memory_type") or (existing.memory_type if existing else "semantic"))
        if not content or memory_type not in VALID_MEMORY_TYPES:
            return None
        try:
            confidence = float(raw.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        operation.update({
            "content": content[:12000],
            "summary": str(raw.get("summary") or "")[:4000],
            "memory_type": memory_type,
            "confidence": max(0.0, min(1.0, confidence)),
            "source_refs_json": raw.get("source_refs_json") if isinstance(raw.get("source_refs_json"), dict) else {},
        })
    return operation


async def respond_to_memory_curator(req: MemoryCuratorRespondRequest) -> Dict[str, Any]:
    scopes, thread, project = await _resolve_visible_scopes(req.context)
    consent = _consent_snapshot(thread, project)
    if not await check_chat_model_ready(req.llm_model):
        raise MemoryCuratorModelUnavailableError(f"Chat model {req.llm_model} is unavailable")
    review = await _review_batch(thread) if req.mode == "conversation_review" else None
    if req.mode == "conversation_review" and not (review or {}).get("turns"):
        return {
            "message": "There are no new completed conversation turns to review.",
            "state": "no_changes",
            "choices": [],
            "operations": [],
            "review": review,
            "embedding_readiness": [],
            "consent": consent,
        }

    transcript = [{"role": item.role, "content": item.content} for item in req.messages]
    query_parts = [item["content"] for item in transcript[-4:]]
    if review:
        for turn in review["turns"]:
            query_parts.extend([turn["question"], turn["answer"]])
    query = "\n".join(part for part in query_parts if part).strip()
    memories, readiness = await _curator_memory_context(scopes, query, req.memory_id)
    context_memories = _bounded_context_payload(memories, context_window=req.context_window)
    visible_scope_set = {(scope["scope_type"], scope["scope_id"]) for scope in scopes}

    system_prompt = (
        "You are AskPDF's memory curator. Help the user create, correct, consolidate, or remove "
        "durable memory. Never write memory directly. Return one strict JSON object with keys "
        "message, state, choices, and operations. state must be clarification, conflict, proposal, "
        "or no_changes. choices is an array of {id,label,description,user_message}. operations is "
        "an array using action create, update, delete, or noop. Create/update operations require "
        "scope_type, scope_id, memory_type, content, summary, confidence, and source_refs_json. "
        "Update/delete operations require memory_id. Prefer updating an existing memory over "
        "creating a duplicate. Surface contradictions as conflict choices. Do not infer sensitive "
        "facts or broaden scope without the user's clear direction. A proposal is never approval."
    )
    prompt_payload = {
        "mode": req.mode,
        "selected_scope": {
            "scope_type": req.context.selected_scope_type,
            "scope_id": (
                LOCAL_USER_MEMORY_SCOPE_ID
                if req.context.selected_scope_type == MemoryScopeType.USER.value
                else req.context.selected_scope_id
            ),
        },
        "visible_scopes": scopes,
        "selected_memory_id": req.memory_id,
        "conversation": transcript,
        "review": review,
        "existing_memories": context_memories,
        "recall_consent": consent,
    }
    response = await invoke_with_retry(
        get_llm(req.llm_model, temperature=0.0).ainvoke,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(prompt_payload, ensure_ascii=True)),
        ],
    )
    parsed = safe_json_object(str(getattr(response, "content", "") or ""))
    state = str(parsed.get("state") or "")
    if state not in {"clarification", "conflict", "proposal", "no_changes"}:
        state = "clarification"
    choices = [
        choice
        for index, raw in enumerate(parsed.get("choices") or [])
        if (choice := _normalize_choice(raw, index)) is not None
    ][:6]
    memory_lookup = {memory.id: memory for memory in memories}
    operations = [
        operation
        for raw in (parsed.get("operations") or [])
        if (operation := _normalize_operation(
            raw,
            memory_lookup=memory_lookup,
            visible_scopes=visible_scope_set,
        )) is not None
    ][:20]
    if state == "proposal" and not [op for op in operations if op["action"] != "noop"]:
        state = "clarification"
    if state in {"clarification", "conflict"} and not choices:
        choices = [{
            "id": "clarify",
            "label": "Describe the intended memory",
            "description": "Tell the curator what should be remembered and at which scope.",
            "user_message": "Please help me clarify the exact memory and scope.",
        }]
    return {
        "message": str(parsed.get("message") or "Please review the proposed memory changes.")[:6000],
        "state": state,
        "choices": choices,
        "operations": operations,
        "review": review,
        "embedding_readiness": readiness,
        "context_memory_count": len(context_memories),
        "consent": consent,
    }


def _operation_scope(operation: MemoryCuratorOperation) -> tuple[str, str]:
    scope_type = str(operation.scope_type or "")
    scope_id = str(operation.scope_id or "").strip()
    if scope_type == MemoryScopeType.USER.value:
        scope_id = LOCAL_USER_MEMORY_SCOPE_ID
    return scope_type, scope_id


def _timestamp_matches(memory: Memory, expected: str | None) -> bool:
    if not expected:
        return False
    return expected == iso_utc_z(memory.updated_at or memory.created_at)


async def apply_memory_curator_change_set(req: MemoryCuratorApplyRequest) -> Dict[str, Any]:
    scopes, context_thread, _project = await _resolve_visible_scopes(req.context)
    visible_scopes = {(scope["scope_type"], scope["scope_id"]) for scope in scopes}
    operations = [operation for operation in req.operations if operation.action != "noop"]
    affected_ids = [operation.memory_id for operation in operations if operation.memory_id]
    if len(affected_ids) != len(set(affected_ids)):
        raise MemoryCuratorError("A memory may only appear once in a change set")

    create_models: Dict[int, str] = {}
    for index, operation in enumerate(operations):
        scope = _operation_scope(operation)
        if operation.action == "create":
            if scope not in visible_scopes:
                raise MemoryCuratorError("Create operation targets an unavailable scope")
            if not operation.content or operation.memory_type not in VALID_MEMORY_TYPES:
                raise MemoryCuratorError("Create operation requires valid content and memory_type")
            model = await resolve_scope_embedding_model(*scope)
            await require_embedding_model_ready(model)
            create_models[index] = model

    changed: List[Memory] = []
    deleted_ids: List[str] = []
    cleanup_targets: List[Tuple[str, str]] = []
    cleaned_vector_targets: List[Tuple[str, str]] = []
    try:
        async with async_session_maker() as session:
            async with session.begin():
                locked: Dict[str, Memory] = {}
                if affected_ids:
                    result = await session.execute(
                        select(Memory)
                        .where(Memory.id.in_(affected_ids))
                        .order_by(Memory.id)
                        .with_for_update()
                    )
                    locked = {memory.id: memory for memory in result.scalars().all()}
                missing = [memory_id for memory_id in affected_ids if memory_id not in locked]
                if missing:
                    raise MemoryCuratorNotFoundError(f"Memory not found: {missing[0]}")
                stale = [
                    memory for memory_id, memory in locked.items()
                    if not _timestamp_matches(
                        memory,
                        next(op.expected_updated_at for op in operations if op.memory_id == memory_id),
                    )
                ]
                if stale:
                    raise MemoryChangedError(stale)

                for operation in operations:
                    if operation.action == "create":
                        continue
                    memory = locked[operation.memory_id or ""]
                    if (memory.scope_type, memory.scope_id) not in visible_scopes:
                        raise MemoryCuratorError("Operation targets memory outside this workspace")
                    if _operation_scope(operation) != (memory.scope_type, memory.scope_id):
                        raise MemoryCuratorError("Memory scope cannot be changed by update or delete")
                    if operation.action == "update":
                        if not operation.content or operation.memory_type not in VALID_MEMORY_TYPES:
                            raise MemoryCuratorError("Update operation requires valid content and memory_type")
                        await require_embedding_model_ready(memory.embedding_model)
                    cleanup_targets.append((memory.id, memory.embedding_model))

                for operation in operations:
                    if operation.action not in {"create", "update"}:
                        continue
                    scope_type, scope_id = _operation_scope(operation)
                    duplicate_query = select(Memory).where(
                        Memory.scope_type == scope_type,
                        Memory.scope_id == scope_id,
                        Memory.status == "active",
                        Memory.content_hash == memory_content_hash(operation.content or ""),
                    )
                    if operation.memory_id:
                        duplicate_query = duplicate_query.where(Memory.id != operation.memory_id)
                    duplicate = (await session.execute(duplicate_query.limit(1))).scalar_one_or_none()
                    if duplicate is not None:
                        raise MemoryCuratorError(
                            f"An identical active memory already exists in this scope: {duplicate.id}"
                        )

                reviewed_turn = None
                review_thread = None
                if req.review_cursor is not None:
                    cursor = req.review_cursor
                    if context_thread is None or cursor.thread_id != context_thread.id:
                        raise MemoryCuratorError("Review cursor does not match the active thread")
                    reviewed_turn = await session.get(ChatTurn, cursor.reviewed_through_turn_id)
                    if (
                        reviewed_turn is None
                        or reviewed_turn.thread_id != cursor.thread_id
                        or reviewed_turn.status != ChatTurnStatus.COMPLETED.value
                        or reviewed_turn.created_at != cursor.reviewed_through_created_at
                    ):
                        raise MemoryCuratorError("Review cursor does not reference an eligible turn")
                    review_thread = await session.get(Thread, cursor.thread_id, with_for_update=True)
                    if review_thread is None:
                        raise MemoryCuratorNotFoundError("Review thread not found")

                for memory_id, embedding_model in cleanup_targets:
                    if not await get_vector_db().delete_memory_vectors(memory_id, embedding_model):
                        raise MemoryCuratorError(f"Failed to clean vectors for memory {memory_id}")
                    cleaned_vector_targets.append((memory_id, embedding_model))

                now = utc_now()
                for index, operation in enumerate(operations):
                    scope_type, scope_id = _operation_scope(operation)
                    source_refs = dict(operation.source_refs_json or {})
                    if req.review_cursor is not None:
                        source_refs.update({
                            "curator_mode": "conversation_review",
                            "source_thread_id": req.review_cursor.thread_id,
                            "reviewed_through_turn_id": req.review_cursor.reviewed_through_turn_id,
                        })
                    if operation.action == "create":
                        memory = Memory(
                            id=str(uuid.uuid4()),
                            scope_type=scope_type,
                            scope_id=scope_id,
                            memory_type=operation.memory_type or MemoryType.SEMANTIC.value,
                            content=str(operation.content or "").strip(),
                            summary=operation.summary or "",
                            embedding_model=create_models[index],
                            content_hash=memory_content_hash(operation.content or ""),
                            index_status="pending",
                            index_attempts=0,
                            source_refs_json=source_refs,
                            confidence=operation.confidence if operation.confidence is not None else 1.0,
                            status="active",
                            visibility=(
                                MemoryVisibility.PROJECT.value
                                if scope_type == MemoryScopeType.PROJECT.value
                                else MemoryVisibility.PRIVATE.value
                            ),
                            created_by=req.actor_id,
                            created_at=now,
                            updated_at=now,
                        )
                        session.add(memory)
                        await session.flush()
                        session.add(MemoryEvent(
                            memory_id=memory.id,
                            event_type="curator_created",
                            actor_id=req.actor_id,
                            payload_json={"mode": "confirmed_change_set"},
                            created_at=now,
                        ))
                        changed.append(memory)
                    elif operation.action == "update":
                        memory = locked[operation.memory_id or ""]
                        updated = await MemoryRepository(session).update_memory(
                            memory.id,
                            memory_type=operation.memory_type or memory.memory_type,
                            content=str(operation.content or "").strip(),
                            summary=operation.summary or "",
                            content_hash=memory_content_hash(operation.content or ""),
                            confidence=(
                                operation.confidence
                                if operation.confidence is not None
                                else memory.confidence
                            ),
                            source_refs_json=source_refs,
                            actor_id=req.actor_id,
                            event_type="curator_updated",
                            event_payload={"mode": "confirmed_change_set"},
                            updated_at=now,
                        )
                        if updated is None:
                            raise MemoryCuratorNotFoundError("Memory disappeared during update")
                        changed.append(updated)
                    elif operation.action == "delete":
                        memory = locked[operation.memory_id or ""]
                        await session.execute(delete(MemoryEvent).where(MemoryEvent.memory_id == memory.id))
                        await session.delete(memory)
                        deleted_ids.append(memory.id)

                if reviewed_turn is not None and review_thread is not None:
                    metadata = dict(review_thread.thread_metadata or {})
                    metadata["memory_curator"] = {
                        "reviewed_through_turn_id": reviewed_turn.id,
                        "reviewed_through_created_at": iso_utc_z(reviewed_turn.created_at),
                        "reviewed_at": iso_utc_z(now),
                    }
                    replace_jsonb_field(review_thread, "thread_metadata", metadata)

                touched_projects = {
                    scope_id
                    for scope_type, scope_id in [
                        _operation_scope(operation) for operation in operations
                    ]
                    if scope_type == MemoryScopeType.PROJECT.value
                }
                for project_id in touched_projects:
                    await touch_project_activity(session, project_id, occurred_at=now)
                await session.flush()
    except Exception:
        for memory_id, _embedding_model in cleaned_vector_targets:
            try:
                async with async_session_maker() as session:
                    current = await session.get(Memory, memory_id)
                if current is not None:
                    await index_memory_record(current)
            except Exception:
                logger.exception(
                    "Failed to restore memory vector after curator rollback | memory_id=%s",
                    memory_id,
                )
        raise

    warnings = []
    indexed_records = []
    for memory in changed:
        try:
            await index_memory_record(memory)
        except Exception as exc:
            warnings.append({
                "code": "memory_index_failed",
                "memory_id": memory.id,
                "message": str(exc)[:500],
            })
        async with async_session_maker() as session:
            refreshed = await session.get(Memory, memory.id)
        if refreshed is not None:
            indexed_records.append(memory_payload(refreshed))
    return {
        "changed_memories": indexed_records,
        "deleted_memory_ids": deleted_ids,
        "warnings": warnings,
        "review_cursor_advanced": bool(req.review_cursor),
    }
