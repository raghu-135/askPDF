"""Context-aware effective durable-memory projection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import and_, or_
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import MemoryScopeType
from app.db.models_sqlmodel import Memory, MemoryOverride, Project, Thread
from app.services.embedding_model_service import EmbeddingModelResolutionError
from app.services.memory_policy import (
    LOCAL_USER_MEMORY_SCOPE_ID,
    normalize_project_memory_settings,
    normalize_thread_memory_settings,
)
from app.time_utils import iso_utc_z


_SCOPE_ORDER = (
    MemoryScopeType.THREAD.value,
    MemoryScopeType.PROJECT.value,
    MemoryScopeType.USER.value,
)


def _allowed_scope_types(raw: Optional[List[str]]) -> set[str]:
    if raw is None:
        return set(_SCOPE_ORDER)
    return {str(item) for item in raw if str(item) in _SCOPE_ORDER}


def _memory_ref(memory: Memory) -> Dict[str, Any]:
    return {
        "id": memory.id,
        "scope_type": memory.scope_type,
        "scope_id": memory.scope_id,
        "content": memory.content,
        "updated_at": iso_utc_z(memory.updated_at or memory.created_at),
    }


def memory_payload(
    memory: Memory,
    *,
    outgoing: Sequence[Memory] = (),
    incoming: Sequence[Memory] = (),
) -> Dict[str, Any]:
    return {
        "id": memory.id,
        "scope_type": memory.scope_type,
        "scope_id": memory.scope_id,
        "content": memory.content,
        "embedding_model": memory.embedding_model,
        "content_hash": memory.content_hash,
        "index_status": memory.index_status,
        "index_attempts": memory.index_attempts,
        "indexed_at": iso_utc_z(memory.indexed_at) if memory.indexed_at else None,
        "index_error": memory.index_error,
        "source_refs_json": memory.source_refs_json or {},
        "overrides": [_memory_ref(item) for item in outgoing],
        "overridden_by": [_memory_ref(item) for item in incoming],
        "created_at": iso_utc_z(memory.created_at) if memory.created_at else None,
        "updated_at": iso_utc_z(memory.updated_at) if memory.updated_at else None,
    }


def _workspace_scopes(
    *,
    thread: Optional[Thread],
    project: Optional[Project],
) -> List[Dict[str, str]]:
    scopes: List[Dict[str, str]] = []
    if thread is not None:
        scopes.append({"scope_type": MemoryScopeType.THREAD.value, "scope_id": thread.id})
    if project is not None:
        scopes.append({"scope_type": MemoryScopeType.PROJECT.value, "scope_id": project.id})
    scopes.append({
        "scope_type": MemoryScopeType.USER.value,
        "scope_id": LOCAL_USER_MEMORY_SCOPE_ID,
    })
    return scopes


async def _workspace_sections(
    *,
    thread: Optional[Thread],
    project: Optional[Project],
    policy: Dict[str, Any],
    applied_edges: Sequence[MemoryOverride],
    suppressed_ids: set[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Build the administrative hierarchy without changing recall eligibility."""

    scopes = _workspace_scopes(thread=thread, project=project)
    clauses = [
        and_(Memory.scope_type == scope["scope_type"], Memory.scope_id == scope["scope_id"])
        for scope in scopes
    ]
    async with async_session_maker() as session:
        memories = list((await session.execute(
            select(Memory)
            .where(or_(*clauses))
            .order_by(Memory.updated_at.desc().nullslast(), Memory.created_at.desc(), Memory.id)
        )).scalars().all())
        visible_ids = {memory.id for memory in memories}
        edges = list((await session.execute(
            select(MemoryOverride).where(
                MemoryOverride.overriding_memory_id.in_(visible_ids),
                MemoryOverride.overridden_memory_id.in_(visible_ids),
            )
        )).scalars().all()) if visible_ids else []

    by_id = {memory.id: memory for memory in memories}
    outgoing: Dict[str, List[Memory]] = {}
    incoming: Dict[str, List[Memory]] = {}
    for edge in edges:
        source = by_id.get(edge.overriding_memory_id)
        target = by_id.get(edge.overridden_memory_id)
        if source and target:
            outgoing.setdefault(source.id, []).append(target)
            incoming.setdefault(target.id, []).append(source)

    applied_keys = {
        (edge.overriding_memory_id, edge.overridden_memory_id)
        for edge in applied_edges
    }
    applied_outgoing: Dict[str, List[Memory]] = {}
    applied_incoming: Dict[str, List[Memory]] = {}
    for source_id, target_id in applied_keys:
        source = by_id.get(source_id)
        target = by_id.get(target_id)
        if source and target:
            applied_outgoing.setdefault(source_id, []).append(target)
            applied_incoming.setdefault(target_id, []).append(source)

    searched = {
        (scope["scope_type"], scope["scope_id"])
        for scope in policy["searched_scopes"]
    }
    skipped_reasons = {
        scope["scope_type"]: scope["reason"]
        for scope in policy["skipped_scopes"]
    }
    bounded_limit = max(1, min(int(limit), 500))
    sections: List[Dict[str, Any]] = []
    for scope in scopes:
        scope_key = (scope["scope_type"], scope["scope_id"])
        recall_enabled = scope_key in searched
        scoped = [
            memory for memory in memories
            if (memory.scope_type, memory.scope_id) == scope_key
        ]
        records: List[Dict[str, Any]] = []
        for memory in scoped[:bounded_limit]:
            payload = memory_payload(
                memory,
                outgoing=outgoing.get(memory.id, []),
                incoming=incoming.get(memory.id, []),
            )
            if memory.index_status != "indexed":
                resolution_status = "unavailable"
            elif not recall_enabled:
                resolution_status = "recall_disabled"
            elif memory.id in suppressed_ids:
                resolution_status = "overridden"
            else:
                resolution_status = "effective"
            payload.update({
                "resolution_status": resolution_status,
                "applied_overrides": [
                    _memory_ref(item) for item in applied_outgoing.get(memory.id, [])
                ],
                "applied_overridden_by": [
                    _memory_ref(item) for item in applied_incoming.get(memory.id, [])
                ],
            })
            records.append(payload)
        sections.append({
            **scope,
            "recall_enabled": recall_enabled,
            "recall_skip_reason": None if recall_enabled else skipped_reasons.get(scope["scope_type"], "not_requested"),
            "memories": records,
            "truncated": len(scoped) > bounded_limit,
        })
    return sections


async def serialize_memories_with_relationships(memories: Sequence[Memory]) -> List[Dict[str, Any]]:
    """Serialize stored memories with compact incoming and outgoing relation details."""

    if not memories:
        return []
    selected_ids = {memory.id for memory in memories}
    async with async_session_maker() as session:
        edges = list((await session.execute(
            select(MemoryOverride).where(
                or_(
                    MemoryOverride.overriding_memory_id.in_(selected_ids),
                    MemoryOverride.overridden_memory_id.in_(selected_ids),
                )
            )
        )).scalars().all())
        related_ids = {
            memory_id
            for edge in edges
            for memory_id in (edge.overriding_memory_id, edge.overridden_memory_id)
        }
        related = list((await session.execute(
            select(Memory).where(Memory.id.in_(related_ids))
        )).scalars().all()) if related_ids else []
    by_id = {memory.id: memory for memory in related}
    outgoing: Dict[str, List[Memory]] = {}
    incoming: Dict[str, List[Memory]] = {}
    for edge in edges:
        source = by_id.get(edge.overriding_memory_id)
        target = by_id.get(edge.overridden_memory_id)
        if source and target:
            outgoing.setdefault(source.id, []).append(target)
            incoming.setdefault(target.id, []).append(source)
    return [
        memory_payload(
            memory,
            outgoing=outgoing.get(memory.id, []),
            incoming=incoming.get(memory.id, []),
        )
        for memory in memories
    ]


async def _scope_policy(
    *,
    thread_id: Optional[str],
    project_id: Optional[str],
    allowed_scopes: Optional[List[str]],
) -> tuple[Dict[str, Any], Optional[Thread], Optional[Project]]:
    allowed = _allowed_scope_types(allowed_scopes)
    scopes: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []
    thread: Optional[Thread] = None
    project: Optional[Project] = None

    async with async_session_maker() as session:
        if thread_id:
            thread = await session.get(Thread, thread_id)
            if thread is None:
                raise EmbeddingModelResolutionError(f"Thread not found: {thread_id}")
            project = await session.get(Project, thread.project_id) if thread.project_id else None
            if project is None:
                raise EmbeddingModelResolutionError(f"Project not found for thread: {thread_id}")
        elif project_id:
            project = await session.get(Project, project_id)
            if project is None:
                raise EmbeddingModelResolutionError(f"Project not found: {project_id}")

    requested = [item for item in _SCOPE_ORDER if item in allowed]
    if thread is not None:
        thread_settings = normalize_thread_memory_settings(thread.settings)
        project_settings = normalize_project_memory_settings(project.settings_json if project else {})
        if MemoryScopeType.THREAD.value in allowed:
            scopes.append({"scope_type": "thread", "scope_id": thread.id})
        else:
            skipped.append({"scope_type": "thread", "reason": "not_requested"})
        if MemoryScopeType.PROJECT.value not in allowed:
            skipped.append({"scope_type": "project", "reason": "not_requested"})
        elif not thread_settings["thread_reads_project_memory"]:
            skipped.append({"scope_type": "project", "reason": "thread_opt_out"})
        else:
            scopes.append({"scope_type": "project", "scope_id": project.id})
        if MemoryScopeType.USER.value not in allowed:
            skipped.append({"scope_type": "user", "reason": "not_requested"})
        elif not project_settings["project_reads_user_memory"]:
            skipped.append({"scope_type": "user", "reason": "project_opt_out"})
        elif not thread_settings["thread_reads_user_memory"]:
            skipped.append({"scope_type": "user", "reason": "thread_opt_out"})
        else:
            scopes.append({"scope_type": "user", "scope_id": LOCAL_USER_MEMORY_SCOPE_ID})
    elif project is not None:
        project_settings = normalize_project_memory_settings(project.settings_json)
        if MemoryScopeType.PROJECT.value in allowed:
            scopes.append({"scope_type": "project", "scope_id": project.id})
        else:
            skipped.append({"scope_type": "project", "reason": "not_requested"})
        if MemoryScopeType.USER.value not in allowed:
            skipped.append({"scope_type": "user", "reason": "not_requested"})
        elif not project_settings["project_reads_user_memory"]:
            skipped.append({"scope_type": "user", "reason": "project_opt_out"})
        else:
            scopes.append({"scope_type": "user", "scope_id": LOCAL_USER_MEMORY_SCOPE_ID})
    else:
        requested = [item for item in requested if item == MemoryScopeType.USER.value]
        if MemoryScopeType.USER.value in allowed:
            scopes.append({"scope_type": "user", "scope_id": LOCAL_USER_MEMORY_SCOPE_ID})
        else:
            skipped.append({"scope_type": "user", "reason": "not_requested"})

    return ({
        "requested_scopes": requested,
        "searched_scopes": scopes,
        "skipped_scopes": skipped,
    }, thread, project)


async def resolve_effective_memory_context(
    *,
    thread_id: Optional[str] = None,
    project_id: Optional[str] = None,
    allowed_scopes: Optional[List[str]] = None,
    limit: int = 500,
) -> Dict[str, Any]:
    """Resolve the indexed, non-overridden memory projection for one context."""

    if thread_id and project_id:
        raise ValueError("Specify thread_id or project_id, not both")
    policy, thread, project = await _scope_policy(
        thread_id=thread_id,
        project_id=project_id,
        allowed_scopes=allowed_scopes,
    )
    scopes = policy["searched_scopes"]
    if not scopes:
        workspace_sections = await _workspace_sections(
            thread=thread,
            project=project,
            policy=policy,
            applied_edges=[],
            suppressed_ids=set(),
            limit=limit,
        )
        return {
            "context": {"type": "thread" if thread_id else "project" if project_id else "global", "id": thread_id or project_id or LOCAL_USER_MEMORY_SCOPE_ID},
            "policy": policy,
            "memories": [],
            "memory_records": [],
            "applied_overrides": [],
            "suppressed_memory_ids": [],
            "excluded_memory_ids": [],
            "unavailable_memory_count": 0,
            "truncated": False,
            "workspace_sections": workspace_sections,
        }

    clauses = [
        and_(Memory.scope_type == scope["scope_type"], Memory.scope_id == scope["scope_id"])
        for scope in scopes
    ]
    async with async_session_maker() as session:
        memories = list((await session.execute(
            select(Memory)
            .where(or_(*clauses))
            .order_by(Memory.updated_at.desc().nullslast(), Memory.created_at.desc(), Memory.id)
        )).scalars().all())
        by_id = {memory.id: memory for memory in memories}
        memory_ids = list(by_id)
        edges = []
        if memory_ids:
            edges = list((await session.execute(
                select(MemoryOverride).where(
                    MemoryOverride.overriding_memory_id.in_(memory_ids),
                    MemoryOverride.overridden_memory_id.in_(memory_ids),
                )
            )).scalars().all())

    indexed_ids = {memory.id for memory in memories if memory.index_status == "indexed"}
    unavailable_ids = {memory.id for memory in memories if memory.id not in indexed_ids}
    adjacency: Dict[str, List[MemoryOverride]] = {}
    for edge in edges:
        adjacency.setdefault(edge.overriding_memory_id, []).append(edge)
    applied_by_key: Dict[tuple[str, str], MemoryOverride] = {}
    suppressed_ids: set[str] = set()
    stack = list(indexed_ids)
    traversed_sources: set[str] = set()
    while stack:
        source_id = stack.pop()
        if source_id in traversed_sources:
            continue
        traversed_sources.add(source_id)
        for edge in adjacency.get(source_id, []):
            key = (edge.overriding_memory_id, edge.overridden_memory_id)
            applied_by_key[key] = edge
            if edge.overridden_memory_id not in suppressed_ids:
                suppressed_ids.add(edge.overridden_memory_id)
                stack.append(edge.overridden_memory_id)
    applied_edges = list(applied_by_key.values())
    effective = [
        memory for memory in memories
        if memory.id in indexed_ids and memory.id not in suppressed_ids
    ]
    outgoing: Dict[str, List[Memory]] = {}
    incoming: Dict[str, List[Memory]] = {}
    for edge in edges:
        source = by_id.get(edge.overriding_memory_id)
        target = by_id.get(edge.overridden_memory_id)
        if source and target:
            outgoing.setdefault(source.id, []).append(target)
            incoming.setdefault(target.id, []).append(source)

    bounded_limit = max(1, min(int(limit), 500))
    visible = effective[:bounded_limit]
    workspace_sections = await _workspace_sections(
        thread=thread,
        project=project,
        policy=policy,
        applied_edges=applied_edges,
        suppressed_ids=suppressed_ids,
        limit=limit,
    )
    return {
        "context": {
            "type": "thread" if thread else "project" if project else "global",
            "id": thread.id if thread else project.id if project else LOCAL_USER_MEMORY_SCOPE_ID,
            "project_id": project.id if project else None,
        },
        "policy": policy,
        "memories": [
            memory_payload(
                memory,
                outgoing=outgoing.get(memory.id, []),
                incoming=incoming.get(memory.id, []),
            )
            for memory in visible
        ],
        "memory_records": effective,
        "applied_overrides": [
            {
                "overriding_memory_id": edge.overriding_memory_id,
                "overridden_memory_id": edge.overridden_memory_id,
            }
            for edge in applied_edges
        ],
        "suppressed_memory_ids": sorted(suppressed_ids),
        "excluded_memory_ids": sorted(suppressed_ids | unavailable_ids),
        "unavailable_memory_count": len(memories) - len(indexed_ids),
        "truncated": len(effective) > bounded_limit,
        "workspace_sections": workspace_sections,
    }


async def memory_scope_policy_for_thread(
    thread_id: str,
    allowed_scopes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    policy, _thread, _project = await _scope_policy(
        thread_id=thread_id,
        project_id=None,
        allowed_scopes=allowed_scopes,
    )
    return policy
