"""Framework-neutral read and proposal tools for app-owned durable memory."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import and_, func, or_
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import MemoryScopeType
from app.db.models_sqlmodel import GlobalMemoryRepresentation, Memory, MemoryOverride, Project, Thread
from app.db.vector import get_vector_db
from app.models.llm_server_client import get_embedding_model
from app.models.memory_tools import (
    MEMORY_APPLY_CONFIRMED,
    MEMORY_PROPOSE,
    MEMORY_READ_EFFECTIVE,
    MEMORY_READ_STORED,
    MemoryChangeIntent,
    MemoryGetInput,
    MemoryOperationSummary,
    MemoryPrepareChangeInput,
    MemorySearchInput,
    MemoryToolContext,
    MemoryToolScope,
)
from app.models.requests import MemoryCuratorApplyRequest, MemoryCuratorOperation
from app.models.retry import invoke_with_retry
from app.services.effective_memory_service import (
    resolve_effective_memory_context,
    serialize_memories_with_relationships,
)
from app.services.embedding_model_service import (
    GLOBAL_MEMORY_EMBEDDING_MODEL,
    require_embedding_model_ready,
    resolve_scope_embedding_model,
)
from app.services.memory_policy import LOCAL_USER_MEMORY_SCOPE_ID
from app.services.memory_repair_scheduler import schedule_global_representation_repair
from app.services.memory_service import _merge_same_model_memory_hits, memory_content_hash
from app.models.memory_tools import normalize_memory_attributes
from app.time_utils import iso_utc_z


class MemoryToolError(ValueError):
    code = "memory_tool_error"


class MemoryToolPermissionError(MemoryToolError):
    code = "memory_tool_permission_denied"


class MemoryToolNotFoundError(MemoryToolError):
    code = "memory_not_found"


_SCOPE_RANK = {"user": 0, "project": 1, "thread": 2}
logger = logging.getLogger(__name__)


async def _workspace_search_model(context: MemoryToolContext) -> str:
    if context.thread_id:
        return await resolve_scope_embedding_model(MemoryScopeType.THREAD.value, context.thread_id)
    if context.project_id:
        return await resolve_scope_embedding_model(MemoryScopeType.PROJECT.value, context.project_id)
    return GLOBAL_MEMORY_EMBEDDING_MODEL


async def _schedule_missing_global_repairs(
    context: MemoryToolContext,
    memories: Sequence[Memory],
    embedding_model: str,
) -> int:
    if embedding_model == GLOBAL_MEMORY_EMBEDDING_MODEL:
        return 0
    global_ids = {
        memory.id for memory in memories
        if memory.scope_type == MemoryScopeType.USER.value
    }
    if not global_ids:
        return 0
    async with async_session_maker() as session:
        indexed_ids = set((await session.execute(
            select(GlobalMemoryRepresentation.memory_id).where(
                GlobalMemoryRepresentation.memory_id.in_(global_ids),
                GlobalMemoryRepresentation.embedding_model == embedding_model,
                GlobalMemoryRepresentation.index_status == "indexed",
            )
        )).scalars().all())
    missing = global_ids - indexed_ids
    if missing:
        logger.error(
            "Global memory representations are unavailable for memory tool search; scheduling repair | "
            "model=%s missing=%s project_id=%s thread_id=%s",
            embedding_model,
            len(missing),
            context.project_id,
            context.thread_id,
        )
        schedule_global_representation_repair(embedding_model)
    return len(missing)


def _require_capability(context: MemoryToolContext, capability: str) -> None:
    if capability not in context.capabilities:
        raise MemoryToolPermissionError(f"Memory capability is not granted: {capability}")


def _canonical_scope_id(scope_type: str, scope_id: str | None) -> str:
    if scope_type == MemoryScopeType.USER.value:
        return LOCAL_USER_MEMORY_SCOPE_ID
    return str(scope_id or "").strip()


def _visible_scope_map(context: MemoryToolContext) -> Dict[str, MemoryToolScope]:
    return {scope.scope_type: scope for scope in context.visible_scopes}


def _scope_for_type(context: MemoryToolContext, scope_type: str | None) -> MemoryToolScope:
    requested = str(scope_type or context.selected_scope.scope_type)
    scope = _visible_scope_map(context).get(requested)
    if scope is None:
        raise MemoryToolPermissionError(f"Memory scope is not available in this workspace: {requested}")
    return scope


async def build_memory_tool_context(
    *,
    selected_scope_type: str,
    selected_scope_id: str,
    capabilities: Sequence[str],
    thread_id: str | None = None,
    project_id: str | None = None,
) -> tuple[MemoryToolContext, Thread | None, Project | None]:
    """Resolve trusted visible scopes from canonical thread/project ownership."""

    selected_id = _canonical_scope_id(selected_scope_type, selected_scope_id)
    async with async_session_maker() as session:
        thread = await session.get(Thread, thread_id) if thread_id else None
        if thread_id and thread is None:
            raise MemoryToolNotFoundError("Thread not found")
        resolved_project_id = project_id or (thread.project_id if thread else None)
        project = await session.get(Project, resolved_project_id) if resolved_project_id else None
        if resolved_project_id and project is None:
            raise MemoryToolNotFoundError("Project not found")
        if thread and project and thread.project_id != project.id:
            raise MemoryToolPermissionError("Thread does not belong to the selected project")

    scopes: List[MemoryToolScope] = []
    if thread is not None:
        scopes.append(MemoryToolScope(scope_type="thread", scope_id=thread.id))
    if project is not None:
        scopes.append(MemoryToolScope(scope_type="project", scope_id=project.id))
    scopes.append(MemoryToolScope(scope_type="user", scope_id=LOCAL_USER_MEMORY_SCOPE_ID))
    visible = {(scope.scope_type, scope.scope_id): scope for scope in scopes}
    selected = visible.get((selected_scope_type, selected_id))
    if selected is None:
        raise MemoryToolPermissionError("Selected memory scope is not available in this workspace")
    return MemoryToolContext(
        selected_scope=selected,
        visible_scopes=list(visible.values()),
        capabilities=list(dict.fromkeys(capabilities)),
        thread_id=thread.id if thread else None,
        project_id=project.id if project else None,
    ), thread, project


async def _stored_memories(context: MemoryToolContext, scope_types: Sequence[str] | None) -> List[Memory]:
    allowed_types = set(scope_types) if scope_types is not None else None
    scopes = [
        scope for scope in context.visible_scopes
        if allowed_types is None or scope.scope_type in allowed_types
    ]
    if not scopes:
        return []
    clauses = [
        and_(Memory.scope_type == scope.scope_type, Memory.scope_id == scope.scope_id)
        for scope in scopes
    ]
    async with async_session_maker() as session:
        return list((await session.execute(
            select(Memory)
            .where(or_(*clauses))
            .order_by(Memory.updated_at.desc().nullslast(), Memory.created_at.desc(), Memory.id)
        )).scalars().all())


async def _semantic_stored_search(
    context: MemoryToolContext,
    memories: Sequence[Memory],
    query: str,
    limit: int,
) -> tuple[List[Memory], List[Dict[str, Any]]]:
    by_id = {memory.id: memory for memory in memories}
    model = await _workspace_search_model(context)
    scopes: List[Dict[str, str]] = []
    for scope in context.visible_scopes:
        if not any(
            memory.scope_type == scope.scope_type and memory.scope_id == scope.scope_id
            for memory in memories
        ):
            continue
        scopes.append(scope.model_dump())

    readiness: List[Dict[str, Any]] = []
    missing = await _schedule_missing_global_repairs(context, memories, model)
    try:
        await require_embedding_model_ready(model)
        readiness.append({
            "embedding_model": model,
            "scopes": scopes,
            "ready": missing == 0,
            "reason": "global_representation_warming" if missing else None,
        })
        vector = await invoke_with_retry(get_embedding_model(model).aembed_query, query)
        hits = await get_vector_db().search_memory(
            query_vector=vector,
            embedding_model=model,
            scope_filters=scopes,
            limit=limit,
            query_text=query,
        )
    except Exception as exc:
        readiness.append({
            "embedding_model": model,
            "scopes": scopes,
            "ready": False,
            "reason": str(exc)[:300],
        })
        hits = []

    ordered: List[Memory] = []
    for hit in _merge_same_model_memory_hits(model, [hits]):
        memory = by_id.get(str(hit.get("memory_id") or ""))
        if memory is not None and memory not in ordered:
            ordered.append(memory)
    return ordered[:limit], readiness


async def search_memory_tool(
    context: MemoryToolContext,
    req: MemorySearchInput,
    *,
    query_vector: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Search effective or administratively stored memory inside trusted scopes."""

    if req.view == "effective":
        _require_capability(context, MEMORY_READ_EFFECTIVE)
        allowed = list(req.scope_types) if req.scope_types is not None else None
        if context.thread_id and req.query.strip():
            from app.services.memory_service import search_thread_memory
            result = await search_thread_memory(
                thread_id=context.thread_id,
                query=req.query,
                allowed_scopes=allowed,
                max_results=req.max_results,
                char_budget=req.char_budget,
                score_floor=req.score_floor,
                relative_score_ratio=req.relative_score_ratio,
                query_vector=query_vector,
            )
            return {
                **result,
                "readiness": [{
                    "embedding_model": item.get("embedding_model"),
                    "scopes": [{"scope_type": item.get("scope_type"), "scope_id": "default"}],
                    "ready": False,
                    "reason": item.get("reason"),
                } for item in result.get("representation_issues", [])],
            }
        effective = await resolve_effective_memory_context(
            thread_id=context.thread_id,
            project_id=None if context.thread_id else context.project_id,
            allowed_scopes=allowed,
            limit=max(req.max_results, 1),
        )
        if not req.query.strip():
            return {
                **{key: value for key, value in effective.items() if key != "memory_records"},
                "readiness": [],
            }
        policy = effective["policy"]
        scopes = policy["searched_scopes"]
        model = await _workspace_search_model(context)
        missing = await _schedule_missing_global_repairs(context, effective["memory_records"], model)
        await require_embedding_model_ready(model)
        readiness = [{
            "embedding_model": model,
            "scopes": scopes,
            "ready": missing == 0,
            "reason": "global_representation_warming" if missing else None,
        }]
        vector = await invoke_with_retry(get_embedding_model(model).aembed_query, req.query)
        hits = await get_vector_db().search_memory(
            query_vector=vector,
            embedding_model=model,
            scope_filters=scopes,
            excluded_memory_ids=effective["excluded_memory_ids"],
            limit=req.max_results,
            query_text=req.query,
        )
        effective_by_id = {memory.id: memory for memory in effective["memory_records"]}
        memories = []
        seen = set()
        for hit in _merge_same_model_memory_hits(model, [hits]):
            memory_id = str(hit.get("memory_id") or "")
            memory = effective_by_id.get(memory_id)
            if memory is None or memory_id in seen:
                continue
            seen.add(memory_id)
            memories.append({
                "id": memory.id,
                "scope_type": memory.scope_type,
                "scope_id": memory.scope_id,
                "content": memory.content,
                "attributes": normalize_memory_attributes(memory.attributes_json),
                "source_refs": memory.source_refs_json or {},
                "score": hit.get("score"),
                "score_type": hit.get("score_type") or "similarity",
                "raw_score": hit.get("raw_score"),
                "embedding_model": hit.get("embedding_model"),
                "created_at": iso_utc_z(memory.created_at) if memory.created_at else None,
                "updated_at": iso_utc_z(memory.updated_at) if memory.updated_at else None,
            })
            if len(memories) >= req.max_results:
                break
        return {
            "memories": memories,
            "scopes": policy["searched_scopes"],
            "scope_policy": policy,
            "applied_overrides": effective["applied_overrides"],
            "suppressed_memory_ids": effective["suppressed_memory_ids"],
            "unavailable_memory_count": effective["unavailable_memory_count"],
            "readiness": readiness,
        }

    _require_capability(context, MEMORY_READ_STORED)
    memories = await _stored_memories(context, req.scope_types)
    selected = None
    if req.selected_memory_id:
        selected = next((item for item in memories if item.id == req.selected_memory_id), None)
        if selected is None:
            raise MemoryToolNotFoundError("Selected memory is outside the memory workspace")
    readiness: List[Dict[str, Any]] = []
    ordered = memories
    if req.query.strip():
        ordered, readiness = await _semantic_stored_search(
            context,
            memories,
            req.query,
            req.max_results,
        )
    if selected is not None:
        ordered = [selected, *[item for item in ordered if item.id != selected.id]]
    bounded = ordered[:req.max_results]
    return {
        "memories": await serialize_memories_with_relationships(bounded),
        "readiness": readiness,
        "truncated": len(ordered) > len(bounded),
    }


async def get_memory_tool(context: MemoryToolContext, req: MemoryGetInput) -> Dict[str, Any]:
    _require_capability(context, MEMORY_READ_STORED)
    visible = {(scope.scope_type, scope.scope_id) for scope in context.visible_scopes}
    async with async_session_maker() as session:
        memories = list((await session.execute(
            select(Memory).where(Memory.id.in_(list(dict.fromkeys(req.memory_ids))))
        )).scalars().all())
    by_id = {memory.id: memory for memory in memories if (memory.scope_type, memory.scope_id) in visible}
    missing = [memory_id for memory_id in req.memory_ids if memory_id not in by_id]
    if missing:
        raise MemoryToolNotFoundError(f"Memory is not available in this workspace: {missing[0]}")
    ordered = [by_id[memory_id] for memory_id in req.memory_ids]
    return {"memories": await serialize_memories_with_relationships(ordered)}


async def _incoming_override_count(memory_id: str) -> int:
    async with async_session_maker() as session:
        return int((await session.execute(
            select(func.count()).select_from(MemoryOverride).where(
                MemoryOverride.overridden_memory_id == memory_id
            )
        )).scalar() or 0)


async def _memory_lookup(context: MemoryToolContext, ids: Iterable[str]) -> Dict[str, Memory]:
    unique_ids = sorted({str(memory_id) for memory_id in ids if memory_id})
    if not unique_ids:
        return {}
    visible = {(scope.scope_type, scope.scope_id) for scope in context.visible_scopes}
    async with async_session_maker() as session:
        rows = list((await session.execute(
            select(Memory).where(Memory.id.in_(unique_ids))
        )).scalars().all())
    lookup = {row.id: row for row in rows if (row.scope_type, row.scope_id) in visible}
    missing = [memory_id for memory_id in unique_ids if memory_id not in lookup]
    if missing:
        raise MemoryToolNotFoundError(f"Memory is not available in this workspace: {missing[0]}")
    return lookup


def _target_specs(targets: Sequence[Memory]) -> List[Dict[str, str]]:
    return [
        {"memory_id": target.id, "expected_updated_at": iso_utc_z(target.updated_at or target.created_at)}
        for target in targets
    ]


def _validate_target_scopes(source_scope: MemoryToolScope, targets: Sequence[Memory]) -> None:
    source_rank = _SCOPE_RANK[source_scope.scope_type]
    for target in targets:
        target_rank = _SCOPE_RANK[target.scope_type]
        if source_rank < target_rank:
            raise MemoryToolError("A broader memory cannot override a narrower memory")
        if source_rank == target_rank and source_scope.scope_id != target.scope_id:
            raise MemoryToolError("Memory overrides cannot cross peers at the same scope")


async def _find_exact_duplicate(
    scope: MemoryToolScope,
    content: str,
    *,
    exclude_memory_id: str | None = None,
) -> Memory | None:
    query = select(Memory).where(
        Memory.scope_type == scope.scope_type,
        Memory.scope_id == scope.scope_id,
        Memory.content_hash == memory_content_hash(content),
    )
    if exclude_memory_id:
        query = query.where(Memory.id != exclude_memory_id)
    async with async_session_maker() as session:
        return (await session.execute(query.limit(1))).scalar_one_or_none()


async def _validate_prepared_cycles(operations: Sequence[MemoryCuratorOperation]) -> None:
    replacements = {
        str(operation.memory_id or f"new:{operation.operation_group_id}")
        for operation in operations if operation.action in {"create", "update"}
    }
    async with async_session_maker() as session:
        existing = list((await session.execute(select(MemoryOverride))).scalars().all())
    edges = [
        (edge.overriding_memory_id, edge.overridden_memory_id)
        for edge in existing if edge.overriding_memory_id not in replacements
    ]
    for operation in operations:
        if operation.action in {"create", "update"}:
            source_id = str(operation.memory_id or f"new:{operation.operation_group_id}")
            edges.extend((source_id, target.memory_id) for target in operation.override_targets)
    adjacency: Dict[str, List[str]] = {}
    for source_id, target_id in edges:
        adjacency.setdefault(source_id, []).append(target_id)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(memory_id: str) -> None:
        if memory_id in visiting:
            raise MemoryToolError("Memory override relationships cannot contain a cycle")
        if memory_id in visited:
            return
        visiting.add(memory_id)
        for target_id in adjacency.get(memory_id, []):
            visit(target_id)
        visiting.remove(memory_id)
        visited.add(memory_id)

    for source_id in list(adjacency):
        visit(source_id)


async def prepare_memory_change(
    context: MemoryToolContext,
    req: MemoryPrepareChangeInput,
) -> Dict[str, Any]:
    """Normalize untrusted semantic intents into canonical, confirmable operations."""

    _require_capability(context, MEMORY_PROPOSE)
    referenced_ids = {
        memory_id
        for intent in req.intents
        for memory_id in [intent.memory_id, *(intent.override_target_ids or [])]
        if memory_id
    }
    lookup = await _memory_lookup(context, referenced_ids)
    operations: List[MemoryCuratorOperation] = []
    summaries: List[MemoryOperationSummary] = []

    for intent in req.intents:
        if intent.action == "noop":
            continue
        group_id = str(uuid.uuid4())
        source = lookup.get(intent.memory_id or "")
        if intent.action in {"update", "delete", "move", "set_overrides"} and source is None:
            raise MemoryToolNotFoundError("The requested memory does not exist in this workspace")
        target_ids = intent.override_target_ids
        if intent.action in {"create", "update", "move", "set_overrides"} and target_ids is None:
            raise MemoryToolError("A complete override target set is required")
        targets = [lookup[memory_id] for memory_id in (target_ids or [])]
        specs = _target_specs(targets)

        if intent.action == "create":
            scope = _scope_for_type(context, intent.scope_type)
            content = str(intent.content or "").strip()
            if not content:
                raise MemoryToolError("Create requires non-empty memory content")
            if await _find_exact_duplicate(scope, content) is not None:
                raise MemoryToolError("An identical memory already exists in the destination scope")
            _validate_target_scopes(scope, targets)
            operations.append(MemoryCuratorOperation(
                action="create",
                scope_type=scope.scope_type,
                scope_id=scope.scope_id,
                content=content,
                attributes=intent.attributes,
                override_targets=specs,
                semantic_action="create",
                operation_group_id=group_id,
            ))
            summaries.append(MemoryOperationSummary(
                operation_group_id=group_id,
                action="create",
                label=f"Create {scope.scope_type} memory",
                content=content,
                attributes=intent.attributes,
                destination_scope=scope,
                override_target_ids=list(target_ids or []),
            ))
            continue

        assert source is not None
        source_scope = MemoryToolScope(scope_type=source.scope_type, scope_id=source.scope_id)
        expected = iso_utc_z(source.updated_at or source.created_at)
        if intent.action == "delete":
            operations.append(MemoryCuratorOperation(
                action="delete",
                scope_type=source.scope_type,
                scope_id=source.scope_id,
                memory_id=source.id,
                expected_updated_at=expected,
                semantic_action="delete",
                operation_group_id=group_id,
            ))
            summaries.append(MemoryOperationSummary(
                operation_group_id=group_id,
                action="delete",
                label=f"Delete {source.scope_type} memory",
                content=source.content,
                source_memory_id=source.id,
                source_scope=source_scope,
                removed_incoming_override_count=await _incoming_override_count(source.id),
            ))
            continue

        if intent.action in {"update", "set_overrides"}:
            content = source.content if intent.action == "set_overrides" else str(intent.content or "").strip()
            if not content:
                raise MemoryToolError("Update requires non-empty memory content")
            if await _find_exact_duplicate(source_scope, content, exclude_memory_id=source.id) is not None:
                raise MemoryToolError("An identical memory already exists in this scope")
            _validate_target_scopes(source_scope, targets)
            operations.append(MemoryCuratorOperation(
                action="update",
                scope_type=source.scope_type,
                scope_id=source.scope_id,
                memory_id=source.id,
                expected_updated_at=expected,
                content=content,
                attributes=intent.attributes,
                override_targets=specs,
                semantic_action=intent.action,
                operation_group_id=group_id,
            ))
            summaries.append(MemoryOperationSummary(
                operation_group_id=group_id,
                action=intent.action,
                label="Update memory" if intent.action == "update" else "Update memory relationships",
                content=content,
                attributes=intent.attributes,
                source_memory_id=source.id,
                source_scope=source_scope,
                destination_memory_id=source.id,
                destination_scope=source_scope,
                override_target_ids=list(target_ids or []),
            ))
            continue

        destination_scope = _scope_for_type(context, intent.target_scope_type)
        if destination_scope == source_scope:
            raise MemoryToolError("Move destination must differ from the current memory scope")
        if source.id in (target_ids or []):
            raise MemoryToolError("A moved memory cannot override its source record")
        _validate_target_scopes(destination_scope, targets)
        duplicate = await _find_exact_duplicate(destination_scope, source.content)
        moved_attributes = intent.attributes or normalize_memory_attributes(source.attributes_json)
        if duplicate is not None:
            duplicate_expected = iso_utc_z(duplicate.updated_at or duplicate.created_at)
            operations.append(MemoryCuratorOperation(
                action="update",
                scope_type=duplicate.scope_type,
                scope_id=duplicate.scope_id,
                memory_id=duplicate.id,
                expected_updated_at=duplicate_expected,
                content=duplicate.content,
                attributes=moved_attributes,
                override_targets=specs,
                semantic_action="move",
                operation_group_id=group_id,
                move_source_memory_id=source.id,
                move_destination_memory_id=duplicate.id,
            ))
            destination_id = duplicate.id
        else:
            operations.append(MemoryCuratorOperation(
                action="create",
                scope_type=destination_scope.scope_type,
                scope_id=destination_scope.scope_id,
                content=source.content,
                attributes=moved_attributes,
                override_targets=specs,
                semantic_action="move",
                operation_group_id=group_id,
                move_source_memory_id=source.id,
            ))
            destination_id = None
        operations.append(MemoryCuratorOperation(
            action="delete",
            scope_type=source.scope_type,
            scope_id=source.scope_id,
            memory_id=source.id,
            expected_updated_at=expected,
            semantic_action="move",
            operation_group_id=group_id,
            move_source_memory_id=source.id,
            move_destination_memory_id=destination_id,
        ))
        summaries.append(MemoryOperationSummary(
            operation_group_id=group_id,
            action="move",
            label=f"Move memory from {source.scope_type} to {destination_scope.scope_type}",
            content=source.content,
            source_memory_id=source.id,
            source_scope=source_scope,
            destination_memory_id=destination_id,
            destination_scope=destination_scope,
            override_target_ids=list(target_ids or []),
            removed_incoming_override_count=await _incoming_override_count(source.id),
        ))

    await _validate_prepared_cycles(operations)
    return {
        "operations": [operation.model_dump(mode="json", exclude_none=True) for operation in operations],
        "operation_summaries": [summary.model_dump(mode="json", exclude_none=True) for summary in summaries],
    }


async def apply_confirmed_memory_change(req: MemoryCuratorApplyRequest) -> Dict[str, Any]:
    """Apply only a UI-confirmed canonical change set through the existing transaction."""

    context, _thread, _project = await build_memory_tool_context(
        selected_scope_type=req.context.selected_scope_type,
        selected_scope_id=req.context.selected_scope_id,
        thread_id=req.context.thread_id,
        project_id=req.context.project_id,
        capabilities=[MEMORY_APPLY_CONFIRMED],
    )
    _require_capability(context, MEMORY_APPLY_CONFIRMED)
    incoming_counts = {
        str(operation.memory_id): await _incoming_override_count(str(operation.memory_id))
        for operation in req.operations
        if operation.action == "delete" and operation.memory_id
    }
    from app.services.memory_curator_service import apply_memory_curator_change_set

    result = await apply_memory_curator_change_set(req)
    changed = {item["id"]: item for item in result.get("changed_memories", [])}
    warnings_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for warning in result.get("warnings", []):
        warnings_by_id.setdefault(str(warning.get("memory_id") or ""), []).append(warning)
    grouped: Dict[str, List[MemoryCuratorOperation]] = {}
    for operation in req.operations:
        grouped.setdefault(operation.operation_group_id or str(uuid.uuid4()), []).append(operation)
    receipts = []
    for group_id, operations in grouped.items():
        first = operations[0]
        semantic_action = first.semantic_action or first.action
        source_id = next((item.move_source_memory_id for item in operations if item.move_source_memory_id), None)
        source_operation = next((item for item in operations if item.memory_id == source_id), None)
        result_record = next(
            (changed[item.memory_id] for item in operations if item.memory_id in changed),
            None,
        )
        if result_record is None:
            created_candidates = [item for item in result.get("changed_memories", []) if any(
                operation.action == "create"
                and operation.scope_type == item.get("scope_type")
                and operation.scope_id == item.get("scope_id")
                and operation.content == item.get("content")
                for operation in operations
            )]
            result_record = created_candidates[0] if created_candidates else None
        destination_scope = None
        if result_record:
            destination_scope = {
                "scope_type": result_record["scope_type"],
                "scope_id": result_record["scope_id"],
            }
        receipts.append({
            "operation_group_id": group_id,
            "action": semantic_action,
            "source_memory_id": source_id or (first.memory_id if first.action == "delete" else None),
            "result_memory_id": result_record.get("id") if result_record else first.move_destination_memory_id,
            "source_scope": (
                {"scope_type": source_operation.scope_type, "scope_id": source_operation.scope_id}
                if source_operation
                else (
                    {"scope_type": first.scope_type, "scope_id": first.scope_id}
                    if first.action == "delete" else None
                )
            ),
            "destination_scope": destination_scope,
            "deleted_memory_ids": [
                str(item.memory_id) for item in operations
                if item.action == "delete" and item.memory_id in result.get("deleted_memory_ids", [])
            ],
            "override_target_ids": [target.memory_id for target in first.override_targets],
            "removed_incoming_override_count": sum(
                incoming_counts.get(str(item.memory_id), 0)
                for item in operations if item.action == "delete"
            ),
            "index_status": result_record.get("index_status") if result_record else None,
            "warnings": warnings_by_id.get(str(result_record.get("id") if result_record else ""), []),
        })
    return {**result, "receipts": receipts}
