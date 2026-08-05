"""Unified, browser-held memory planning facade."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import OrderedDict
from typing import Any, Dict, Iterable, List

from app.models.memory_manager_budget import (
    compute_memory_manager_budget,
    pack_candidate_groups,
    operation_cost,
)
from app.models.requests import (
    MemoryConsistencyReviewCursor,
    MemoryChangeApplyRequest,
    MemoryManagerContext,
    MemoryManagerMessage,
    MemoryChangeOperation,
    MemoryManagerConversationRequest,
    MemoryReviewCursor,
    MemoryManagerApplyRequest,
    MemoryManagerOperation,
    MemoryManagerPlan,
    MemoryManagerPlanRequest,
)
from app.services.memory_manager_engine import (
    apply_memory_change_set,
    respond_to_memory_manager,
)
from app.services.memory_policy import LOCAL_USER_MEMORY_SCOPE_ID
from app.services.memory_review_service import get_memory_review_status


_APPLIED_RESULTS: OrderedDict[str, Dict[str, Any]] = OrderedDict()
_MAX_APPLIED_RESULTS = 512


def _review_context(req: MemoryManagerPlanRequest) -> tuple[str, str] | None:
    if req.mode != "consistency_review":
        return None
    if req.context.thread_id:
        return "thread", req.context.thread_id
    if req.context.project_id:
        return "project", req.context.project_id
    if req.context.selected_scope_type == "user":
        return "user", LOCAL_USER_MEMORY_SCOPE_ID
    return None


async def _scope_versions(req: MemoryManagerPlanRequest) -> Dict[str, int]:
    context = _review_context(req)
    if context is None:
        return {}
    status = await get_memory_review_status(*context)
    return {str(key): int(value) for key, value in (status.get("current_scope_versions") or {}).items()}


def _canonical_plan_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonical_plan_value(item)
            for key, item in sorted(value.items())
            if item is not None
        }
    if isinstance(value, list):
        return [_canonical_plan_value(item) for item in value]
    return value


def _stable_plan_hash(plan: Dict[str, Any]) -> str:
    payload = _canonical_plan_value(dict(plan))
    payload.pop("plan_hash", None)
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _manager_type(operation: Dict[str, Any]) -> str:
    semantic = str(operation.get("semantic_action") or "")
    action = str(operation.get("action") or "")
    if action == "update" and semantic == "set_overrides":
        return "relationship_replace"
    return {
        "create": "memory_create",
        "update": "memory_update",
        "delete": "memory_delete",
    }.get(action, "memory_update")


def _to_manager_operation(operation: Dict[str, Any]) -> MemoryManagerOperation:
    op_type = _manager_type(operation)
    targets = operation.get("override_targets") or []
    target_ids = [str(item.get("memory_id")) for item in targets if isinstance(item, dict) and item.get("memory_id")]
    target_versions = {
        str(item.get("memory_id")): str(item.get("expected_updated_at"))
        for item in targets
        if isinstance(item, dict) and item.get("memory_id") and item.get("expected_updated_at")
    }
    return MemoryManagerOperation(
        type=op_type,
        memory_id=operation.get("memory_id"),
        source_memory_id=operation.get("move_source_memory_id"),
        destination_memory_id=operation.get("move_destination_memory_id"),
        scope_type=operation.get("scope_type"),
        scope_id=operation.get("scope_id"),
        content=operation.get("content"),
        attributes=operation.get("attributes"),
        override_target_ids=target_ids,
        override_target_versions=target_versions,
        expected_updated_at=operation.get("expected_updated_at"),
        operation_group_id=operation.get("operation_group_id"),
    )


def _to_curator_operation(operation: MemoryManagerOperation) -> MemoryChangeOperation:
    action = {
        "memory_create": "create",
        "memory_update": "update",
        "memory_delete": "delete",
        "relationship_replace": "update",
        "memory_move": "update",
        "memory_merge": "update",
    }[operation.type]
    semantic = {
        "relationship_replace": "set_overrides",
        "memory_move": "move",
        "memory_merge": "move",
    }.get(operation.type)
    return MemoryChangeOperation(
        action=action,
        scope_type=operation.scope_type,
        scope_id=operation.scope_id,
        memory_id=operation.memory_id,
        expected_updated_at=operation.expected_updated_at,
        content=operation.content,
        attributes=operation.attributes,
        override_targets=[
            {"memory_id": memory_id, "expected_updated_at": operation.override_target_versions.get(memory_id, "")}
            for memory_id in operation.override_target_ids
        ],
        semantic_action=semantic or action,
        operation_group_id=operation.operation_group_id,
        move_source_memory_id=operation.source_memory_id,
        move_destination_memory_id=operation.destination_memory_id,
    )


def _assign_stable_create_ids(operations: List[MemoryManagerOperation], plan_id: str) -> List[MemoryManagerOperation]:
    assigned = []
    for index, operation in enumerate(operations):
        if operation.type == "memory_create" and not operation.memory_id:
            operation = operation.model_copy(update={"memory_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"askpdf:{plan_id}:{index}"))})
        assigned.append(operation)
    return assigned


def _bounded_operations(operations: List[MemoryManagerOperation], budget) -> List[MemoryManagerOperation]:
    grouped: OrderedDict[str, List[MemoryManagerOperation]] = OrderedDict()
    for index, operation in enumerate(operations):
        group_id = operation.operation_group_id or f"operation-{index}"
        grouped.setdefault(group_id, []).append(operation)
    selected: List[MemoryManagerOperation] = []
    used_units = 0
    for group in grouped.values():
        raw = [item.model_dump(mode="json", exclude_none=True) for item in group]
        group_cost = sum(operation_cost(item) for item in raw)
        if selected and (
            len(selected) + len(group) > budget.max_canonical_operations
            or used_units + group_cost > budget.operation_budget_units
        ):
            break
        if len(group) > budget.max_canonical_operations:
            continue
        selected.extend(group)
        used_units += group_cost
    return selected


def _validate_mode_operations(mode: str, operations: Iterable[MemoryManagerOperation]) -> None:
    if mode != "conversation_extract":
        return
    for operation in operations:
        if operation.type != "memory_create" or operation.scope_type != "thread":
            raise ValueError("Conversation extraction can only create Thread memories")


def _plan_errors(mode: str, operations: List[MemoryManagerOperation]) -> List[str]:
    errors: List[str] = []
    seen_sources: set[str] = set()
    for operation in operations:
        if operation.type in {"memory_update", "memory_delete", "relationship_replace", "memory_move", "memory_merge"}:
            if not operation.memory_id:
                errors.append(f"{operation.type} requires memory_id")
            elif operation.memory_id in seen_sources:
                errors.append(f"memory appears in multiple operations: {operation.memory_id}")
            seen_sources.add(operation.memory_id)
        if len(operation.override_target_ids) > 20:
            errors.append("relationship target limit exceeded")
        if operation.type in {"memory_move", "memory_merge"}:
            errors.append(f"{operation.type} must be normalized before apply")
    try:
        _validate_mode_operations(mode, operations)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


async def create_memory_manager_plan(req: MemoryManagerPlanRequest) -> Dict[str, Any]:
    mode_map = {
        "direct_edit": "edit",
        "conversation_extract": "conversation_review",
        "consistency_review": "memory_review",
    }
    review_id = req.review_id or (str(uuid.uuid4()) if req.mode == "consistency_review" else None)
    old_req = MemoryManagerConversationRequest(
        mode=mode_map[req.mode],
        context=req.context,
        memory_id=req.memory_id,
        messages=req.messages,
        llm_model=req.llm_model,
        context_window=req.context_window,
        web_search_mode=req.web_search_mode,
        web_search_decision=req.web_search_decision,
        memory_review_cursor=req.memory_review_cursor,
    )
    result = await respond_to_memory_manager(old_req)
    raw_operations = list(result.get("operations") or [])
    operations = [_to_manager_operation(item) for item in raw_operations]
    plan_id = str(uuid.uuid4())
    operations = _assign_stable_create_ids(operations, plan_id)
    review_groups = list((result.get("memory_review") or {}).get("candidate_groups") or [])
    budget = compute_memory_manager_budget(
        req.context_window,
        req.mode,
        req.review_round,
        candidate_groups=review_groups,
    )
    packed_groups = pack_candidate_groups(review_groups, budget)
    operations = _bounded_operations(operations, budget)
    errors = _plan_errors(req.mode, operations)
    replan_count = 0
    previous_signature: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    while errors and replan_count < 2:
        signature = (
            tuple(sorted(errors)),
            tuple(json.dumps(item.model_dump(mode="json", exclude_none=True), sort_keys=True) for item in operations),
        )
        if signature == previous_signature:
            break
        previous_signature = signature
        replan_count += 1
        repair_message = MemoryManagerMessage(
            role="user",
            content=(
                "Repair the proposed memory plan using the supplied evidence. "
                "Return one complete concrete proposal and do not ask the same question again. "
                f"Validation errors: {'; '.join(errors)}"
            ),
        )
        repair_req = old_req.model_copy(update={"messages": [*old_req.messages, repair_message]})
        result = await respond_to_memory_manager(repair_req)
        operations = [_to_manager_operation(item) for item in (result.get("operations") or [])]
        operations = _assign_stable_create_ids(operations, plan_id)
        operations = _bounded_operations(operations, budget)
        errors = _plan_errors(req.mode, operations)
    if errors:
        result = {
            **result,
            "state": "blocked",
            "message": "The proposed memory plan could not be validated after bounded replanning.",
            "choices": [],
        }
    result_state = str(result.get("state") or "no_changes")
    if result_state == "conflict":
        result_state = "clarification"
    versions = await _scope_versions(req)
    next_cursor = None
    if result.get("review"):
        next_cursor = (result["review"] or {}).get("cursor")
    elif result.get("memory_review"):
        review_payload = result["memory_review"] or {}
        next_cursor = {
            key: review_payload[key]
            for key in (
                "context_type",
                "context_id",
                "snapshot_at",
                "snapshot_scope_versions",
                "anchor_position",
                "reviewed_anchor_count",
                "remaining_anchor_count",
            )
            if key in review_payload
        }
    plan_data = {
        "plan_id": plan_id,
        "mode": req.mode,
        "context": req.context.model_dump(mode="json"),
        "state": "blocked" if errors else ("proposal" if operations else result_state),
        "message": result.get("message") or "",
        "choices": result.get("choices") or [],
        "embedding_readiness": result.get("embedding_readiness") or [],
        "pending_web_search": result.get("pending_web_search"),
        "web_sources": result.get("web_sources") or [],
        "consent": result.get("consent"),
        "operations": [item.model_dump(mode="json", exclude_none=True) for item in operations],
        "analysis": result.get("operation_summaries") or [],
        "review": result.get("review"),
        "memory_review": {
            **(result.get("memory_review") or {}),
            "candidate_groups": packed_groups,
        } if result.get("memory_review") is not None else None,
        "budget": {
            "usable_context_chars": budget.usable_context_chars,
            "evidence_chars": budget.evidence_chars,
            "candidate_group_chars": budget.candidate_group_chars,
            "analysis_chars": budget.analysis_chars,
            "plan_chars": budget.plan_chars,
            "operation_budget_units": budget.operation_budget_units,
            "max_candidate_groups": budget.max_candidate_groups,
            "max_canonical_operations": budget.max_canonical_operations,
        },
        "review_id": review_id,
        "next_cursor": next_cursor,
        "scope_versions": versions,
    }
    plan_data["plan_hash"] = _stable_plan_hash(plan_data)
    return MemoryManagerPlan.model_validate(plan_data).model_dump(mode="json")


async def apply_memory_manager_plan(req: MemoryManagerApplyRequest) -> Dict[str, Any]:
    if req.plan_hash != req.plan.plan_hash or req.plan_hash != _stable_plan_hash(req.plan.model_dump(mode="json")):
        raise ValueError("Memory manager plan hash is invalid")
    cached = _APPLIED_RESULTS.get(req.idempotency_key)
    if cached is not None:
        return cached
    current_req = MemoryManagerPlanRequest(
        mode=req.plan.mode,
        context=req.plan.context,
        llm_model="unused",
        review_id=req.plan.review_id,
        memory_review_cursor=MemoryConsistencyReviewCursor.model_validate(req.plan.next_cursor)
        if req.plan.mode == "consistency_review" and req.plan.next_cursor and "context_type" in req.plan.next_cursor else None,
    )
    current_versions = await _scope_versions(current_req)
    if req.plan.scope_versions and any(
        int(current_versions.get(key, 0)) != int(value)
        for key, value in req.plan.scope_versions.items()
    ):
        raise ValueError("Memory manager plan is stale; start a new review round")
    curator_req = MemoryChangeApplyRequest(
        context=req.plan.context,
        confirmed=True,
        operations=[_to_curator_operation(item) for item in req.plan.operations],
        review_cursor=(
            MemoryReviewCursor.model_validate(req.plan.next_cursor)
            if req.plan.mode == "conversation_extract" and req.plan.next_cursor else None
        ),
        memory_review_cursor=(
            MemoryConsistencyReviewCursor.model_validate(req.plan.next_cursor)
            if req.plan.mode == "consistency_review" and req.plan.next_cursor and "context_type" in req.plan.next_cursor else None
        ),
        actor_id=req.actor_id,
    )
    result = await apply_memory_change_set(curator_req)
    response = {
        **result,
        "plan_id": req.plan.plan_id,
        "plan_hash": req.plan.plan_hash,
        "idempotency_key": req.idempotency_key,
        "status": "committed" if not result.get("warnings") else "indexing_pending",
        "review_id": req.plan.review_id,
    }
    _APPLIED_RESULTS[req.idempotency_key] = response
    _APPLIED_RESULTS.move_to_end(req.idempotency_key)
    while len(_APPLIED_RESULTS) > _MAX_APPLIED_RESULTS:
        _APPLIED_RESULTS.popitem(last=False)
    return response
