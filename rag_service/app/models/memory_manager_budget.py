"""Context-aware budgets for unified memory planning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

from app.models.llm_server_client import CHARS_PER_TOKEN


HARD_MAX_CANDIDATE_GROUPS = 20
HARD_MAX_CANONICAL_OPERATIONS = 20
HARD_MAX_RELATIONSHIP_TARGETS = 20
MIN_CANDIDATE_GROUPS = 1
DEFAULT_ESTIMATED_GROUP_CHARS = 1800
DEFAULT_ESTIMATED_OPERATION_CHARS = 900


@dataclass(frozen=True)
class MemoryManagerBudget:
    usable_context_chars: int
    evidence_chars: int
    candidate_group_chars: int
    analysis_chars: int
    plan_chars: int
    operation_budget_units: int
    max_candidate_groups: int
    max_canonical_operations: int


def _serialized_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")))


def operation_cost(operation: dict[str, Any]) -> int:
    """Estimate weighted planning cost; relationships and moves cost more."""

    kind = str(operation.get("type") or operation.get("action") or "")
    content_cost = max(0, len(str(operation.get("content") or "")) // 900)
    target_cost = len(operation.get("override_target_ids") or operation.get("override_targets") or [])
    if kind in {"memory_move", "move"}:
        return 3 + content_cost + target_cost
    if kind in {"memory_merge", "merge"}:
        return 4 + content_cost + target_cost
    if kind in {"relationship_replace", "set_overrides"}:
        return 1 + target_cost
    return 1 + content_cost + (target_cost // 4)


def compute_memory_manager_budget(
    context_window: int,
    mode: str = "direct_edit",
    review_round: int = 1,
    *,
    candidate_groups: Iterable[dict[str, Any]] | None = None,
) -> MemoryManagerBudget:
    """Allocate evidence and plan space while retaining hard safety ceilings."""

    usable = int(max(256, context_window) * 0.70 * CHARS_PER_TOKEN)
    if mode == "conversation_extract":
        evidence_ratio = 0.42
        plan_ratio = 0.18
    elif mode == "consistency_review":
        evidence_ratio = 0.46
        plan_ratio = 0.20
    else:
        evidence_ratio = 0.30
        plan_ratio = 0.22
    round_factor = max(0.75, min(1.0, 1.0 - max(0, review_round - 1) * 0.05))
    evidence = max(1800, int(usable * evidence_ratio * round_factor))
    plan_chars = max(1600, int(usable * plan_ratio))
    analysis = max(1600, int(usable * 0.14))
    group_chars = max(1200, int(evidence * 0.82))
    group_values = list(candidate_groups or [])
    average_group = (
        max(600, sum(_serialized_size(item) for item in group_values) // len(group_values))
        if group_values else DEFAULT_ESTIMATED_GROUP_CHARS
    )
    max_groups = min(
        HARD_MAX_CANDIDATE_GROUPS,
        max(MIN_CANDIDATE_GROUPS, group_chars // average_group),
    )
    operation_units = max(1, plan_chars // DEFAULT_ESTIMATED_OPERATION_CHARS)
    max_operations = min(HARD_MAX_CANONICAL_OPERATIONS, max(1, operation_units))
    return MemoryManagerBudget(
        usable_context_chars=usable,
        evidence_chars=evidence,
        candidate_group_chars=group_chars,
        analysis_chars=analysis,
        plan_chars=plan_chars,
        operation_budget_units=operation_units,
        max_candidate_groups=max_groups,
        max_canonical_operations=max_operations,
    )


def pack_candidate_groups(
    groups: Iterable[dict[str, Any]],
    budget: MemoryManagerBudget,
) -> list[dict[str, Any]]:
    """Pack complete candidate groups without splitting a group."""

    packed: list[dict[str, Any]] = []
    used = 0
    for group in groups:
        if len(packed) >= budget.max_candidate_groups:
            break
        size = _serialized_size(group)
        if packed and used + size > budget.candidate_group_chars:
            break
        packed.append(group)
        used += size
    return packed


def bounded_operation_count(operations: Iterable[dict[str, Any]], budget: MemoryManagerBudget) -> int:
    """Return the prefix that fits both weighted and hard operation budgets."""

    total = 0
    count = 0
    for operation in operations:
        cost = operation_cost(operation)
        if count >= budget.max_canonical_operations or total + cost > budget.operation_budget_units:
            break
        total += cost
        count += 1
    return count
