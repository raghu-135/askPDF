from app.models.memory_manager_budget import (
    HARD_MAX_CANONICAL_OPERATIONS,
    HARD_MAX_CANDIDATE_GROUPS,
    compute_memory_manager_budget,
    operation_cost,
    pack_candidate_groups,
)


def test_memory_manager_budget_scales_with_context_window():
    small = compute_memory_manager_budget(2048, "consistency_review")
    large = compute_memory_manager_budget(40000, "consistency_review")

    assert large.evidence_chars > small.evidence_chars
    assert large.plan_chars > small.plan_chars
    assert large.max_candidate_groups >= small.max_candidate_groups
    assert large.max_canonical_operations >= small.max_canonical_operations
    assert large.max_candidate_groups <= HARD_MAX_CANDIDATE_GROUPS
    assert large.max_canonical_operations <= HARD_MAX_CANONICAL_OPERATIONS


def test_candidate_groups_are_packed_as_complete_groups():
    budget = compute_memory_manager_budget(2048, "consistency_review")
    groups = [{"id": index, "memories": [{"content": "x" * 400}]} for index in range(20)]
    packed = pack_candidate_groups(groups, budget)

    assert packed
    assert len(packed) <= budget.max_candidate_groups
    assert [item["id"] for item in packed] == list(range(len(packed)))


def test_weighted_operation_cost_reflects_relationship_and_merge_size():
    basic = operation_cost({"type": "memory_update", "content": "short"})
    relationship = operation_cost({"type": "relationship_replace", "override_target_ids": ["a", "b"]})
    merge = operation_cost({"type": "memory_merge", "content": "short"})

    assert relationship > basic
    assert merge > basic
