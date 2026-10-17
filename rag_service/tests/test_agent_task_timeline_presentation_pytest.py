from app.services.agent_task_presentation import followup_guidance_entries, timeline_sort_key


def test_timeline_sort_key_orders_guidance_after_objective_before_plan():
    objective = {"timestamp": "2026-01-01T00:00:00Z", "type": "objective", "id": "objective:1"}
    guidance = {"timestamp": "2026-01-01T00:00:00Z", "type": "research_guidance", "id": "guidance:1", "sort_order": 15}
    plan = {"timestamp": "2026-01-01T00:00:00Z", "type": "plan", "id": "plan:1"}
    ordered = sorted([guidance, plan, objective], key=timeline_sort_key)
    assert [item["type"] for item in ordered] == ["objective", "research_guidance", "plan"]


def test_followup_guidance_entries_matches_parent_run():
    run = type("Run", (), {"id": "run-2", "parent_run_id": "run-1"})()
    config = {
        "result_review_context": [
            {"source_run_id": "run-1", "followup_input": "Skip web benchmarks.", "idempotency_key": "k1"},
        ],
    }
    entries = followup_guidance_entries(config, run)
    assert entries == [{"followup": "Skip web benchmarks.", "key": "k1"}]
