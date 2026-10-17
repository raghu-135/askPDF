import pytest

from app.services.agent_task_presentation import followup_guidance_entries, timeline_sort_key


def test_timeline_sort_key_orders_guidance_after_objective_before_plan():
    objective = {"timestamp": "2026-01-01T00:00:00Z", "type": "objective", "id": "objective:1"}
    guidance = {"timestamp": "2026-01-01T00:00:00Z", "type": "research_guidance", "id": "guidance:1", "sort_order": 15}
    plan = {"timestamp": "2026-01-01T00:00:00Z", "type": "plan", "id": "plan:1"}
    ordered = sorted([guidance, plan, objective], key=timeline_sort_key)
    assert [item["type"] for item in ordered] == ["objective", "research_guidance", "plan"]


def test_followup_guidance_entries_matches_parent_run():
    run = type("Run", (), {"id": "run-2", "parent_run_id": "run-1", "run_metadata_json": {}})()
    config = {
        "result_review_context": [
            {"source_run_id": "run-1", "followup_input": "Skip web benchmarks.", "idempotency_key": "k1"},
        ],
    }
    entries = followup_guidance_entries(config, run)
    assert entries == [{"followup": "Skip web benchmarks.", "key": "k1"}]


def test_followup_guidance_entries_matches_result_review_run_metadata():
    run = type(
        "Run",
        (),
        {
            "id": "run-2",
            "parent_run_id": None,
            "run_metadata_json": {
                "result_review_source_run_id": "run-1",
                "result_review_idempotency_key": "k1",
            },
        },
    )()
    config = {
        "result_review_context": [
            {"source_run_id": "run-1", "followup_input": "Skip web benchmarks.", "idempotency_key": "k1"},
        ],
    }
    entries = followup_guidance_entries(config, run)
    assert entries == [{"followup": "Skip web benchmarks.", "key": "k1"}]


@pytest.mark.asyncio
async def test_timeline_report_markdown_prefers_final_body_and_strips_appendix(monkeypatch):
    from app.services.agent_task_presentation import timeline_report_markdown

    run = type(
        "Run",
        (),
        {
            "id": "run-1",
            "pending_interrupt_json": {
                "provisional_answer": "Thin summary only.\n\n## Unresolved required work\n- todo-1",
            },
        },
    )()
    artifact = type(
        "Artifact",
        (),
        {
            "kind": "final_report",
            "validity": "valid",
            "agent_run_id": "run-1",
            "object_key": "reports/final.md",
            "summary_json": {"text": "Meta summary"},
        },
    )()

    class _Store:
        async def read(self, _key: str) -> bytes:
            return b"Full final report.\n\n## Unresolved required work\n- todo-1"

    monkeypatch.setattr(
        "app.services.content_store.get_content_store",
        lambda: _Store(),
    )

    markdown = await timeline_report_markdown(run, [artifact])
    assert markdown == "Full final report."
