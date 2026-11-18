from __future__ import annotations

from types import SimpleNamespace

from app.api.agent_workflows import _run_payload
from app.product_orchestration.delegation_context import inherited_context_from_run_metadata


def test_inherited_context_from_run_metadata():
    meta = {
        "inherited_context_window": 12000,
        "inherited_context_budget": {"document_ratio": 0.3, "usable_ratio": 0.8},
    }
    payload = inherited_context_from_run_metadata(meta)
    assert payload == {
        "context_window": 12000,
        "context_budget": {"document_ratio": 0.3, "usable_ratio": 0.8},
    }


def test_run_payload_exposes_inherited_context():
    run = SimpleNamespace(
        id="child-1",
        thread_id="thread-1",
        user_id=None,
        workflow_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        definition_category=None,
        task_id=None,
        parent_run_id="parent-1",
        task_attempt=None,
        resolved_spec_json={},
        status="completed",
        runtime_binding_status="active",
        pending_interrupt_json=None,
        started_at=None,
        completed_at=None,
        error_json=None,
        metrics_json={},
        run_metadata_json={
            "inherited_context_window": 8192,
            "inherited_context_budget": {"recent_ratio": 0.2},
        },
        debug_trace_json={},
    )
    payload = _run_payload(run, turns=[])
    assert payload["inherited_context"] == {
        "context_window": 8192,
        "context_budget": {"recent_ratio": 0.2},
    }