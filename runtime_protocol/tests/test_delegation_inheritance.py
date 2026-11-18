from __future__ import annotations

from runtime_protocol.delegation_inheritance import (
    append_delegated_child_run,
    delegated_child_runs_from_technical,
    delegated_child_runs_technical_details,
    merge_delegate_inherited_tool_settings,
)


def test_merge_delegate_inherited_tool_settings_prefers_ask_web_mode():
    merged = merge_delegate_inherited_tool_settings(
        {"web_search_mode": "ask"},
        {"use_web_search": True, "hitl_web_approval": False},
    )
    assert merged["web_search_mode"] == "ask"
    assert merged["use_web_search"] is True
    assert merged["hitl_web_approval"] is True


def test_append_delegated_child_run_keeps_order_for_replans():
    rows = append_delegated_child_run([], run_id="child-1", workflow_id="router_rag_agent")
    rows = append_delegated_child_run(rows, run_id="child-2", workflow_id="router_rag_agent")
    rows = append_delegated_child_run(rows, run_id="child-1", workflow_id="router_rag_agent")
    assert [row["run_id"] for row in rows] == ["child-1", "child-2"]
    assert rows[1]["order"] == 2


def test_delegated_child_runs_from_technical_supports_legacy_single_child():
    rows = delegated_child_runs_from_technical({"child_run_id": "legacy", "child_workflow_id": "router_rag_agent"})
    assert rows == [{"run_id": "legacy", "workflow_id": "router_rag_agent", "order": 1}]


def test_delegated_child_runs_technical_details_emits_list_and_latest_ids():
    details = delegated_child_runs_technical_details(
        {"delegated_child_runs": [{"run_id": "a", "workflow_id": "w1", "order": 1}]},
        latest_run_id="b",
        latest_workflow_id="w2",
    )
    assert len(details["delegated_child_runs"]) == 2
    assert details["child_run_id"] == "b"
    assert details["child_workflow_id"] == "w2"