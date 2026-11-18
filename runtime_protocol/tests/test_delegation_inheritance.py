from __future__ import annotations

from runtime_protocol.delegated_hitl import child_workflow_gate_ids, merge_child_delegate_hitl_policy
from runtime_protocol.delegation_inheritance import (
    append_delegated_child_run,
    delegate_inherited_hitl_policy,
    delegated_child_runs_from_technical,
    delegated_child_runs_technical_details,
    merge_delegate_inherited_tool_settings,
    pick_delegate_inherited_tool_settings,
)


def test_delegate_inherited_hitl_policy_keeps_only_child_shared_gates():
    parent = {
        "enabled": True,
        "gates": {
            "plan_approval_gate": {"target": {"node_id": "workflow_delegate"}},
            "task_pause_gate": {"enabled": True},
        },
        "tools": {"search_web": {"mode": "ask", "scope": "run"}},
        "max_interrupts_per_run": 8,
    }
    inherited = delegate_inherited_hitl_policy(parent, child_gate_ids=frozenset({"task_pause_gate"}))
    assert "plan_approval_gate" not in (inherited or {}).get("gates", {})
    assert inherited["gates"]["task_pause_gate"]["enabled"] is True


def test_merge_child_delegate_hitl_policy_preserves_child_gates():
    child = {
        "enabled": True,
        "gates": {"task_pause_gate": {"enabled": True, "mode": "approval"}},
        "max_interrupts_per_run": 20,
    }
    parent_tools = {"tools": {"search_web": {"mode": "ask", "scope": "run"}}, "max_interrupts_per_run": 8}
    merged = merge_child_delegate_hitl_policy(child, parent_tools, child_gate_ids=frozenset({"task_pause_gate"}))
    assert merged["gates"]["task_pause_gate"]["enabled"] is True
    assert merged["tools"]["search_web"]["mode"] == "ask"
    assert merged["max_interrupts_per_run"] == 8


def test_child_workflow_gate_ids_reads_builtin_shape():
    spec = {"config": {"hitl_policy": {"gates": {"plan_approval_gate": {}, "task_pause_gate": {}}}}}
    assert child_workflow_gate_ids(spec) == frozenset({"plan_approval_gate", "task_pause_gate"})


def test_delegate_inherited_hitl_policy_strips_parent_graph_gates():
    parent = {
        "enabled": True,
        "gates": {
            "plan_approval_gate": {
                "target": {"node_id": "workflow_delegate", "node_type": "workflow_delegate"},
                "routes": {"approve": "workflow_delegate"},
            },
        },
        "tools": {"search_web": {"mode": "ask", "scope": "run"}},
        "max_interrupts_per_run": 8,
    }
    inherited = delegate_inherited_hitl_policy(parent)
    assert "gates" not in (inherited or {})
    assert inherited["tools"]["search_web"]["mode"] == "ask"
    picked = pick_delegate_inherited_tool_settings({"hitl_policy": parent, "web_search_mode": "ask"})
    assert "gates" not in picked.get("hitl_policy", {})


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