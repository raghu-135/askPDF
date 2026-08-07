import json
import ast
from pathlib import Path

from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.agent_workflows.corrective_contracts import (
    DEFAULT_CORRECTIVE_POLICY,
    collect_corrective_policy_errors,
    normalized_corrective_policy,
)
from app.agent_workflows.corrective_nodes import (
    corrective_route_for_report,
    grounded_route_for_report,
    normalize_grounding_report,
    normalize_retrieval_quality_report,
)
from app.agent_workflows.evidence import canonical_source_id, normalized_source_url
from app.agent_workflows.parallel_runtime import aggregate_parallel_results, normalize_work_items
from app.agent_workflows.validator import WorkflowValidator


def _packet(packet_id="p1", source_id="doc:file:1"):
    return {"id": packet_id, "kind": "document", "content": "Evidence", "source_ids": [source_id]}


def _corrective_state(**overrides):
    state = {
        "workflow_id": "corrective_self_rag_agent",
        "agent_run_id": "run-1",
        "question": "What happened?",
        "corrective_policy": dict(DEFAULT_CORRECTIVE_POLICY),
        "corrective_wave": 0,
        "evidence_packets": [_packet()],
        "evidence_gaps": [],
        "worker_result_packets": [],
        "parallel_attempt_records": [],
    }
    state.update(overrides)
    return state


def test_corrective_policy_normalizes_bounds_and_rejects_unknown_keys():
    policy = normalized_corrective_policy({"minimum_relevance_confidence": 2, "max_total_work_items": 10, "max_total_tool_attempts": 2})
    assert policy["minimum_relevance_confidence"] == 1.0
    assert policy["max_total_tool_attempts"] == 10
    errors = collect_corrective_policy_errors({"unknown": True, "max_corrective_waves": 0})
    assert any("unknown keys" in error for error in errors)
    assert any("max_corrective_waves" in error for error in errors)


def test_canonical_source_ids_are_stable_and_urls_drop_fragments():
    assert normalized_source_url("HTTPS://Example.COM/a//b/#section") == "https://example.com/a/b"
    assert canonical_source_id({"file_hash": "abc", "page_start": 7}) == "doc:abc:7"
    assert canonical_source_id({"url": "https://example.com/a#one"}) == "web:https://example.com/a"
    assert canonical_source_id({"message_id": "m1"}) == "conversation:m1"
    assert canonical_source_id({"memory_id": "mem1"}) == "memory:mem1"


def test_retrieval_grader_fails_closed_for_unknown_and_missing_packet_ids():
    report = normalize_retrieval_quality_report({
        "packet_assessments": [{"packet_id": "invented", "relevant": True, "confidence": 1, "provenance_complete": True}],
        "missing_requirements": [],
        "material_contradictions": [],
    }, _corrective_state())
    assert report["verdict"] == "incorrect"
    assert report["unknown_packet_ids"] == ["invented"]
    assert report["packet_assessments"][0]["confidence"] == 0


def test_retrieval_route_is_correct_only_for_eligible_confident_evidence():
    state = _corrective_state()
    report = normalize_retrieval_quality_report({
        "packet_assessments": [{
            "packet_id": "p1", "relevant": True, "confidence": 0.8,
            "provenance_complete": True, "instruction_injection_risk": False,
            "coverage": [], "contradiction_signals": [],
        }],
        "missing_requirements": [], "material_contradictions": [], "reason": "covered",
    }, state)
    assert report["verdict"] == "correct"
    assert corrective_route_for_report(report, state) == ("synthesize", "")
    assert corrective_route_for_report({"verdict": "ambiguous"}, {**state, "corrective_wave": 2}) == ("insufficient", "max_corrective_waves")


def test_grounding_unknown_citations_fail_closed_and_verified_claims_pass():
    state = _corrective_state()
    invalid = normalize_grounding_report({
        "claims": [{"claim": "A", "support": "full", "source_ids": ["made-up"]}],
        "citation_violations": [], "contradictions": [], "unresolved_gaps": [], "usefulness_score": 5,
    }, state)
    assert invalid["supported_claim_ratio"] == 0
    assert invalid["citation_violations"]
    assert grounded_route_for_report(invalid, state)[0] == "correct"

    valid = normalize_grounding_report({
        "claims": [{"claim": "A", "support": "full", "source_ids": ["doc:file:1"]}],
        "citation_violations": [], "contradictions": [], "unresolved_gaps": [], "usefulness_score": 3,
    }, state)
    assert valid["verified_claims"][0]["claim"] == "A"
    assert grounded_route_for_report(valid, state) == ("pass", "")


def test_focused_work_item_requires_linked_file_hash_and_respects_source_policy():
    available = [
        {"id": "retrieval", "type": "retrieval_worker"},
        {"id": "memory", "type": "durable_memory_worker"},
        {"id": "web", "type": "web_worker"},
    ]
    state = _corrective_state(
        available_worker_nodes=available,
        pre_fetch_bundle={"documents": [{"file_hash": "owned"}]},
        use_web_search=True,
        corrective_policy={**DEFAULT_CORRECTIVE_POLICY, "allow_web_fallback": False, "memory_evidence_mode": "disabled"},
        parallel_policy={"max_work_items": 4},
    )
    items = normalize_work_items([
        {"worker_node_id": "retrieval", "query": "q", "file_hash": "unowned"},
        {"worker_node_id": "retrieval", "query": "q", "file_hash": "owned"},
        {"worker_node_id": "memory", "query": "q"},
        {"worker_node_id": "web", "query": "q"},
    ], state=state, dispatch_node_id="dispatch", dispatch_visit=1)
    assert [(item["worker_type"], item["file_hash"]) for item in items] == [("retrieval_worker", "owned")]


def test_aggregation_filters_prior_wave_reducer_deltas_by_dispatch():
    state = _corrective_state(
        dispatch_id="d2",
        dispatch_started_epoch_ms=0,
        parallel_policy={"continue_on_insufficient_successes": True},
        work_items=[{"dispatch_id": "d2", "work_id": "w2", "ordinal": 0}],
        worker_result_packets=[{
            "dispatch_id": "d2", "work_id": "w2", "ordinal": 0, "status": "completed", "attempt": 1,
            "evidence_packets": [], "document_sources": [], "web_sources": [], "chat_ids": [], "memory_refs": [],
            "node_events": [], "tool_events": [], "errors": [],
        }],
        parallel_evidence_deltas=[
            {**_packet("old"), "dispatch_id": "d1"},
            {**_packet("new", "doc:file:2"), "content": "New evidence", "dispatch_id": "d2"},
        ],
    )
    result = aggregate_parallel_results(state)
    assert [item["id"] for item in result["evidence_packets"]] == ["p1", "new"]


def test_corrective_builtin_validates_without_changing_existing_builtins():
    workflows = {item["builtin_key"]: item["spec_json"] for item in load_builtin_workflows()}
    assert "corrective_self_rag_agent" in workflows
    validator = WorkflowValidator()
    assert validator.collect_errors(workflows["corrective_self_rag_agent"]) == []
    for key in ("router_rag_agent", "plan_execute_rag_agent", "evaluator_replanner_rag_agent", "orchestrator_worker_rag_agent"):
        assert validator.collect_errors(workflows[key]) == []


def test_migration_contains_same_immutable_v1_snapshot_as_builtin():
    root = Path(__file__).parents[1]
    builtin = json.loads((root / "app/agent_workflows/builtins/corrective_self_rag_agent.json").read_text())["spec_json"]
    migration_text = (root / "alembic/versions/e7c4a1b9d2f6_seed_corrective_self_rag.py").read_text()
    assert 'revision = "e7c4a1b9d2f6"' in migration_text
    tree = ast.parse(migration_text)
    snapshot = next(
        ast.literal_eval(node.value.args[0])
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SPEC_JSON" for target in node.targets)
        and isinstance(node.value, ast.Call)
    )
    assert json.loads(snapshot) == builtin
