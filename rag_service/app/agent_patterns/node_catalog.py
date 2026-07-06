from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


NODE_CATALOG: Dict[str, Dict[str, Any]] = {
    "context_loader": {
        "display_name": "Context Loader",
        "category": "context",
        "capabilities": ["context.prefetch"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": ["thread_shape"],
        "allowed_parent_types": ["START"],
        "allowed_child_types": ["router", "planner"],
    },
    "router": {
        "display_name": "Router",
        "category": "control",
        "capabilities": ["route.intent", "clarify"],
        "allowed_route_functions": ["router_route"],
        "allowed_tool_contract_ids": ["clarify_intent"],
        "allowed_parent_types": ["context_loader", "hitl_gate"],
        "allowed_child_types": [
            "retrieval_worker",
            "memory_worker",
            "timeline_worker",
            "web_worker",
            "direct_answer",
            "finalizer",
            "hitl_gate",
        ],
    },
    "planner": {
        "display_name": "Planner",
        "category": "control",
        "capabilities": ["plan.execution", "clarify"],
        "allowed_route_functions": ["planner_route"],
        "allowed_tool_contract_ids": ["clarify_intent"],
        "allowed_parent_types": ["context_loader", "hitl_gate"],
        "allowed_child_types": ["retrieval_worker", "direct_answer", "finalizer", "hitl_gate"],
    },
    "retrieval_worker": {
        "display_name": "Document Retrieval",
        "category": "retrieval",
        "capabilities": ["retrieval.document"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": ["document_evidence", "focused_document_evidence"],
        "allowed_parent_types": ["router", "planner", "replanner", "hitl_gate"],
        "allowed_child_types": [
            "memory_worker",
            "timeline_worker",
            "web_worker",
            "evidence_evaluator",
            "synthesizer",
            "finalizer",
            "hitl_gate",
        ],
    },
    "memory_worker": {
        "display_name": "Memory Retrieval",
        "category": "retrieval",
        "capabilities": ["retrieval.memory"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": ["deep_memory"],
        "allowed_parent_types": ["router", "retrieval_worker", "planner", "replanner", "hitl_gate"],
        "allowed_child_types": [
            "timeline_worker",
            "web_worker",
            "evidence_evaluator",
            "synthesizer",
            "finalizer",
            "hitl_gate",
        ],
    },
    "timeline_worker": {
        "display_name": "Timeline Retrieval",
        "category": "retrieval",
        "capabilities": ["retrieval.timeline"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": ["thread_timeline"],
        "allowed_parent_types": ["router", "memory_worker", "planner", "replanner", "hitl_gate"],
        "allowed_child_types": ["web_worker", "evidence_evaluator", "synthesizer", "finalizer", "hitl_gate"],
    },
    "web_worker": {
        "display_name": "Web Retrieval",
        "category": "retrieval",
        "capabilities": ["retrieval.web", "external_research"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [
            "live_web_recon",
            "wikipedia_reference",
            "wikidata_reference",
            "arxiv_research",
            "pubmed_research",
            "semantic_scholar_research",
            "stackexchange_reference",
            "yahoo_finance_news",
        ],
        "allowed_parent_types": ["router", "timeline_worker", "memory_worker", "planner", "replanner", "hitl_gate"],
        "allowed_child_types": ["evidence_evaluator", "synthesizer", "finalizer", "hitl_gate"],
    },
    "evidence_evaluator": {
        "display_name": "Evidence Evaluator",
        "category": "control",
        "capabilities": ["evaluate.evidence", "clarify"],
        "allowed_route_functions": ["evaluator_route"],
        "allowed_tool_contract_ids": ["clarify_intent"],
        "allowed_parent_types": ["retrieval_worker", "memory_worker", "timeline_worker", "web_worker", "hitl_gate"],
        "allowed_child_types": ["synthesizer", "replanner", "hitl_gate"],
    },
    "replanner": {
        "display_name": "Replanner",
        "category": "control",
        "capabilities": ["plan.replan", "clarify"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": ["clarify_intent"],
        "allowed_parent_types": ["evidence_evaluator", "hitl_gate"],
        "allowed_child_types": ["retrieval_worker", "memory_worker", "timeline_worker", "web_worker", "hitl_gate"],
    },
    "direct_answer": {
        "display_name": "Direct Answer",
        "category": "answer",
        "capabilities": ["answer.direct"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [],
        "allowed_parent_types": ["router", "planner", "hitl_gate"],
        "allowed_child_types": ["finalizer", "hitl_gate"],
    },
    "synthesizer": {
        "display_name": "Synthesizer",
        "category": "answer",
        "capabilities": ["answer.synthesize"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [],
        "allowed_parent_types": [
            "retrieval_worker",
            "memory_worker",
            "timeline_worker",
            "web_worker",
            "evidence_evaluator",
            "hitl_gate",
        ],
        "allowed_child_types": ["finalizer", "hitl_gate"],
    },
    "finalizer": {
        "display_name": "Finalizer",
        "category": "answer",
        "capabilities": ["answer.final", "clarify"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": ["clarify_intent"],
        "allowed_parent_types": [
            "router",
            "planner",
            "retrieval_worker",
            "memory_worker",
            "timeline_worker",
            "web_worker",
            "direct_answer",
            "synthesizer",
            "hitl_gate",
        ],
        "allowed_child_types": ["END"],
    },
    "hitl_gate": {
        "display_name": "HITL Gate",
        "category": "human_review",
        "capabilities": ["hitl.interrupt"],
        "allowed_route_functions": ["hitl_gate_route"],
        "allowed_tool_contract_ids": [],
        "allowed_parent_types": ["router", "planner", "retrieval_worker", "memory_worker", "timeline_worker", "web_worker", "evidence_evaluator", "replanner", "direct_answer", "synthesizer", "finalizer"],
        "allowed_child_types": ["router", "planner", "retrieval_worker", "memory_worker", "timeline_worker", "web_worker", "evidence_evaluator", "replanner", "direct_answer", "synthesizer", "finalizer", "END"],
    },
}


def get_node_catalog() -> Dict[str, Dict[str, Any]]:
    return deepcopy(NODE_CATALOG)


def get_node_type_metadata(node_type: str) -> Dict[str, Any]:
    return deepcopy(NODE_CATALOG.get(node_type) or {})


def known_node_types() -> set[str]:
    return set(NODE_CATALOG)


def node_type_capabilities(node_type: str) -> list[str]:
    metadata = NODE_CATALOG.get(node_type) or {}
    return list(metadata.get("capabilities") or [])


def node_type_allowed_tool_contract_ids(node_type: str) -> set[str]:
    metadata = NODE_CATALOG.get(node_type) or {}
    return {str(item) for item in metadata.get("allowed_tool_contract_ids") or [] if item}
