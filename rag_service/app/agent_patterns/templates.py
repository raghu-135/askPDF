from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


ROUTER_RAG_AGENT_ID = "router_rag_agent"
ROUTER_RAG_AGENT_VERSION = 2
ROUTER_RAG_AGENT_VERSION_ID = f"{ROUTER_RAG_AGENT_ID}:v{ROUTER_RAG_AGENT_VERSION}"
ROUTER_RAG_AGENT_V2_VERSION = 2
ROUTER_RAG_AGENT_V2_VERSION_ID = f"{ROUTER_RAG_AGENT_ID}:v{ROUTER_RAG_AGENT_V2_VERSION}"
PLAN_EXECUTE_RAG_AGENT_ID = "plan_execute_rag_agent"
PLAN_EXECUTE_RAG_AGENT_VERSION = 2
PLAN_EXECUTE_RAG_AGENT_VERSION_ID = f"{PLAN_EXECUTE_RAG_AGENT_ID}:v{PLAN_EXECUTE_RAG_AGENT_VERSION}"
PLAN_EXECUTE_RAG_AGENT_V2_VERSION = 2
PLAN_EXECUTE_RAG_AGENT_V2_VERSION_ID = f"{PLAN_EXECUTE_RAG_AGENT_ID}:v{PLAN_EXECUTE_RAG_AGENT_V2_VERSION}"
EVALUATOR_REPLANNER_RAG_AGENT_ID = "evaluator_replanner_rag_agent"
EVALUATOR_REPLANNER_RAG_AGENT_VERSION = 2
EVALUATOR_REPLANNER_RAG_AGENT_VERSION_ID = (
    f"{EVALUATOR_REPLANNER_RAG_AGENT_ID}:v{EVALUATOR_REPLANNER_RAG_AGENT_VERSION}"
)
EVALUATOR_REPLANNER_RAG_AGENT_V2_VERSION = 2
EVALUATOR_REPLANNER_RAG_AGENT_V2_VERSION_ID = (
    f"{EVALUATOR_REPLANNER_RAG_AGENT_ID}:v{EVALUATOR_REPLANNER_RAG_AGENT_V2_VERSION}"
)
SUPPORTED_BUILTIN_TEMPLATE_IDS = {
    ROUTER_RAG_AGENT_ID,
    PLAN_EXECUTE_RAG_AGENT_ID,
    EVALUATOR_REPLANNER_RAG_AGENT_ID,
}
WEB_APPROVAL_GATE_ID = "web_approval_gate"

ROUTER_RAG_REQUIRED_TOOL_IDS = {
    "document_evidence",
    "deep_memory",
    "thread_timeline",
    "live_web_recon",
    "clarify_intent",
}

ROUTER_RAG_NODE_TOOL_REQUIREMENTS = {
    "retrieval_worker": "document_evidence",
    "memory_worker": "deep_memory",
    "timeline_worker": "thread_timeline",
    "web_worker": "live_web_recon",
    "router": "clarify_intent",
    "finalizer": "clarify_intent",
}

PLAN_EXECUTE_RAG_REQUIRED_TOOL_IDS = set(ROUTER_RAG_REQUIRED_TOOL_IDS)

PLAN_EXECUTE_RAG_NODE_TOOL_REQUIREMENTS = {
    "retrieval_worker": "document_evidence",
    "memory_worker": "deep_memory",
    "timeline_worker": "thread_timeline",
    "web_worker": "live_web_recon",
    "planner": "clarify_intent",
    "finalizer": "clarify_intent",
}

EVALUATOR_REPLANNER_RAG_REQUIRED_TOOL_IDS = set(PLAN_EXECUTE_RAG_REQUIRED_TOOL_IDS)

EVALUATOR_REPLANNER_RAG_NODE_TOOL_REQUIREMENTS = {
    **PLAN_EXECUTE_RAG_NODE_TOOL_REQUIREMENTS,
    "evidence_evaluator": "clarify_intent",
    "replanner": "clarify_intent",
}

PLAN_EXECUTE_WORKER_NODES = [
    "retrieval_worker",
    "memory_worker",
    "timeline_worker",
    "web_worker",
]


ALLOWED_ROUTER_RAG_CONFIG_KEYS = {
    "use_web_search",
    "use_reranker",
    "system_role",
    "tool_instructions",
    "custom_instructions",
    "allowed_tool_ids",
    "prefetch_policy",
    "hitl_policy",
    "replans",
    "graph",
}


BUILTIN_ROUTER_RAG_SPEC: Dict[str, Any] = {
    "schema_version": 1,
    "pattern_type": ROUTER_RAG_AGENT_ID,
    "config": {
        "use_web_search": False,
        "use_reranker": True,
        "system_role": "",
        "tool_instructions": {},
        "custom_instructions": "",
        "allowed_tool_ids": [
            "document_evidence",
            "deep_memory",
            "thread_timeline",
            "live_web_recon",
            "clarify_intent",
        ],
        "prefetch_policy": {
            "enabled": True,
        },
        "hitl_policy": {
            "enabled": False,
            "gates": {},
        },
        "graph": {
            "nodes": [
                {"id": "context_loader", "type": "context_loader"},
                {"id": "router", "type": "router"},
                {"id": "retrieval_worker", "type": "retrieval_worker"},
                {"id": "memory_worker", "type": "memory_worker"},
                {"id": "timeline_worker", "type": "timeline_worker"},
                {"id": "web_worker", "type": "web_worker"},
                {"id": "direct_answer", "type": "direct_answer"},
                {"id": "synthesizer", "type": "synthesizer"},
                {"id": "finalizer", "type": "finalizer"},
            ],
            "edges": [
                {"from": "START", "to": "context_loader"},
                {"from": "context_loader", "to": "router"},
                {
                    "from": "router",
                    "conditional": True,
                    "routes": {
                        "document": "retrieval_worker",
                        "memory": "memory_worker",
                        "timeline": "timeline_worker",
                        "web": "web_worker",
                        "direct": "direct_answer",
                        "clarify": "finalizer",
                    },
                },
                {"from": "retrieval_worker", "to": "synthesizer"},
                {"from": "memory_worker", "to": "synthesizer"},
                {"from": "timeline_worker", "to": "synthesizer"},
                {"from": "web_worker", "to": "synthesizer"},
                {"from": "direct_answer", "to": "finalizer"},
                {"from": "synthesizer", "to": "finalizer"},
                {"from": "finalizer", "to": "END"},
            ],
        },
    },
}


BUILTIN_ROUTER_RAG_HITL_WEB_SPEC: Dict[str, Any] = deepcopy(BUILTIN_ROUTER_RAG_SPEC)
BUILTIN_ROUTER_RAG_HITL_WEB_SPEC["config"]["hitl_policy"] = {
    "enabled": True,
    "gates": {
        WEB_APPROVAL_GATE_ID: {
            "enabled": True,
            "mode": "approval",
            "phase": "before",
            "target": {"node_id": "web_worker", "node_type": "web_worker"},
            "interrupt_type": "tool_approval",
            "title": "Approve web search?",
            "prompt": "This answer needs live web research. Approve web search or continue without it.",
            "allowed_actions": ["approve", "continue_without"],
            "default_action": "continue_without",
            "routes": {
                "approve": "web_worker",
                "continue_without": "synthesizer",
            },
            "payload_projection": ["question", "route", "route_reason", "evidence_summary"],
        },
    },
}


BUILTIN_PLAN_EXECUTE_RAG_SPEC: Dict[str, Any] = {
    "schema_version": 1,
    "pattern_type": PLAN_EXECUTE_RAG_AGENT_ID,
    "config": {
        "use_web_search": False,
        "use_reranker": True,
        "system_role": "",
        "tool_instructions": {},
        "custom_instructions": "",
        "allowed_tool_ids": [
            "document_evidence",
            "deep_memory",
            "thread_timeline",
            "live_web_recon",
            "clarify_intent",
        ],
        "prefetch_policy": {
            "enabled": True,
        },
        "hitl_policy": {
            "enabled": False,
            "gates": {},
        },
        "graph": {
            "nodes": [
                {"id": "context_loader", "type": "context_loader"},
                {"id": "planner", "type": "planner"},
                {"id": "direct_answer", "type": "direct_answer"},
                {"id": "retrieval_worker", "type": "retrieval_worker"},
                {"id": "memory_worker", "type": "memory_worker"},
                {"id": "timeline_worker", "type": "timeline_worker"},
                {"id": "web_worker", "type": "web_worker"},
                {"id": "synthesizer", "type": "synthesizer"},
                {"id": "finalizer", "type": "finalizer"},
            ],
            "edges": [
                {"from": "START", "to": "context_loader"},
                {"from": "context_loader", "to": "planner"},
                {
                    "from": "planner",
                    "conditional": True,
                    "routes": {
                        "execute": "retrieval_worker",
                        "direct": "direct_answer",
                        "clarify": "finalizer",
                    },
                },
                {"from": "direct_answer", "to": "finalizer"},
                {"from": "retrieval_worker", "to": "memory_worker"},
                {"from": "memory_worker", "to": "timeline_worker"},
                {"from": "timeline_worker", "to": "web_worker"},
                {"from": "web_worker", "to": "synthesizer"},
                {"from": "synthesizer", "to": "finalizer"},
                {"from": "finalizer", "to": "END"},
            ],
        },
    },
}


BUILTIN_EVALUATOR_REPLANNER_RAG_SPEC: Dict[str, Any] = {
    "schema_version": 1,
    "pattern_type": EVALUATOR_REPLANNER_RAG_AGENT_ID,
    "config": {
        "use_web_search": False,
        "use_reranker": True,
        "replans": 1,
        "system_role": "",
        "tool_instructions": {},
        "custom_instructions": "",
        "allowed_tool_ids": [
            "document_evidence",
            "deep_memory",
            "thread_timeline",
            "live_web_recon",
            "clarify_intent",
        ],
        "prefetch_policy": {
            "enabled": True,
        },
        "hitl_policy": {
            "enabled": False,
            "gates": {},
        },
        "graph": {
            "nodes": [
                {"id": "context_loader", "type": "context_loader"},
                {"id": "planner", "type": "planner"},
                {"id": "direct_answer", "type": "direct_answer"},
                {"id": "retrieval_worker", "type": "retrieval_worker"},
                {"id": "memory_worker", "type": "memory_worker"},
                {"id": "timeline_worker", "type": "timeline_worker"},
                {"id": "web_worker", "type": "web_worker"},
                {"id": "evidence_evaluator", "type": "evidence_evaluator"},
                {"id": "replanner", "type": "replanner"},
                {"id": "synthesizer", "type": "synthesizer"},
                {"id": "finalizer", "type": "finalizer"},
            ],
            "edges": [
                {"from": "START", "to": "context_loader"},
                {"from": "context_loader", "to": "planner"},
                {
                    "from": "planner",
                    "conditional": True,
                    "routes": {
                        "execute": "retrieval_worker",
                        "direct": "direct_answer",
                        "clarify": "finalizer",
                    },
                },
                {"from": "direct_answer", "to": "finalizer"},
                {"from": "retrieval_worker", "to": "memory_worker"},
                {"from": "memory_worker", "to": "timeline_worker"},
                {"from": "timeline_worker", "to": "web_worker"},
                {"from": "web_worker", "to": "evidence_evaluator"},
                {
                    "from": "evidence_evaluator",
                    "conditional": True,
                    "routes": {
                        "answer": "synthesizer",
                        "replan": "replanner",
                        "answer_budget_exhausted": "synthesizer",
                    },
                },
                {"from": "replanner", "to": "retrieval_worker"},
                {"from": "synthesizer", "to": "finalizer"},
                {"from": "finalizer", "to": "END"},
            ],
        },
    },
}


def legacy_builtin_router_rag_v1_spec() -> Dict[str, Any]:
    return deepcopy(BUILTIN_ROUTER_RAG_SPEC)


def legacy_builtin_router_rag_hitl_web_v1_spec() -> Dict[str, Any]:
    return deepcopy(BUILTIN_ROUTER_RAG_HITL_WEB_SPEC)


def legacy_builtin_plan_execute_rag_v1_spec() -> Dict[str, Any]:
    return deepcopy(BUILTIN_PLAN_EXECUTE_RAG_SPEC)


def legacy_builtin_evaluator_replanner_rag_v1_spec() -> Dict[str, Any]:
    return deepcopy(BUILTIN_EVALUATOR_REPLANNER_RAG_SPEC)


def _with_v2_contract(spec: Dict[str, Any], *, max_total_visits: int) -> Dict[str, Any]:
    v2 = deepcopy(spec)
    v2["schema_version"] = 2
    config = v2.setdefault("config", {})
    config.setdefault(
        "context_policy",
        {
            "evidence_packet_limit": 12,
            "evidence_packet_content_limit": 2000,
            "final_prompt_assembly": "legacy_evidence",
        },
    )
    graph = config.get("graph") if isinstance(config.get("graph"), dict) else {}
    node_types = {
        str(node.get("id")): str(node.get("type"))
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("id") and node.get("type")
    }
    repeatable_node_ids = {
        node_id
        for node_id, node_type in node_types.items()
        if node_type in {"retrieval_worker", "memory_worker", "timeline_worker", "web_worker", "evidence_evaluator"}
    }
    config["loop_policy"] = {
        "max_total_visits": max_total_visits,
        "default_max_node_visits": 1,
        "node_visit_limits": {node_id: 2 for node_id in sorted(repeatable_node_ids)},
    }
    route_by_type = {
        "router": "router_route",
        "planner": "planner_route",
        "evidence_evaluator": "evaluator_route",
        "hitl_gate": "hitl_gate_route",
    }
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict) or not edge.get("conditional"):
            continue
        source = str(edge.get("from"))
        route_fn = route_by_type.get(node_types.get(source, source))
        if route_fn:
            edge["route_fn"] = route_fn
    return v2


def builtin_router_rag_v2_spec() -> Dict[str, Any]:
    return _with_v2_contract(BUILTIN_ROUTER_RAG_SPEC, max_total_visits=9)


def builtin_plan_execute_rag_v2_spec() -> Dict[str, Any]:
    return _with_v2_contract(BUILTIN_PLAN_EXECUTE_RAG_SPEC, max_total_visits=9)


def builtin_evaluator_replanner_rag_v2_spec() -> Dict[str, Any]:
    return _with_v2_contract(BUILTIN_EVALUATOR_REPLANNER_RAG_SPEC, max_total_visits=16)


def builtin_router_rag_spec() -> Dict[str, Any]:
    return builtin_router_rag_v2_spec()


def builtin_router_rag_hitl_web_spec() -> Dict[str, Any]:
    spec = builtin_router_rag_v2_spec()
    spec["config"]["hitl_policy"] = deepcopy(BUILTIN_ROUTER_RAG_HITL_WEB_SPEC["config"]["hitl_policy"])
    return spec


def builtin_plan_execute_rag_spec() -> Dict[str, Any]:
    return builtin_plan_execute_rag_v2_spec()


def builtin_evaluator_replanner_rag_spec() -> Dict[str, Any]:
    return builtin_evaluator_replanner_rag_v2_spec()


def builtin_templates() -> list[Dict[str, Any]]:
    return [
        {
            "id": ROUTER_RAG_AGENT_ID,
            "name": "Router RAG Agent",
            "description": "A compiled LangGraph pattern that loads context, routes to document retrieval, memory retrieval, direct answer, or clarification, then synthesizes a final response.",
            "visibility": "builtin",
            "is_builtin": True,
            "current_version_id": ROUTER_RAG_AGENT_VERSION_ID,
            "version": {
                "id": ROUTER_RAG_AGENT_VERSION_ID,
                "version": ROUTER_RAG_AGENT_VERSION,
                "schema_version": 2,
                "spec_json": builtin_router_rag_v2_spec(),
                "changelog": "Catalog-backed v2 Router RAG Agent pattern.",
            },
            "versions": [
                {
                    "id": ROUTER_RAG_AGENT_VERSION_ID,
                    "version": ROUTER_RAG_AGENT_VERSION,
                    "schema_version": 2,
                    "spec_json": builtin_router_rag_v2_spec(),
                    "changelog": "Catalog-backed v2 Router RAG Agent pattern.",
                },
            ],
        },
        {
            "id": PLAN_EXECUTE_RAG_AGENT_ID,
            "name": "Plan-and-Execute RAG Agent",
            "description": "A scoped compiled RAG pattern that loads context, plans a bounded set of retrieval workers, executes them in a fixed safe order, then synthesizes a final response.",
            "visibility": "builtin",
            "is_builtin": True,
            "current_version_id": PLAN_EXECUTE_RAG_AGENT_VERSION_ID,
            "version": {
                "id": PLAN_EXECUTE_RAG_AGENT_VERSION_ID,
                "version": PLAN_EXECUTE_RAG_AGENT_VERSION,
                "schema_version": 2,
                "spec_json": builtin_plan_execute_rag_v2_spec(),
                "changelog": "Catalog-backed v2 Plan-and-Execute RAG Agent pattern.",
            },
            "versions": [
                {
                    "id": PLAN_EXECUTE_RAG_AGENT_VERSION_ID,
                    "version": PLAN_EXECUTE_RAG_AGENT_VERSION,
                    "schema_version": 2,
                    "spec_json": builtin_plan_execute_rag_v2_spec(),
                    "changelog": "Catalog-backed v2 Plan-and-Execute RAG Agent pattern.",
                },
            ],
        },
        {
            "id": EVALUATOR_REPLANNER_RAG_AGENT_ID,
            "name": "Evaluator/Replanner RAG Agent",
            "description": "A bounded compiled RAG pattern that plans retrieval, evaluates evidence sufficiency, optionally replans once, then synthesizes a final response.",
            "visibility": "builtin",
            "is_builtin": True,
            "current_version_id": EVALUATOR_REPLANNER_RAG_AGENT_VERSION_ID,
            "version": {
                "id": EVALUATOR_REPLANNER_RAG_AGENT_VERSION_ID,
                "version": EVALUATOR_REPLANNER_RAG_AGENT_VERSION,
                "schema_version": 2,
                "spec_json": builtin_evaluator_replanner_rag_v2_spec(),
                "changelog": "Catalog-backed v2 Evaluator/Replanner RAG Agent pattern.",
            },
            "versions": [
                {
                    "id": EVALUATOR_REPLANNER_RAG_AGENT_VERSION_ID,
                    "version": EVALUATOR_REPLANNER_RAG_AGENT_VERSION,
                    "schema_version": 2,
                    "spec_json": builtin_evaluator_replanner_rag_v2_spec(),
                    "changelog": "Catalog-backed v2 Evaluator/Replanner RAG Agent pattern.",
                },
            ],
        },
    ]
