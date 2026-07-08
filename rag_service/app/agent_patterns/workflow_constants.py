from __future__ import annotations


ROUTER_RAG_AGENT_ID = "router_rag_agent"
PLAN_EXECUTE_RAG_AGENT_ID = "plan_execute_rag_agent"
EVALUATOR_REPLANNER_RAG_AGENT_ID = "evaluator_replanner_rag_agent"
WEB_APPROVAL_GATE_ID = "web_approval_gate"
EVALUATOR_REPLANNER_REPEATABLE_NODE_TYPES = {
    "retrieval_worker",
    "memory_worker",
    "timeline_worker",
    "web_worker",
    "evidence_evaluator",
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
