from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


ROUTER_RAG_AGENT_ID = "router_rag_agent"
ROUTER_RAG_AGENT_VERSION = 1
ROUTER_RAG_AGENT_VERSION_ID = f"{ROUTER_RAG_AGENT_ID}:v{ROUTER_RAG_AGENT_VERSION}"
SUPPORTED_BUILTIN_TEMPLATE_IDS = {ROUTER_RAG_AGENT_ID}


ALLOWED_ROUTER_RAG_CONFIG_KEYS = {
    "use_intent_agent",
    "use_web_search",
    "use_reranker",
    "max_iterations",
    "system_role",
    "tool_instructions",
    "custom_instructions",
    "allowed_tool_ids",
    "prefetch_policy",
    "graph",
}


BUILTIN_ROUTER_RAG_SPEC: Dict[str, Any] = {
    "schema_version": 1,
    "pattern_type": ROUTER_RAG_AGENT_ID,
    "config": {
        "use_intent_agent": False,
        "use_web_search": False,
        "use_reranker": True,
        "max_iterations": 1,
        "system_role": "",
        "tool_instructions": {},
        "custom_instructions": "",
        "allowed_tool_ids": [
            "document_evidence",
            "deep_memory",
            "clarify_intent",
        ],
        "prefetch_policy": {
            "enabled": True,
        },
        "graph": {
            "nodes": [
                {"id": "context_loader", "type": "context_loader"},
                {"id": "router", "type": "router"},
                {"id": "retrieval_worker", "type": "retrieval_worker"},
                {"id": "memory_worker", "type": "memory_worker"},
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
                        "direct": "direct_answer",
                        "clarify": "finalizer",
                    },
                },
                {"from": "retrieval_worker", "to": "synthesizer"},
                {"from": "memory_worker", "to": "synthesizer"},
                {"from": "direct_answer", "to": "finalizer"},
                {"from": "synthesizer", "to": "finalizer"},
                {"from": "finalizer", "to": "END"},
            ],
        },
    },
}


def builtin_router_rag_spec() -> Dict[str, Any]:
    return deepcopy(BUILTIN_ROUTER_RAG_SPEC)


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
                "schema_version": 1,
                "spec_json": builtin_router_rag_spec(),
                "changelog": "Initial compiled Router RAG Agent pattern.",
            },
        },
    ]
