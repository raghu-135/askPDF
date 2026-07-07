from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from app.models.llm_server_client import REPLANS_LIMIT


NODE_CATALOG: Dict[str, Dict[str, Any]] = {
    "context_loader": {
        "display_name": "Context Loader",
        "category": "context",
        "capabilities": ["context.prefetch"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": ["thread_shape"],
        "allowed_parent_types": ["START"],
        "allowed_child_types": ["router", "planner"],
        "limits": {"default_max_visits": 1},
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
        "limits": {"default_max_visits": 1},
    },
    "planner": {
        "display_name": "Planner",
        "category": "control",
        "capabilities": ["plan.execution", "clarify"],
        "allowed_route_functions": ["planner_route"],
        "allowed_tool_contract_ids": ["clarify_intent"],
        "allowed_parent_types": ["context_loader", "hitl_gate"],
        "allowed_child_types": ["retrieval_worker", "direct_answer", "finalizer", "hitl_gate"],
        "limits": {"default_max_visits": 1},
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
        "limits": {"default_max_visits": 2, "max_visits": REPLANS_LIMIT + 1},
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
        "limits": {"default_max_visits": 2, "max_visits": REPLANS_LIMIT + 1},
    },
    "timeline_worker": {
        "display_name": "Timeline Retrieval",
        "category": "retrieval",
        "capabilities": ["retrieval.timeline"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": ["thread_timeline"],
        "allowed_parent_types": ["router", "memory_worker", "planner", "replanner", "hitl_gate"],
        "allowed_child_types": ["web_worker", "evidence_evaluator", "synthesizer", "finalizer", "hitl_gate"],
        "limits": {"default_max_visits": 2, "max_visits": REPLANS_LIMIT + 1},
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
        "limits": {"default_max_visits": 2, "max_visits": REPLANS_LIMIT + 1},
    },
    "evidence_evaluator": {
        "display_name": "Evidence Evaluator",
        "category": "control",
        "capabilities": ["evaluate.evidence", "clarify"],
        "allowed_route_functions": ["evaluator_route"],
        "allowed_tool_contract_ids": ["clarify_intent"],
        "allowed_parent_types": ["retrieval_worker", "memory_worker", "timeline_worker", "web_worker", "hitl_gate"],
        "allowed_child_types": ["synthesizer", "replanner", "hitl_gate"],
        "limits": {"default_max_visits": 2, "max_visits": REPLANS_LIMIT + 1},
    },
    "replanner": {
        "display_name": "Replanner",
        "category": "control",
        "capabilities": ["plan.replan", "clarify"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": ["clarify_intent"],
        "allowed_parent_types": ["evidence_evaluator", "hitl_gate"],
        "allowed_child_types": ["retrieval_worker", "memory_worker", "timeline_worker", "web_worker", "hitl_gate"],
        "limits": {"default_max_visits": 1, "max_visits": REPLANS_LIMIT},
    },
    "direct_answer": {
        "display_name": "Direct Answer",
        "category": "answer",
        "capabilities": ["answer.direct"],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [],
        "allowed_parent_types": ["router", "planner", "hitl_gate"],
        "allowed_child_types": ["finalizer", "hitl_gate"],
        "limits": {"default_max_visits": 1},
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
        "limits": {"default_max_visits": 1},
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
        "allowed_child_types": ["hitl_gate", "END"],
        "limits": {"default_max_visits": 1},
    },
    "hitl_gate": {
        "display_name": "HITL Gate",
        "category": "human_review",
        "capabilities": ["hitl.interrupt"],
        "allowed_route_functions": ["hitl_gate_route"],
        "allowed_tool_contract_ids": [],
        "allowed_parent_types": ["router", "planner", "retrieval_worker", "memory_worker", "timeline_worker", "web_worker", "evidence_evaluator", "replanner", "direct_answer", "synthesizer", "finalizer"],
        "allowed_child_types": ["router", "planner", "retrieval_worker", "memory_worker", "timeline_worker", "web_worker", "evidence_evaluator", "replanner", "direct_answer", "synthesizer", "finalizer", "END"],
        "limits": {"default_max_visits": 1},
    },
}


_NODE_CATALOG_METADATA: Dict[str, Dict[str, Any]] = {
    "context_loader": {
        "state_reads": ["thread_id", "question", "embedding_model", "context_window", "use_web_search", "use_reranker"],
        "state_writes": ["pre_fetch_bundle", "document_sources", "web_sources", "used_chat_ids"],
        "prompt_slots": [],
        "context_policy": {"mode": "prefetch", "input_budget": "request", "output_budget": "bounded"},
        "observability": {
            "span_kind": "context",
            "event_prefix": "context_loader",
            "summary_fields": ["document_source_count", "web_source_count", "used_chat_id_count"],
            "raw_payload": "bounded",
        },
        "max_instances": 1,
    },
    "router": {
        "state_reads": ["question", "pre_fetch_bundle", "use_web_search", "client_timezone", "client_locale", "client_now_iso"],
        "state_writes": ["route", "route_reason", "clarification_options"],
        "prompt_slots": ["router"],
        "context_policy": {"mode": "route", "input_budget": "bounded_prefetch", "output_budget": "decision"},
        "observability": {
            "span_kind": "control",
            "event_prefix": "router",
            "summary_fields": ["route", "route_reason"],
            "raw_payload": "bounded",
        },
        "max_instances": 1,
    },
    "planner": {
        "state_reads": ["question", "pre_fetch_bundle", "use_web_search", "client_timezone", "client_locale", "client_now_iso"],
        "state_writes": ["route", "route_reason", "execution_plan", "clarification_options"],
        "prompt_slots": ["planner"],
        "context_policy": {"mode": "plan", "input_budget": "bounded_prefetch", "output_budget": "decision"},
        "observability": {
            "span_kind": "control",
            "event_prefix": "planner",
            "summary_fields": ["route", "route_reason", "execution_plan"],
            "raw_payload": "bounded",
        },
        "max_instances": 1,
    },
    "retrieval_worker": {
        "state_reads": ["question", "thread_id", "embedding_model", "use_reranker", "execution_plan", "evidence"],
        "state_writes": ["evidence", "evidence_packets", "document_sources", "web_sources", "tool_events"],
        "prompt_slots": [],
        "context_policy": {"mode": "append_evidence", "input_budget": "tool_query", "output_budget": "evidence_packet"},
        "observability": {
            "span_kind": "tool_worker",
            "event_prefix": "retrieval_worker",
            "summary_fields": ["document_source_count", "web_source_count", "evidence_chars"],
            "raw_payload": "bounded",
        },
        "max_instances": 4,
    },
    "memory_worker": {
        "state_reads": ["question", "thread_id", "embedding_model", "execution_plan", "evidence"],
        "state_writes": ["evidence", "evidence_packets", "used_chat_ids", "tool_events"],
        "prompt_slots": [],
        "context_policy": {"mode": "append_evidence", "input_budget": "tool_query", "output_budget": "evidence_packet"},
        "observability": {
            "span_kind": "tool_worker",
            "event_prefix": "memory_worker",
            "summary_fields": ["used_chat_id_count", "evidence_chars"],
            "raw_payload": "bounded",
        },
        "max_instances": 4,
    },
    "timeline_worker": {
        "state_reads": ["question", "thread_id", "embedding_model", "execution_plan", "evidence"],
        "state_writes": ["evidence", "evidence_packets", "tool_events"],
        "prompt_slots": [],
        "context_policy": {"mode": "append_evidence", "input_budget": "tool_query", "output_budget": "evidence_packet"},
        "observability": {
            "span_kind": "tool_worker",
            "event_prefix": "timeline_worker",
            "summary_fields": ["timeline_event_count", "evidence_chars"],
            "raw_payload": "bounded",
        },
        "max_instances": 4,
    },
    "web_worker": {
        "state_reads": ["question", "use_web_search", "execution_plan", "evidence"],
        "state_writes": ["evidence", "evidence_packets", "web_sources", "tool_events"],
        "prompt_slots": ["web_search_mandate"],
        "context_policy": {"mode": "append_evidence", "input_budget": "tool_query", "output_budget": "evidence_packet"},
        "observability": {
            "span_kind": "tool_worker",
            "event_prefix": "web_worker",
            "summary_fields": ["web_source_count", "evidence_chars"],
            "raw_payload": "bounded",
        },
        "max_instances": 4,
    },
    "evidence_evaluator": {
        "state_reads": ["question", "evidence", "evidence_packets", "replan_count", "replans"],
        "state_writes": ["evaluator_report", "evidence_gaps", "evaluation_confidence", "evaluator_route"],
        "prompt_slots": ["evaluator"],
        "context_policy": {"mode": "evaluate_evidence", "input_budget": "bounded_evidence", "output_budget": "decision"},
        "observability": {
            "span_kind": "control",
            "event_prefix": "evidence_evaluator",
            "summary_fields": ["evaluator_route", "evaluation_confidence", "evidence_gaps"],
            "raw_payload": "bounded",
        },
        "max_instances": 2,
    },
    "replanner": {
        "state_reads": ["question", "evidence", "evaluator_report", "replan_count", "replans"],
        "state_writes": ["execution_plan", "replan_count", "replan_reason", "replan_history"],
        "prompt_slots": ["replanner"],
        "context_policy": {"mode": "plan", "input_budget": "bounded_evidence", "output_budget": "decision"},
        "observability": {
            "span_kind": "control",
            "event_prefix": "replanner",
            "summary_fields": ["execution_plan", "replan_count", "replan_reason"],
            "raw_payload": "bounded",
        },
        "max_instances": 1,
    },
    "direct_answer": {
        "state_reads": ["question", "pre_fetch_bundle", "route_reason"],
        "state_writes": ["final_answer", "reasoning", "reasoning_available", "reasoning_format"],
        "prompt_slots": ["final_answer"],
        "context_policy": {"mode": "assemble_answer", "input_budget": "bounded_prefetch", "output_budget": "answer"},
        "observability": {
            "span_kind": "answer",
            "event_prefix": "direct_answer",
            "summary_fields": ["answer_chars"],
            "raw_payload": "bounded",
        },
        "max_instances": 1,
    },
    "synthesizer": {
        "state_reads": ["question", "evidence", "evidence_packets", "document_sources", "web_sources", "used_chat_ids"],
        "state_writes": ["final_answer", "reasoning", "reasoning_available", "reasoning_format"],
        "prompt_slots": ["final_answer"],
        "context_policy": {"mode": "assemble_answer", "input_budget": "bounded_evidence", "output_budget": "answer"},
        "observability": {
            "span_kind": "answer",
            "event_prefix": "synthesizer",
            "summary_fields": ["answer_chars", "document_source_count", "web_source_count"],
            "raw_payload": "bounded",
        },
        "max_instances": 1,
    },
    "finalizer": {
        "state_reads": ["final_answer", "clarification_options", "document_sources", "web_sources", "used_chat_ids"],
        "state_writes": ["final_answer", "reasoning", "reasoning_available", "reasoning_format"],
        "prompt_slots": ["final_answer"],
        "context_policy": {"mode": "finalize", "input_budget": "answer", "output_budget": "answer"},
        "observability": {
            "span_kind": "answer",
            "event_prefix": "finalizer",
            "summary_fields": ["answer_chars", "clarification_option_count"],
            "raw_payload": "bounded",
        },
        "max_instances": 1,
    },
    "hitl_gate": {
        "state_reads": ["hitl_policy", "hitl_gate_routes", "hitl_interrupt_counts", "route", "route_reason", "final_answer"],
        "state_writes": ["hitl_gate_route", "hitl_gate_routes", "hitl_decisions", "hitl_interrupt_counts", "human_review_decision"],
        "prompt_slots": [],
        "context_policy": {"mode": "interrupt", "input_budget": "bounded_summary", "output_budget": "decision"},
        "observability": {
            "span_kind": "human_review",
            "event_prefix": "hitl_gate",
            "summary_fields": ["action", "gate_id", "target_node_id"],
            "raw_payload": "bounded",
        },
        "max_instances": 8,
    },
}

for _node_type, _metadata in _NODE_CATALOG_METADATA.items():
    NODE_CATALOG[_node_type].update(deepcopy(_metadata))

REQUIRED_NODE_CATALOG_KEYS = {
    "display_name",
    "category",
    "capabilities",
    "allowed_route_functions",
    "allowed_tool_contract_ids",
    "allowed_parent_types",
    "allowed_child_types",
    "limits",
    "state_reads",
    "state_writes",
    "prompt_slots",
    "context_policy",
    "observability",
    "max_instances",
}


def collect_node_catalog_errors(catalog: Dict[str, Dict[str, Any]] | None = None) -> list[str]:
    errors: list[str] = []
    source = catalog if isinstance(catalog, dict) else NODE_CATALOG
    for node_type, metadata in sorted(source.items()):
        if not isinstance(metadata, dict):
            errors.append(f"{node_type} metadata must be an object")
            continue
        missing = sorted(REQUIRED_NODE_CATALOG_KEYS - set(metadata))
        if missing:
            errors.append(f"{node_type} missing catalog keys: {', '.join(missing)}")

        for key in (
            "capabilities",
            "allowed_route_functions",
            "allowed_tool_contract_ids",
            "allowed_parent_types",
            "allowed_child_types",
            "state_reads",
            "state_writes",
            "prompt_slots",
        ):
            value = metadata.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                errors.append(f"{node_type}.{key} must be a list of non-empty strings")

        limits = metadata.get("limits")
        if not isinstance(limits, dict):
            errors.append(f"{node_type}.limits must be an object")

        context_policy = metadata.get("context_policy")
        if not isinstance(context_policy, dict):
            errors.append(f"{node_type}.context_policy must be an object")
        else:
            for key in ("mode", "input_budget", "output_budget"):
                if not isinstance(context_policy.get(key), str) or not context_policy.get(key):
                    errors.append(f"{node_type}.context_policy.{key} must be a non-empty string")

        observability = metadata.get("observability")
        if not isinstance(observability, dict):
            errors.append(f"{node_type}.observability must be an object")
        else:
            for key in ("span_kind", "event_prefix", "raw_payload"):
                if not isinstance(observability.get(key), str) or not observability.get(key):
                    errors.append(f"{node_type}.observability.{key} must be a non-empty string")
            summary_fields = observability.get("summary_fields")
            if not isinstance(summary_fields, list) or not all(isinstance(item, str) and item for item in summary_fields):
                errors.append(f"{node_type}.observability.summary_fields must be a list of non-empty strings")

        max_instances = metadata.get("max_instances")
        if not isinstance(max_instances, int) or isinstance(max_instances, bool) or max_instances < 1:
            errors.append(f"{node_type}.max_instances must be a positive integer")
    return errors


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


def node_type_default_max_visits(node_type: str) -> int:
    metadata = NODE_CATALOG.get(node_type) or {}
    limits = metadata.get("limits") if isinstance(metadata.get("limits"), dict) else {}
    try:
        return max(1, int(limits.get("default_max_visits", 1)))
    except (TypeError, ValueError):
        return 1


def node_type_max_visits(node_type: str) -> int:
    metadata = NODE_CATALOG.get(node_type) or {}
    limits = metadata.get("limits") if isinstance(metadata.get("limits"), dict) else {}
    default = node_type_default_max_visits(node_type)
    try:
        return max(default, int(limits.get("max_visits", default)))
    except (TypeError, ValueError):
        return default
