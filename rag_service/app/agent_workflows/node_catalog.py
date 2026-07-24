from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from app.agent_workflows.enums import (
    ContextBudget,
    ContextPolicyMode,
    GraphSentinel,
    NodeCapability,
    NodeCategory,
    RouteFunctionId,
    ToolContractId,
    TracePayloadMode,
    TraceSpanKind,
    WorkflowNodeType,
)
from app.models.llm_server_client import REPLANS_LIMIT


NODE_CONTEXT_LOADER = WorkflowNodeType.CONTEXT_LOADER.value
NODE_ROUTER = WorkflowNodeType.ROUTER.value
NODE_PLANNER = WorkflowNodeType.PLANNER.value
NODE_RETRIEVAL_WORKER = WorkflowNodeType.RETRIEVAL_WORKER.value
NODE_MEMORY_WORKER = WorkflowNodeType.MEMORY_WORKER.value
NODE_TIMELINE_WORKER = WorkflowNodeType.TIMELINE_WORKER.value
NODE_WEB_WORKER = WorkflowNodeType.WEB_WORKER.value
NODE_EVIDENCE_EVALUATOR = WorkflowNodeType.EVIDENCE_EVALUATOR.value
NODE_REPLANNER = WorkflowNodeType.REPLANNER.value
NODE_DIRECT_ANSWER = WorkflowNodeType.DIRECT_ANSWER.value
NODE_SYNTHESIZER = WorkflowNodeType.SYNTHESIZER.value
NODE_FINALIZER = WorkflowNodeType.FINALIZER.value
NODE_HITL_GATE = WorkflowNodeType.HITL_GATE.value
START_NODE = GraphSentinel.START.value
END_NODE = GraphSentinel.END.value

CAT_CONTEXT = NodeCategory.CONTEXT.value
CAT_CONTROL = NodeCategory.CONTROL.value
CAT_RETRIEVAL = NodeCategory.RETRIEVAL.value
CAT_ANSWER = NodeCategory.ANSWER.value
CAT_HUMAN_REVIEW = NodeCategory.HUMAN_REVIEW.value

CAP_CONTEXT_PREFETCH = NodeCapability.CONTEXT_PREFETCH.value
CAP_ROUTE_INTENT = NodeCapability.ROUTE_INTENT.value
CAP_CLARIFY = NodeCapability.CLARIFY.value
CAP_PLAN_EXECUTION = NodeCapability.PLAN_EXECUTION.value
CAP_PLAN_REPLAN = NodeCapability.PLAN_REPLAN.value
CAP_RETRIEVAL_DOCUMENT = NodeCapability.RETRIEVAL_DOCUMENT.value
CAP_RETRIEVAL_MEMORY = NodeCapability.RETRIEVAL_MEMORY.value
CAP_RETRIEVAL_TIMELINE = NodeCapability.RETRIEVAL_TIMELINE.value
CAP_RETRIEVAL_WEB = NodeCapability.RETRIEVAL_WEB.value
CAP_EXTERNAL_RESEARCH = NodeCapability.EXTERNAL_RESEARCH.value
CAP_EVALUATE_EVIDENCE = NodeCapability.EVALUATE_EVIDENCE.value
CAP_ANSWER_DIRECT = NodeCapability.ANSWER_DIRECT.value
CAP_ANSWER_SYNTHESIZE = NodeCapability.ANSWER_SYNTHESIZE.value
CAP_ANSWER_FINAL = NodeCapability.ANSWER_FINAL.value
CAP_HITL_INTERRUPT = NodeCapability.HITL_INTERRUPT.value

ROUTE_ROUTER = RouteFunctionId.ROUTER.value
ROUTE_PLANNER = RouteFunctionId.PLANNER.value
ROUTE_EVALUATOR = RouteFunctionId.EVALUATOR.value
ROUTE_HITL_GATE = RouteFunctionId.HITL_GATE.value

TOOL_THREAD_SHAPE = ToolContractId.THREAD_SHAPE.value
TOOL_DOCUMENT_EVIDENCE = ToolContractId.DOCUMENT_EVIDENCE.value
TOOL_FOCUSED_DOCUMENT_EVIDENCE = ToolContractId.FOCUSED_DOCUMENT_EVIDENCE.value
TOOL_DEEP_MEMORY = ToolContractId.DEEP_MEMORY.value
TOOL_THREAD_TIMELINE = ToolContractId.THREAD_TIMELINE.value
TOOL_LIVE_WEB_RECON = ToolContractId.LIVE_WEB_RECON.value
TOOL_WIKIPEDIA_REFERENCE = ToolContractId.WIKIPEDIA_REFERENCE.value
TOOL_WIKIDATA_REFERENCE = ToolContractId.WIKIDATA_REFERENCE.value
TOOL_ARXIV_RESEARCH = ToolContractId.ARXIV_RESEARCH.value
TOOL_PUBMED_RESEARCH = ToolContractId.PUBMED_RESEARCH.value
TOOL_SEMANTIC_SCHOLAR_RESEARCH = ToolContractId.SEMANTIC_SCHOLAR_RESEARCH.value
TOOL_STACKEXCHANGE_REFERENCE = ToolContractId.STACKEXCHANGE_REFERENCE.value
TOOL_YAHOO_FINANCE_NEWS = ToolContractId.YAHOO_FINANCE_NEWS.value
TOOL_CLARIFY_INTENT = ToolContractId.CLARIFY_INTENT.value

POLICY_PREFETCH = ContextPolicyMode.PREFETCH.value
POLICY_ROUTE = ContextPolicyMode.ROUTE.value
POLICY_PLAN = ContextPolicyMode.PLAN.value
POLICY_APPEND_EVIDENCE = ContextPolicyMode.APPEND_EVIDENCE.value
POLICY_EVALUATE_EVIDENCE = ContextPolicyMode.EVALUATE_EVIDENCE.value
POLICY_ASSEMBLE_ANSWER = ContextPolicyMode.ASSEMBLE_ANSWER.value
POLICY_FINALIZE = ContextPolicyMode.FINALIZE.value
POLICY_INTERRUPT = ContextPolicyMode.INTERRUPT.value

BUDGET_REQUEST = ContextBudget.REQUEST.value
BUDGET_BOUNDED = ContextBudget.BOUNDED.value
BUDGET_BOUNDED_PREFETCH = ContextBudget.BOUNDED_PREFETCH.value
BUDGET_TOOL_QUERY = ContextBudget.TOOL_QUERY.value
BUDGET_EVIDENCE_PACKET = ContextBudget.EVIDENCE_PACKET.value
BUDGET_BOUNDED_EVIDENCE = ContextBudget.BOUNDED_EVIDENCE.value
BUDGET_BOUNDED_SUMMARY = ContextBudget.BOUNDED_SUMMARY.value
BUDGET_DECISION = ContextBudget.DECISION.value
BUDGET_ANSWER = ContextBudget.ANSWER.value

SPAN_CONTEXT = TraceSpanKind.CONTEXT.value
SPAN_CONTROL = TraceSpanKind.CONTROL.value
SPAN_TOOL_WORKER = TraceSpanKind.TOOL_WORKER.value
SPAN_ANSWER = TraceSpanKind.ANSWER.value
SPAN_HUMAN_REVIEW = TraceSpanKind.HUMAN_REVIEW.value
RAW_PAYLOAD_BOUNDED = TracePayloadMode.BOUNDED.value


NODE_CATALOG: Dict[str, Dict[str, Any]] = {
    NODE_CONTEXT_LOADER: {
        "display_name": "Context Loader",
        "category": CAT_CONTEXT,
        "capabilities": [CAP_CONTEXT_PREFETCH],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [TOOL_THREAD_SHAPE],
        "allowed_parent_types": [START_NODE],
        "allowed_child_types": [NODE_ROUTER, NODE_PLANNER, NODE_HITL_GATE],
        "limits": {"default_max_visits": 1},
    },
    NODE_ROUTER: {
        "display_name": "Router",
        "category": CAT_CONTROL,
        "capabilities": [CAP_ROUTE_INTENT, CAP_CLARIFY],
        "allowed_route_functions": [ROUTE_ROUTER],
        "allowed_tool_contract_ids": [TOOL_CLARIFY_INTENT],
        "allowed_parent_types": [NODE_CONTEXT_LOADER, NODE_HITL_GATE],
        "allowed_child_types": [
            NODE_RETRIEVAL_WORKER,
            NODE_MEMORY_WORKER,
            NODE_TIMELINE_WORKER,
            NODE_WEB_WORKER,
            NODE_DIRECT_ANSWER,
            NODE_FINALIZER,
            NODE_HITL_GATE,
        ],
        "limits": {"default_max_visits": 1},
    },
    NODE_PLANNER: {
        "display_name": "Planner",
        "category": CAT_CONTROL,
        "capabilities": [CAP_PLAN_EXECUTION, CAP_CLARIFY],
        "allowed_route_functions": [ROUTE_PLANNER],
        "allowed_tool_contract_ids": [TOOL_CLARIFY_INTENT],
        "allowed_parent_types": [NODE_CONTEXT_LOADER, NODE_HITL_GATE],
        "allowed_child_types": [NODE_RETRIEVAL_WORKER, NODE_DIRECT_ANSWER, NODE_FINALIZER, NODE_HITL_GATE],
        "limits": {"default_max_visits": 1},
    },
    NODE_RETRIEVAL_WORKER: {
        "display_name": "Document Retrieval",
        "category": CAT_RETRIEVAL,
        "capabilities": [CAP_RETRIEVAL_DOCUMENT],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [TOOL_DOCUMENT_EVIDENCE, TOOL_FOCUSED_DOCUMENT_EVIDENCE],
        "allowed_parent_types": [NODE_ROUTER, NODE_PLANNER, NODE_REPLANNER, NODE_HITL_GATE],
        "allowed_child_types": [
            NODE_MEMORY_WORKER,
            NODE_TIMELINE_WORKER,
            NODE_WEB_WORKER,
            NODE_EVIDENCE_EVALUATOR,
            NODE_SYNTHESIZER,
            NODE_FINALIZER,
            NODE_HITL_GATE,
        ],
        "limits": {"default_max_visits": 2, "max_visits": REPLANS_LIMIT + 1},
    },
    NODE_MEMORY_WORKER: {
        "display_name": "Memory Retrieval",
        "category": CAT_RETRIEVAL,
        "capabilities": [CAP_RETRIEVAL_MEMORY],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [TOOL_DEEP_MEMORY],
        "allowed_parent_types": [NODE_ROUTER, NODE_RETRIEVAL_WORKER, NODE_PLANNER, NODE_REPLANNER, NODE_HITL_GATE],
        "allowed_child_types": [
            NODE_TIMELINE_WORKER,
            NODE_WEB_WORKER,
            NODE_EVIDENCE_EVALUATOR,
            NODE_SYNTHESIZER,
            NODE_FINALIZER,
            NODE_HITL_GATE,
        ],
        "limits": {"default_max_visits": 2, "max_visits": REPLANS_LIMIT + 1},
    },
    NODE_TIMELINE_WORKER: {
        "display_name": "Timeline Retrieval",
        "category": CAT_RETRIEVAL,
        "capabilities": [CAP_RETRIEVAL_TIMELINE],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [TOOL_THREAD_TIMELINE],
        "allowed_parent_types": [NODE_ROUTER, NODE_MEMORY_WORKER, NODE_PLANNER, NODE_REPLANNER, NODE_HITL_GATE],
        "allowed_child_types": [NODE_WEB_WORKER, NODE_EVIDENCE_EVALUATOR, NODE_SYNTHESIZER, NODE_FINALIZER, NODE_HITL_GATE],
        "limits": {"default_max_visits": 2, "max_visits": REPLANS_LIMIT + 1},
    },
    NODE_WEB_WORKER: {
        "display_name": "Web Retrieval",
        "category": CAT_RETRIEVAL,
        "capabilities": [CAP_RETRIEVAL_WEB, CAP_EXTERNAL_RESEARCH],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [
            TOOL_LIVE_WEB_RECON,
            TOOL_WIKIPEDIA_REFERENCE,
            TOOL_WIKIDATA_REFERENCE,
            TOOL_ARXIV_RESEARCH,
            TOOL_PUBMED_RESEARCH,
            TOOL_SEMANTIC_SCHOLAR_RESEARCH,
            TOOL_STACKEXCHANGE_REFERENCE,
            TOOL_YAHOO_FINANCE_NEWS,
        ],
        "allowed_parent_types": [NODE_ROUTER, NODE_TIMELINE_WORKER, NODE_MEMORY_WORKER, NODE_PLANNER, NODE_REPLANNER, NODE_HITL_GATE],
        "allowed_child_types": [NODE_EVIDENCE_EVALUATOR, NODE_SYNTHESIZER, NODE_FINALIZER, NODE_HITL_GATE],
        "limits": {"default_max_visits": 2, "max_visits": REPLANS_LIMIT + 1},
    },
    NODE_EVIDENCE_EVALUATOR: {
        "display_name": "Evidence Evaluator",
        "category": CAT_CONTROL,
        "capabilities": [CAP_EVALUATE_EVIDENCE, CAP_CLARIFY],
        "allowed_route_functions": [ROUTE_EVALUATOR],
        "allowed_tool_contract_ids": [TOOL_CLARIFY_INTENT],
        "allowed_parent_types": [NODE_RETRIEVAL_WORKER, NODE_MEMORY_WORKER, NODE_TIMELINE_WORKER, NODE_WEB_WORKER, NODE_HITL_GATE],
        "allowed_child_types": [NODE_SYNTHESIZER, NODE_REPLANNER, NODE_HITL_GATE],
        "limits": {"default_max_visits": 2, "max_visits": REPLANS_LIMIT + 1},
    },
    NODE_REPLANNER: {
        "display_name": "Replanner",
        "category": CAT_CONTROL,
        "capabilities": [CAP_PLAN_REPLAN, CAP_CLARIFY],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [TOOL_CLARIFY_INTENT],
        "allowed_parent_types": [NODE_EVIDENCE_EVALUATOR, NODE_HITL_GATE],
        "allowed_child_types": [NODE_RETRIEVAL_WORKER, NODE_MEMORY_WORKER, NODE_TIMELINE_WORKER, NODE_WEB_WORKER, NODE_HITL_GATE],
        "limits": {"default_max_visits": 1, "max_visits": REPLANS_LIMIT},
    },
    NODE_DIRECT_ANSWER: {
        "display_name": "Direct Answer",
        "category": CAT_ANSWER,
        "capabilities": [CAP_ANSWER_DIRECT],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [],
        "allowed_parent_types": [NODE_ROUTER, NODE_PLANNER, NODE_HITL_GATE],
        "allowed_child_types": [NODE_FINALIZER, NODE_HITL_GATE],
        "limits": {"default_max_visits": 1},
    },
    NODE_SYNTHESIZER: {
        "display_name": "Synthesizer",
        "category": CAT_ANSWER,
        "capabilities": [CAP_ANSWER_SYNTHESIZE],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [],
        "allowed_parent_types": [
            NODE_RETRIEVAL_WORKER,
            NODE_MEMORY_WORKER,
            NODE_TIMELINE_WORKER,
            NODE_WEB_WORKER,
            NODE_EVIDENCE_EVALUATOR,
            NODE_HITL_GATE,
        ],
        "allowed_child_types": [NODE_FINALIZER, NODE_HITL_GATE],
        "limits": {"default_max_visits": 1},
    },
    NODE_FINALIZER: {
        "display_name": "Finalizer",
        "category": CAT_ANSWER,
        "capabilities": [CAP_ANSWER_FINAL, CAP_CLARIFY],
        "allowed_route_functions": [],
        "allowed_tool_contract_ids": [TOOL_CLARIFY_INTENT],
        "allowed_parent_types": [
            NODE_ROUTER,
            NODE_PLANNER,
            NODE_RETRIEVAL_WORKER,
            NODE_MEMORY_WORKER,
            NODE_TIMELINE_WORKER,
            NODE_WEB_WORKER,
            NODE_DIRECT_ANSWER,
            NODE_SYNTHESIZER,
            NODE_HITL_GATE,
        ],
        "allowed_child_types": [NODE_HITL_GATE, END_NODE],
        "limits": {"default_max_visits": 1},
    },
    NODE_HITL_GATE: {
        "display_name": "HITL Gate",
        "category": CAT_HUMAN_REVIEW,
        "capabilities": [CAP_HITL_INTERRUPT],
        "allowed_route_functions": [ROUTE_HITL_GATE],
        "allowed_tool_contract_ids": [],
        "allowed_parent_types": [START_NODE, NODE_CONTEXT_LOADER, NODE_ROUTER, NODE_PLANNER, NODE_RETRIEVAL_WORKER, NODE_MEMORY_WORKER, NODE_TIMELINE_WORKER, NODE_WEB_WORKER, NODE_EVIDENCE_EVALUATOR, NODE_REPLANNER, NODE_DIRECT_ANSWER, NODE_SYNTHESIZER, NODE_FINALIZER],
        "allowed_child_types": [NODE_ROUTER, NODE_PLANNER, NODE_RETRIEVAL_WORKER, NODE_MEMORY_WORKER, NODE_TIMELINE_WORKER, NODE_WEB_WORKER, NODE_EVIDENCE_EVALUATOR, NODE_REPLANNER, NODE_DIRECT_ANSWER, NODE_SYNTHESIZER, NODE_FINALIZER, END_NODE],
        "limits": {"default_max_visits": 1},
    },
}


_NODE_CATALOG_METADATA: Dict[str, Dict[str, Any]] = {
    NODE_CONTEXT_LOADER: {
        "state_reads": ["thread_id", "question", "embedding_model", "context_window", "use_web_search", "use_reranker"],
        "state_writes": ["pre_fetch_bundle", "document_sources", "web_sources", "used_chat_ids"],
        "prompt_slots": [],
        "context_policy": {"mode": POLICY_PREFETCH, "input_budget": BUDGET_REQUEST, "output_budget": BUDGET_BOUNDED},
        "observability": {
            "span_kind": SPAN_CONTEXT,
            "event_prefix": NODE_CONTEXT_LOADER,
            "summary_fields": ["document_source_count", "web_source_count", "used_chat_id_count"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 1,
    },
    NODE_ROUTER: {
        "state_reads": ["question", "pre_fetch_bundle", "use_web_search", "client_timezone", "client_locale", "client_now_iso"],
        "state_writes": ["route", "route_reason", "clarification_options"],
        "prompt_slots": [NODE_ROUTER],
        "context_policy": {"mode": POLICY_ROUTE, "input_budget": BUDGET_BOUNDED_PREFETCH, "output_budget": BUDGET_DECISION},
        "observability": {
            "span_kind": SPAN_CONTROL,
            "event_prefix": NODE_ROUTER,
            "summary_fields": ["route", "route_reason"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 1,
    },
    NODE_PLANNER: {
        "state_reads": ["question", "pre_fetch_bundle", "use_web_search", "client_timezone", "client_locale", "client_now_iso"],
        "state_writes": ["route", "route_reason", "execution_plan", "clarification_options"],
        "prompt_slots": [NODE_PLANNER],
        "context_policy": {"mode": POLICY_PLAN, "input_budget": BUDGET_BOUNDED_PREFETCH, "output_budget": BUDGET_DECISION},
        "observability": {
            "span_kind": SPAN_CONTROL,
            "event_prefix": NODE_PLANNER,
            "summary_fields": ["route", "route_reason", "execution_plan"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 1,
    },
    NODE_RETRIEVAL_WORKER: {
        "state_reads": ["question", "thread_id", "embedding_model", "use_reranker", "execution_plan", "evidence"],
        "state_writes": ["evidence", "evidence_packets", "document_sources", "web_sources", "tool_events"],
        "prompt_slots": [],
        "context_policy": {"mode": POLICY_APPEND_EVIDENCE, "input_budget": BUDGET_TOOL_QUERY, "output_budget": BUDGET_EVIDENCE_PACKET},
        "observability": {
            "span_kind": SPAN_TOOL_WORKER,
            "event_prefix": NODE_RETRIEVAL_WORKER,
            "summary_fields": ["document_source_count", "web_source_count", "evidence_chars"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 4,
    },
    NODE_MEMORY_WORKER: {
        "state_reads": ["question", "thread_id", "embedding_model", "execution_plan", "evidence"],
        "state_writes": ["evidence", "evidence_packets", "used_chat_ids", "tool_events"],
        "prompt_slots": [],
        "context_policy": {"mode": POLICY_APPEND_EVIDENCE, "input_budget": BUDGET_TOOL_QUERY, "output_budget": BUDGET_EVIDENCE_PACKET},
        "observability": {
            "span_kind": SPAN_TOOL_WORKER,
            "event_prefix": NODE_MEMORY_WORKER,
            "summary_fields": ["used_chat_id_count", "evidence_chars"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 4,
    },
    NODE_TIMELINE_WORKER: {
        "state_reads": ["question", "thread_id", "embedding_model", "execution_plan", "evidence"],
        "state_writes": ["evidence", "evidence_packets", "tool_events"],
        "prompt_slots": [],
        "context_policy": {"mode": POLICY_APPEND_EVIDENCE, "input_budget": BUDGET_TOOL_QUERY, "output_budget": BUDGET_EVIDENCE_PACKET},
        "observability": {
            "span_kind": SPAN_TOOL_WORKER,
            "event_prefix": NODE_TIMELINE_WORKER,
            "summary_fields": ["timeline_event_count", "evidence_chars"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 4,
    },
    NODE_WEB_WORKER: {
        "state_reads": ["question", "use_web_search", "execution_plan", "evidence"],
        "state_writes": ["evidence", "evidence_packets", "web_sources", "tool_events"],
        "prompt_slots": ["web_search_mandate"],
        "context_policy": {"mode": POLICY_APPEND_EVIDENCE, "input_budget": BUDGET_TOOL_QUERY, "output_budget": BUDGET_EVIDENCE_PACKET},
        "observability": {
            "span_kind": SPAN_TOOL_WORKER,
            "event_prefix": NODE_WEB_WORKER,
            "summary_fields": ["web_source_count", "evidence_chars"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 4,
    },
    NODE_EVIDENCE_EVALUATOR: {
        "state_reads": ["question", "evidence", "evidence_packets", "replan_count", "replans"],
        "state_writes": ["evaluator_report", "evidence_gaps", "evaluation_confidence", "evaluator_route"],
        "prompt_slots": ["evaluator"],
        "context_policy": {"mode": POLICY_EVALUATE_EVIDENCE, "input_budget": BUDGET_BOUNDED_EVIDENCE, "output_budget": BUDGET_DECISION},
        "observability": {
            "span_kind": SPAN_CONTROL,
            "event_prefix": NODE_EVIDENCE_EVALUATOR,
            "summary_fields": ["evaluator_route", "evaluation_confidence", "evidence_gaps"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 2,
    },
    NODE_REPLANNER: {
        "state_reads": ["question", "evidence", "evaluator_report", "replan_count", "replans"],
        "state_writes": ["execution_plan", "replan_count", "replan_reason", "replan_history"],
        "prompt_slots": [NODE_REPLANNER],
        "context_policy": {"mode": POLICY_PLAN, "input_budget": BUDGET_BOUNDED_EVIDENCE, "output_budget": BUDGET_DECISION},
        "observability": {
            "span_kind": SPAN_CONTROL,
            "event_prefix": NODE_REPLANNER,
            "summary_fields": ["execution_plan", "replan_count", "replan_reason"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 1,
    },
    NODE_DIRECT_ANSWER: {
        "state_reads": ["question", "pre_fetch_bundle", "route_reason"],
        "state_writes": ["final_answer", "reasoning", "reasoning_available", "reasoning_format"],
        "prompt_slots": ["final_answer"],
        "context_policy": {"mode": POLICY_ASSEMBLE_ANSWER, "input_budget": BUDGET_BOUNDED_PREFETCH, "output_budget": BUDGET_ANSWER},
        "observability": {
            "span_kind": SPAN_ANSWER,
            "event_prefix": NODE_DIRECT_ANSWER,
            "summary_fields": ["answer_chars"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 1,
    },
    NODE_SYNTHESIZER: {
        "state_reads": ["question", "evidence", "evidence_packets", "document_sources", "web_sources", "used_chat_ids"],
        "state_writes": ["final_answer", "reasoning", "reasoning_available", "reasoning_format"],
        "prompt_slots": ["final_answer"],
        "context_policy": {"mode": POLICY_ASSEMBLE_ANSWER, "input_budget": BUDGET_BOUNDED_EVIDENCE, "output_budget": BUDGET_ANSWER},
        "observability": {
            "span_kind": SPAN_ANSWER,
            "event_prefix": NODE_SYNTHESIZER,
            "summary_fields": ["answer_chars", "document_source_count", "web_source_count"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 1,
    },
    NODE_FINALIZER: {
        "state_reads": ["final_answer", "clarification_options", "document_sources", "web_sources", "used_chat_ids"],
        "state_writes": ["final_answer", "reasoning", "reasoning_available", "reasoning_format"],
        "prompt_slots": ["final_answer"],
        "context_policy": {"mode": POLICY_FINALIZE, "input_budget": BUDGET_ANSWER, "output_budget": BUDGET_ANSWER},
        "observability": {
            "span_kind": SPAN_ANSWER,
            "event_prefix": NODE_FINALIZER,
            "summary_fields": ["answer_chars", "clarification_option_count"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
        },
        "max_instances": 1,
    },
    NODE_HITL_GATE: {
        "state_reads": ["hitl_policy", "hitl_gate_routes", "hitl_interrupt_counts", "route", "route_reason", "final_answer"],
        "state_writes": ["hitl_gate_route", "hitl_gate_routes", "hitl_decisions", "hitl_interrupt_counts", "human_review_decision"],
        "prompt_slots": [],
        "context_policy": {"mode": POLICY_INTERRUPT, "input_budget": BUDGET_BOUNDED_SUMMARY, "output_budget": BUDGET_DECISION},
        "observability": {
            "span_kind": SPAN_HUMAN_REVIEW,
            "event_prefix": NODE_HITL_GATE,
            "summary_fields": ["action", "gate_id", "target_node_id"],
            "raw_payload": RAW_PAYLOAD_BOUNDED,
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


NODE_UI_METADATA: Dict[str, Dict[str, Any]] = {
    NODE_CONTEXT_LOADER: {
        "summary": "Loads recent conversation and relevant PDF context before the workflow decides what to do.",
        "use_when": "Use this as the first step of every workflow.",
        "category_label": "Start & context",
        "icon": "context",
        "keywords": ["start", "context", "thread", "pdf", "history"],
        "input_label": "Request",
        "output_label": "Loaded context",
        "uses_llm": False,
        "uses_tools": True,
    },
    NODE_ROUTER: {
        "summary": "Chooses one answer path based on the user's question.",
        "use_when": "Use for fast workflows that normally need one source.",
        "category_label": "Decide",
        "icon": "route",
        "keywords": ["route", "branch", "choose", "intent"],
        "input_label": "Loaded context",
        "output_label": "Decision",
        "uses_llm": True,
        "uses_tools": False,
    },
    NODE_PLANNER: {
        "summary": "Creates a retrieval plan when the answer may need several sources.",
        "use_when": "Use for multi-step research workflows.",
        "category_label": "Decide",
        "icon": "plan",
        "keywords": ["plan", "steps", "research", "multi-source"],
        "input_label": "Loaded context",
        "output_label": "Execution plan",
        "uses_llm": True,
        "uses_tools": False,
    },
    NODE_RETRIEVAL_WORKER: {
        "summary": "Searches PDFs and other documents attached to the thread.",
        "use_when": "Use when the answer should be grounded in uploaded documents.",
        "category_label": "Retrieve",
        "icon": "document",
        "keywords": ["pdf", "document", "search", "evidence"],
        "input_label": "Question or plan",
        "output_label": "Document evidence",
        "uses_llm": False,
        "uses_tools": True,
    },
    NODE_MEMORY_WORKER: {
        "summary": "Searches earlier questions and answers in this conversation.",
        "use_when": "Use when prior discussion may contain relevant context.",
        "category_label": "Retrieve",
        "icon": "memory",
        "keywords": ["memory", "conversation", "history", "previous"],
        "input_label": "Question or plan",
        "output_label": "Conversation evidence",
        "uses_llm": False,
        "uses_tools": True,
    },
    NODE_TIMELINE_WORKER: {
        "summary": "Finds thread events and evidence in chronological order.",
        "use_when": "Use for questions about what happened and when.",
        "category_label": "Retrieve",
        "icon": "timeline",
        "keywords": ["timeline", "chronology", "events", "dates"],
        "input_label": "Question or plan",
        "output_label": "Timeline evidence",
        "uses_llm": False,
        "uses_tools": True,
    },
    NODE_WEB_WORKER: {
        "summary": "Searches approved external sources for current information.",
        "use_when": "Use when uploaded documents may not contain current facts.",
        "category_label": "Retrieve",
        "icon": "web",
        "keywords": ["web", "internet", "current", "external", "research"],
        "input_label": "Question or plan",
        "output_label": "Web evidence",
        "uses_llm": False,
        "uses_tools": True,
        "external_side_effect": True,
    },
    NODE_EVIDENCE_EVALUATOR: {
        "summary": "Checks whether the collected evidence is sufficient to answer.",
        "use_when": "Use before synthesis when weak evidence should trigger another search.",
        "category_label": "Evaluate",
        "icon": "evaluate",
        "keywords": ["evaluate", "quality", "confidence", "evidence"],
        "input_label": "Collected evidence",
        "output_label": "Evidence decision",
        "uses_llm": True,
        "uses_tools": False,
    },
    NODE_REPLANNER: {
        "summary": "Chooses another bounded search step when evidence is missing.",
        "use_when": "Use after an evidence evaluator in a workflow that supports replanning.",
        "category_label": "Decide",
        "icon": "replan",
        "keywords": ["replan", "retry", "gaps", "loop"],
        "input_label": "Evidence gaps",
        "output_label": "Revised plan",
        "uses_llm": True,
        "uses_tools": False,
    },
    NODE_DIRECT_ANSWER: {
        "summary": "Answers without retrieving additional evidence.",
        "use_when": "Use as the direct branch from a Router or Planner.",
        "category_label": "Answer",
        "icon": "answer",
        "keywords": ["direct", "answer", "no retrieval"],
        "input_label": "Question and context",
        "output_label": "Draft answer",
        "uses_llm": True,
        "uses_tools": False,
    },
    NODE_SYNTHESIZER: {
        "summary": "Combines gathered evidence into a grounded draft answer.",
        "use_when": "Use after retrieval or evidence evaluation.",
        "category_label": "Answer",
        "icon": "synthesize",
        "keywords": ["combine", "synthesize", "evidence", "draft"],
        "input_label": "Evidence",
        "output_label": "Draft answer",
        "uses_llm": True,
        "uses_tools": False,
    },
    NODE_FINALIZER: {
        "summary": "Produces the final user-facing answer or clarification.",
        "use_when": "Use as the last executable step before End.",
        "category_label": "Answer",
        "icon": "finish",
        "keywords": ["final", "finish", "clarify", "response"],
        "input_label": "Draft or decision",
        "output_label": "Final response",
        "uses_llm": False,
        "uses_tools": False,
    },
    NODE_HITL_GATE: {
        "summary": "Pauses the workflow so a person can approve, edit, choose, or reject.",
        "use_when": "Use before or after a sensitive action.",
        "category_label": "Human review",
        "icon": "human",
        "keywords": ["human", "approval", "review", "pause", "hitl"],
        "input_label": "Proposed action",
        "output_label": "Human decision",
        "uses_llm": False,
        "uses_tools": False,
    },
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
    catalog = deepcopy(NODE_CATALOG)
    for node_type, metadata in catalog.items():
        ui = deepcopy(NODE_UI_METADATA.get(node_type, {}))
        ui["field_guidance"] = {
            "purpose": "Describe this step in language your team will recognize.",
            "tools": "Choose only the capabilities this step is allowed to call.",
            "branches": "Connect every named outcome before running the workflow.",
        }
        metadata["ui"] = ui
    return catalog


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
