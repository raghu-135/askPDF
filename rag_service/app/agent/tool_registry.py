"""
tool_registry.py - Single source of truth for user-facing tool metadata.

This powers:
  - Tool catalog in the UI
  - Tool playbook injected into the system prompt

Keep this file DRY and free of runtime imports to avoid cycles.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from app.agent.tool_contract import ToolWarningCode
from app.agent_workflows.enums import NodeCapability, NodeCategory, ToolContractId, ToolName, WorkflowNodeType


NODE_CONTEXT_LOADER = WorkflowNodeType.CONTEXT_LOADER.value
NODE_ROUTER = WorkflowNodeType.ROUTER.value
NODE_PLANNER = WorkflowNodeType.PLANNER.value
NODE_RETRIEVAL_WORKER = WorkflowNodeType.RETRIEVAL_WORKER.value
NODE_MEMORY_WORKER = WorkflowNodeType.MEMORY_WORKER.value
NODE_LONG_TERM_MEMORY_WORKER = WorkflowNodeType.LONG_TERM_MEMORY_WORKER.value
NODE_TIMELINE_WORKER = WorkflowNodeType.TIMELINE_WORKER.value
NODE_WEB_WORKER = WorkflowNodeType.WEB_WORKER.value
NODE_EVIDENCE_EVALUATOR = WorkflowNodeType.EVIDENCE_EVALUATOR.value
NODE_REPLANNER = WorkflowNodeType.REPLANNER.value
NODE_FINALIZER = WorkflowNodeType.FINALIZER.value

CAT_CONTEXT = NodeCategory.CONTEXT.value
CAT_CONTROL = NodeCategory.CONTROL.value
CAT_RETRIEVAL = NodeCategory.RETRIEVAL.value
CAT_MEMORY = NodeCategory.MEMORY.value
CAT_TIMELINE = NodeCategory.TIMELINE.value
CAT_WEB = NodeCategory.WEB.value
CAT_EXTERNAL_RESEARCH = NodeCategory.EXTERNAL_RESEARCH.value

CAP_CONTEXT_PREFETCH = NodeCapability.CONTEXT_PREFETCH.value
CAP_ROUTE_INTENT = NodeCapability.ROUTE_INTENT.value
CAP_CLARIFY = NodeCapability.CLARIFY.value
CAP_RETRIEVAL_DOCUMENT = NodeCapability.RETRIEVAL_DOCUMENT.value
CAP_RETRIEVAL_MEMORY = NodeCapability.RETRIEVAL_MEMORY.value
CAP_RETRIEVAL_TIMELINE = NodeCapability.RETRIEVAL_TIMELINE.value
CAP_RETRIEVAL_WEB = NodeCapability.RETRIEVAL_WEB.value
CAP_EXTERNAL_RESEARCH = NodeCapability.EXTERNAL_RESEARCH.value

TOOL_THREAD_SHAPE = ToolContractId.THREAD_SHAPE.value
TOOL_DOCUMENT_EVIDENCE = ToolContractId.DOCUMENT_EVIDENCE.value
TOOL_FOCUSED_DOCUMENT_EVIDENCE = ToolContractId.FOCUSED_DOCUMENT_EVIDENCE.value
TOOL_DEEP_MEMORY = ToolContractId.DEEP_MEMORY.value
TOOL_MEMORY_RECALL = ToolContractId.MEMORY_RECALL.value
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

TOOL_NAME_GET_THREAD_SHAPE = ToolName.GET_THREAD_SHAPE.value
TOOL_NAME_SEARCH_DOCUMENTS = ToolName.SEARCH_DOCUMENTS.value
TOOL_NAME_SEARCH_DOCUMENT_BY_ID = ToolName.SEARCH_DOCUMENT_BY_ID.value
TOOL_NAME_SEARCH_CONVERSATION_HISTORY = ToolName.SEARCH_CONVERSATION_HISTORY.value
TOOL_NAME_SEARCH_LONG_TERM_MEMORY = ToolName.SEARCH_LONG_TERM_MEMORY.value
TOOL_NAME_SEARCH_THREAD_TIMELINE = ToolName.SEARCH_THREAD_TIMELINE.value
TOOL_NAME_SEARCH_WEB = ToolName.SEARCH_WEB.value
TOOL_NAME_WIKIPEDIA = ToolName.WIKIPEDIA.value
TOOL_NAME_WIKIDATA = ToolName.WIKIDATA.value
TOOL_NAME_ARXIV = ToolName.ARXIV.value
TOOL_NAME_PUB_MED = ToolName.PUB_MED.value
TOOL_NAME_PUBMED = ToolName.PUBMED.value
TOOL_NAME_SEMANTIC_SCHOLAR_LEGACY = ToolName.SEMANTIC_SCHOLAR_LEGACY.value
TOOL_NAME_SEMANTIC_SCHOLAR = ToolName.SEMANTIC_SCHOLAR.value
TOOL_NAME_STACK_EXCHANGE = ToolName.STACK_EXCHANGE.value
TOOL_NAME_YAHOO_FINANCE_NEWS = ToolName.YAHOO_FINANCE_NEWS.value
TOOL_NAME_ASK_FOR_CLARIFICATION = ToolName.ASK_FOR_CLARIFICATION.value


TOOL_CONTRACT_METADATA: Dict[str, Dict[str, Any]] = {
    TOOL_NAME_GET_THREAD_SHAPE: {
        "id": TOOL_THREAD_SHAPE,
        "category": CAT_CONTEXT,
        "allowed_caller_nodes": [NODE_CONTEXT_LOADER, NODE_ROUTER, NODE_RETRIEVAL_WORKER, NODE_MEMORY_WORKER, NODE_LONG_TERM_MEMORY_WORKER, NODE_TIMELINE_WORKER],
        "allowed_node_types": [NODE_CONTEXT_LOADER, NODE_ROUTER, NODE_RETRIEVAL_WORKER, NODE_MEMORY_WORKER, NODE_LONG_TERM_MEMORY_WORKER, NODE_TIMELINE_WORKER],
        "required_node_capabilities": [CAP_CONTEXT_PREFETCH, CAP_ROUTE_INTENT, CAP_RETRIEVAL_DOCUMENT, CAP_RETRIEVAL_MEMORY, CAP_RETRIEVAL_TIMELINE],
        "artifact_keys": [TOOL_THREAD_SHAPE],
        "warning_codes": [ToolWarningCode.MISSING_THREAD_ID],
    },
    TOOL_NAME_SEARCH_DOCUMENTS: {
        "id": TOOL_DOCUMENT_EVIDENCE,
        "category": CAT_RETRIEVAL,
        "allowed_caller_nodes": [NODE_RETRIEVAL_WORKER],
        "allowed_node_types": [NODE_RETRIEVAL_WORKER],
        "required_node_capabilities": [CAP_RETRIEVAL_DOCUMENT],
        "artifact_keys": ["document_sources", "web_sources"],
        "warning_codes": [
            ToolWarningCode.MISSING_THREAD_CONTEXT,
            ToolWarningCode.NO_THREAD_DOCUMENTS,
            ToolWarningCode.MISSING_DOCUMENT_VECTORS,
            ToolWarningCode.NO_RELEVANT_CONTENT,
        ],
    },
    TOOL_NAME_SEARCH_DOCUMENT_BY_ID: {
        "id": TOOL_FOCUSED_DOCUMENT_EVIDENCE,
        "category": CAT_RETRIEVAL,
        "allowed_caller_nodes": [NODE_RETRIEVAL_WORKER],
        "allowed_node_types": [NODE_RETRIEVAL_WORKER],
        "required_node_capabilities": [CAP_RETRIEVAL_DOCUMENT],
        "artifact_keys": ["document_sources"],
        "warning_codes": [ToolWarningCode.MISSING_THREAD_CONTEXT, ToolWarningCode.MISSING_DOCUMENT_VECTORS, ToolWarningCode.NO_RELEVANT_CONTENT],
    },
    TOOL_NAME_SEARCH_CONVERSATION_HISTORY: {
        "id": TOOL_DEEP_MEMORY,
        "category": CAT_MEMORY,
        "allowed_caller_nodes": [NODE_MEMORY_WORKER],
        "allowed_node_types": [NODE_MEMORY_WORKER],
        "required_node_capabilities": [CAP_RETRIEVAL_MEMORY],
        "artifact_keys": ["used_chat_ids"],
        "warning_codes": [ToolWarningCode.MISSING_THREAD_CONTEXT, ToolWarningCode.NO_RELEVANT_CONVERSATION_HISTORY],
    },
    TOOL_NAME_SEARCH_LONG_TERM_MEMORY: {
        "id": TOOL_MEMORY_RECALL,
        "category": CAT_MEMORY,
        "allowed_caller_nodes": [NODE_LONG_TERM_MEMORY_WORKER],
        "allowed_node_types": [NODE_LONG_TERM_MEMORY_WORKER],
        "required_node_capabilities": [CAP_RETRIEVAL_MEMORY],
        "artifact_keys": ["memory_refs", "memory_scopes", "memory_scope_policy"],
        "warning_codes": [ToolWarningCode.MISSING_THREAD_CONTEXT, ToolWarningCode.NO_RELEVANT_MEMORY],
    },
    TOOL_NAME_SEARCH_THREAD_TIMELINE: {
        "id": TOOL_THREAD_TIMELINE,
        "category": CAT_TIMELINE,
        "allowed_caller_nodes": [NODE_TIMELINE_WORKER],
        "allowed_node_types": [NODE_TIMELINE_WORKER],
        "required_node_capabilities": [CAP_RETRIEVAL_TIMELINE],
        "artifact_keys": ["timeline_events"],
        "warning_codes": [ToolWarningCode.MISSING_THREAD_CONTEXT, ToolWarningCode.NO_TIMELINE_EVENTS],
    },
    TOOL_NAME_SEARCH_WEB: {
        "id": TOOL_LIVE_WEB_RECON,
        "category": CAT_WEB,
        "allowed_caller_nodes": [NODE_WEB_WORKER],
        "allowed_node_types": [NODE_WEB_WORKER],
        "required_node_capabilities": [CAP_RETRIEVAL_WEB],
        "artifact_keys": ["web_sources"],
        "warning_codes": [ToolWarningCode.WEB_SEARCH_DISABLED, ToolWarningCode.NO_USABLE_WEB_RESULTS],
    },
    TOOL_NAME_WIKIPEDIA: {
        "id": TOOL_WIKIPEDIA_REFERENCE,
        "category": CAT_EXTERNAL_RESEARCH,
        "allowed_caller_nodes": [NODE_WEB_WORKER],
        "allowed_node_types": [NODE_WEB_WORKER],
        "required_node_capabilities": [CAP_EXTERNAL_RESEARCH],
        "artifact_keys": ["provider_tool"],
        "warning_codes": [ToolWarningCode.EMPTY_EXTERNAL_TOOL_RESULT],
    },
    TOOL_NAME_WIKIDATA: {
        "id": TOOL_WIKIDATA_REFERENCE,
        "category": CAT_EXTERNAL_RESEARCH,
        "allowed_caller_nodes": [NODE_WEB_WORKER],
        "allowed_node_types": [NODE_WEB_WORKER],
        "required_node_capabilities": [CAP_EXTERNAL_RESEARCH],
        "artifact_keys": ["provider_tool"],
        "warning_codes": [ToolWarningCode.EMPTY_EXTERNAL_TOOL_RESULT],
    },
    TOOL_NAME_ARXIV: {
        "id": TOOL_ARXIV_RESEARCH,
        "category": CAT_EXTERNAL_RESEARCH,
        "allowed_caller_nodes": [NODE_WEB_WORKER],
        "allowed_node_types": [NODE_WEB_WORKER],
        "required_node_capabilities": [CAP_EXTERNAL_RESEARCH],
        "artifact_keys": ["provider_tool"],
        "warning_codes": [ToolWarningCode.EMPTY_EXTERNAL_TOOL_RESULT],
    },
    TOOL_NAME_PUB_MED: {
        "id": TOOL_PUBMED_RESEARCH,
        "category": CAT_EXTERNAL_RESEARCH,
        "allowed_caller_nodes": [NODE_WEB_WORKER],
        "allowed_node_types": [NODE_WEB_WORKER],
        "required_node_capabilities": [CAP_EXTERNAL_RESEARCH],
        "artifact_keys": ["provider_tool"],
        "warning_codes": [ToolWarningCode.EMPTY_EXTERNAL_TOOL_RESULT],
    },
    TOOL_NAME_PUBMED: {
        "id": TOOL_PUBMED_RESEARCH,
        "category": CAT_EXTERNAL_RESEARCH,
        "allowed_caller_nodes": [NODE_WEB_WORKER],
        "allowed_node_types": [NODE_WEB_WORKER],
        "required_node_capabilities": [CAP_EXTERNAL_RESEARCH],
        "artifact_keys": ["provider_tool"],
        "warning_codes": [ToolWarningCode.EMPTY_EXTERNAL_TOOL_RESULT],
    },
    TOOL_NAME_SEMANTIC_SCHOLAR_LEGACY: {
        "id": TOOL_SEMANTIC_SCHOLAR_RESEARCH,
        "category": CAT_EXTERNAL_RESEARCH,
        "allowed_caller_nodes": [NODE_WEB_WORKER],
        "allowed_node_types": [NODE_WEB_WORKER],
        "required_node_capabilities": [CAP_EXTERNAL_RESEARCH],
        "artifact_keys": ["provider_tool"],
        "warning_codes": [ToolWarningCode.EMPTY_EXTERNAL_TOOL_RESULT],
    },
    TOOL_NAME_SEMANTIC_SCHOLAR: {
        "id": TOOL_SEMANTIC_SCHOLAR_RESEARCH,
        "category": CAT_EXTERNAL_RESEARCH,
        "allowed_caller_nodes": [NODE_WEB_WORKER],
        "allowed_node_types": [NODE_WEB_WORKER],
        "required_node_capabilities": [CAP_EXTERNAL_RESEARCH],
        "artifact_keys": ["provider_tool"],
        "warning_codes": [ToolWarningCode.EMPTY_EXTERNAL_TOOL_RESULT],
    },
    TOOL_NAME_STACK_EXCHANGE: {
        "id": TOOL_STACKEXCHANGE_REFERENCE,
        "category": CAT_EXTERNAL_RESEARCH,
        "allowed_caller_nodes": [NODE_WEB_WORKER],
        "allowed_node_types": [NODE_WEB_WORKER],
        "required_node_capabilities": [CAP_EXTERNAL_RESEARCH],
        "artifact_keys": ["provider_tool"],
        "warning_codes": [ToolWarningCode.EMPTY_EXTERNAL_TOOL_RESULT],
    },
    TOOL_NAME_YAHOO_FINANCE_NEWS: {
        "id": TOOL_YAHOO_FINANCE_NEWS,
        "category": CAT_EXTERNAL_RESEARCH,
        "allowed_caller_nodes": [NODE_WEB_WORKER],
        "allowed_node_types": [NODE_WEB_WORKER],
        "required_node_capabilities": [CAP_EXTERNAL_RESEARCH],
        "artifact_keys": ["provider_tool"],
        "warning_codes": [ToolWarningCode.EMPTY_EXTERNAL_TOOL_RESULT],
    },
    TOOL_NAME_ASK_FOR_CLARIFICATION: {
        "id": TOOL_CLARIFY_INTENT,
        "category": CAT_CONTROL,
        "allowed_caller_nodes": [NODE_ROUTER, NODE_PLANNER, NODE_EVIDENCE_EVALUATOR, NODE_REPLANNER, NODE_FINALIZER],
        "allowed_node_types": [NODE_ROUTER, NODE_PLANNER, NODE_EVIDENCE_EVALUATOR, NODE_REPLANNER, NODE_FINALIZER],
        "required_node_capabilities": [CAP_CLARIFY],
        "artifact_keys": [],
        "warning_codes": [],
    },
}


def get_tool_contract_metadata(tool_name: str) -> Dict[str, Any]:
    """Return contract metadata for a canonical tool name."""

    contract = TOOL_CONTRACT_METADATA.get(tool_name)
    if not contract:
        return {}
    record = {"tool_name": tool_name, **contract}
    record["warning_codes"] = [
        item.value if isinstance(item, ToolWarningCode) else str(item)
        for item in record.get("warning_codes", [])
        if item
    ]
    friendly = TOOL_FRIENDLY_CONFIG.get(tool_name) or {}
    for key in ("display_name", "description"):
        if key in friendly:
            record[key] = friendly[key]
    return deepcopy(record)


def list_tool_contract_metadata() -> List[Dict[str, Any]]:
    """Return all known tool contract records sorted by canonical tool name."""

    records = []
    for tool_name, metadata in sorted(TOOL_CONTRACT_METADATA.items()):
        records.append(get_tool_contract_metadata(tool_name))
    return records


def collect_tool_contract_metadata_errors(records: List[Dict[str, Any]] | None = None) -> list[str]:
    """Return shape errors for the tool contract registry."""

    errors: list[str] = []
    source = records if isinstance(records, list) else list_tool_contract_metadata()
    for index, record in enumerate(source):
        if not isinstance(record, dict):
            errors.append(f"tool contract record {index} must be an object")
            continue
        tool_name = record.get("tool_name")
        prefix = str(tool_name) if isinstance(tool_name, str) and tool_name else f"tool contract record {index}"
        for key in ("tool_name", "id", "category"):
            if not isinstance(record.get(key), str) or not record.get(key):
                errors.append(f"{prefix}.{key} must be a non-empty string")
        for key in (
            "allowed_caller_nodes",
            "allowed_node_types",
            "required_node_capabilities",
            "artifact_keys",
            "warning_codes",
        ):
            value = record.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                errors.append(f"{prefix}.{key} must be a list of non-empty strings")
        if not record.get("allowed_node_types") and not record.get("required_node_capabilities"):
            errors.append(f"{prefix} must declare allowed_node_types or required_node_capabilities")
    return errors


def known_tool_contract_ids() -> set[str]:
    """Public tool IDs allowed in versioned agent workflow specs."""

    return {
        metadata["id"]
        for metadata in TOOL_CONTRACT_METADATA.values()
        if isinstance(metadata, dict) and metadata.get("id")
    }


def get_tool_contract_id(tool_name: str) -> str:
    """Return the public contract ID for a canonical tool name."""

    contract = TOOL_CONTRACT_METADATA.get(tool_name)
    if not contract or not contract.get("id"):
        raise ValueError(f"Unknown tool contract: {tool_name}")
    return str(contract["id"])


def tool_contracts_by_id() -> Dict[str, List[Dict[str, Any]]]:
    """Return contract metadata grouped by public contract ID."""

    records: Dict[str, List[Dict[str, Any]]] = {}
    for tool_name in TOOL_CONTRACT_METADATA:
        metadata = get_tool_contract_metadata(tool_name)
        contract_id = metadata.get("id")
        if isinstance(contract_id, str) and contract_id:
            records.setdefault(contract_id, []).append(metadata)
    return records


def validate_tool_call_allowed(
    tool_name: str,
    caller_node: str,
    *,
    caller_node_type: str | None = None,
    caller_capabilities: List[str] | None = None,
) -> None:
    """Raise when a graph node attempts to call a tool outside its contract."""

    contract = TOOL_CONTRACT_METADATA.get(tool_name)
    if not contract:
        raise ValueError(f"Unknown tool contract: {tool_name}")
    allowed_nodes = contract.get("allowed_caller_nodes") or []
    if caller_node in allowed_nodes:
        return

    allowed_node_types = contract.get("allowed_node_types") or []
    if caller_node_type and caller_node_type in allowed_node_types:
        return

    required_capabilities = set(contract.get("required_node_capabilities") or [])
    capabilities = set(caller_capabilities or [])
    if required_capabilities and capabilities.intersection(required_capabilities):
        return

    raise ValueError(
        f"Tool {tool_name} is not allowed from caller node {caller_node}; "
        f"allowed caller nodes: {', '.join(allowed_nodes) or 'none'}"
    )


TOOL_FRIENDLY_CONFIG = {
    TOOL_NAME_SEARCH_DOCUMENTS: {
        "id": TOOL_DOCUMENT_EVIDENCE,
        "display_name": "Document Evidence",
        "description": "Semantic search across uploaded documents and cached web snippets when the user needs evidence content. Use this when the target document is unknown, the question spans multiple documents, or cached web snippets may contain the answer. Do not use it just to answer first/latest/since/order questions; use search_thread_timeline when chronology is central.",
        "default_prompt": "Use for evidence content from uploaded documents or cached web snippets. Prefer search_document_by_id when a specific file_hash is known. Prefer search_thread_timeline when the user's wording depends on first/latest/earlier/since/before/after or mixed-source ordering.",
    },
    TOOL_NAME_SEARCH_DOCUMENT_BY_ID: {
        "id": TOOL_FOCUSED_DOCUMENT_EVIDENCE,
        "display_name": "Focused Document Evidence",
        "description": "Semantic search within one uploaded document identified by file_hash. Use this when the user names or clearly points to a specific document and thread shape provides the file_hash. Do not use it for cross-document comparison or timeline ordering unless paired with search_thread_timeline.",
        "default_prompt": "Use when a specific document is known and its file_hash is available. Keep the query focused on the requested fact. Use search_thread_timeline instead for document added-to-thread time or chronology questions.",
    },
    TOOL_NAME_SEARCH_CONVERSATION_HISTORY: {
        "id": TOOL_DEEP_MEMORY,
        "display_name": "Conversation History",
        "description": "Semantic search across past Q/A pairs in this thread when the user asks what was previously discussed or decided. Use this for topical recall where ordering is not the main question. Do not use it for first/latest/earlier/since/before/after questions; use search_thread_timeline for temporal reasoning.",
        "default_prompt": "Use for non-temporal recall of prior discussion, decisions, or answers about a topic. Avoid using it merely to reread recent turns already present in prefetch. Prefer search_thread_timeline for chronological questions.",
    },
    TOOL_NAME_SEARCH_LONG_TERM_MEMORY: {
        "id": TOOL_MEMORY_RECALL,
        "display_name": "Long-Term Memory",
        "description": "Policy-scoped search across durable user, project, and thread memories. Use this when shared project facts, durable preferences, or remembered instructions may answer the request. This does not search raw chat turns.",
        "default_prompt": "Use for durable remembered facts and preferences across the thread/project/user scopes allowed by settings. Prefer conversation history when the user asks what was said earlier in this exact thread.",
    },
    TOOL_NAME_SEARCH_THREAD_TIMELINE: {
        "id": TOOL_THREAD_TIMELINE,
        "display_name": "Thread Timeline",
        "description": "Search timestamped timeline events across conversation memory, document added-to-thread time, and cached web evidence. Use this for earliest/latest/first/earlier/since/before/after questions or when mixed-source ordering matters. It returns source-specific timestamps plus derived timeline_event_at and timeline_event_type; document timestamps mean added to this thread, not document publication time.",
        "default_prompt": "Use when the answer depends on chronology, recency, sequence, or comparing event times across conversation, documents, and cached web. Set order=oldest for first/earliest, order=newest for latest/recent, and sources to narrow the search when the user names a source class. Do not use it for ordinary semantic evidence lookup where time is irrelevant.",
    },
    TOOL_NAME_SEARCH_WEB: {
        "id": TOOL_LIVE_WEB_RECON,
        "display_name": "Internet Search",
        "description": "Live web search for external or time-sensitive information; cached to the thread.",
        "default_prompt": "Use when information is outside the uploaded documents or likely time-sensitive. Run in parallel with document search when enabled.",
    },
    TOOL_NAME_WIKIPEDIA: {
        "id": TOOL_WIKIPEDIA_REFERENCE,
        "display_name": "Wikipedia",
        "description": "Lookup concise encyclopedia-style background on people, places, organizations, concepts, and historical topics.",
        "default_prompt": "Use for stable background, definitions, and entity overviews. Input should be a short entity/topic query, not a full multi-part question. Good for orientation before synthesis; do not use as the only source for current events, specialized papers, financial news, or claims that must come from uploaded documents.",
    },
    TOOL_NAME_WIKIDATA: {
        "id": TOOL_WIKIDATA_REFERENCE,
        "display_name": "Wikidata",
        "description": "Lookup structured entity facts from Wikidata.",
        "default_prompt": "Use for structured entity facts such as identifiers, entity type, relationships, dates, locations, creator/author, organization, occupation, and canonical metadata. Input should be an exact entity name or Wikidata QID, optionally with the fact needed. Prefer Wikipedia for narrative context; disclose if Wikidata returns sparse or ambiguous entity matches.",
    },
    TOOL_NAME_ARXIV: {
        "id": TOOL_ARXIV_RESEARCH,
        "display_name": "arXiv",
        "description": "Search arXiv for scientific and technical papers.",
        "default_prompt": "Use for preprints and papers in computer science, math, physics, quantitative biology, quantitative finance, statistics, electrical engineering, economics, and related technical fields. Input may be a concise keyword query, exact paper title, author/topic, or arXiv identifier. Do not use for biomedical-only literature when PubMed is a better fit.",
    },
    TOOL_NAME_PUB_MED: {
        "id": TOOL_PUBMED_RESEARCH,
        "display_name": "PubMed",
        "description": "Search PubMed for biomedical and life-sciences literature.",
        "default_prompt": "Use for biomedical, clinical, medicine, genetics, public-health, and life-sciences literature. Input should be a concise PubMed-style query with key concepts, conditions, interventions, genes, or outcomes; avoid very long natural-language prompts because the API wrapper truncates long queries. Summarize findings cautiously and avoid medical advice.",
    },
    TOOL_NAME_PUBMED: {
        "id": TOOL_PUBMED_RESEARCH,
        "display_name": "PubMed",
        "description": "Search PubMed for biomedical and life-sciences literature.",
        "default_prompt": "Use for biomedical, clinical, medicine, genetics, public-health, and life-sciences literature. Input should be a concise PubMed-style query with key concepts, conditions, interventions, genes, or outcomes; avoid very long natural-language prompts because the API wrapper truncates long queries. Summarize findings cautiously and avoid medical advice.",
    },
    TOOL_NAME_SEMANTIC_SCHOLAR_LEGACY: {
        "id": TOOL_SEMANTIC_SCHOLAR_RESEARCH,
        "display_name": "Semantic Scholar",
        "description": "Search Semantic Scholar for academic papers across disciplines.",
        "default_prompt": "Use for broad scholarly paper discovery across disciplines, especially when the field is not limited to arXiv or PubMed. Input should be a concise paper/topic/author query. Results commonly include title, abstract, venue, year, citations, IDs, authors, and open-access links when available; verify with arXiv/PubMed for field-specific depth.",
    },
    TOOL_NAME_SEMANTIC_SCHOLAR: {
        "id": TOOL_SEMANTIC_SCHOLAR_RESEARCH,
        "display_name": "Semantic Scholar",
        "description": "Search Semantic Scholar for academic papers across disciplines.",
        "default_prompt": "Use for broad scholarly paper discovery across disciplines, especially when the field is not limited to arXiv or PubMed. Input should be a concise paper/topic/author query. Results commonly include title, abstract, venue, year, citations, IDs, authors, and open-access links when available; verify with arXiv/PubMed for field-specific depth.",
    },
    TOOL_NAME_STACK_EXCHANGE: {
        "id": TOOL_STACKEXCHANGE_REFERENCE,
        "display_name": "StackExchange",
        "description": "Search Stack Overflow / StackExchange style technical Q&A.",
        "default_prompt": "Use for programming, debugging, command-line, library usage, library/framework behavior, and practical implementation questions. Input should be a concise technical query with the language/library/error. Treat answers as community Q&A evidence, not authoritative docs; prefer official docs or uploaded project files for final implementation decisions.",
    },
    TOOL_NAME_YAHOO_FINANCE_NEWS: {
        "id": TOOL_YAHOO_FINANCE_NEWS,
        "display_name": "Yahoo Finance News",
        "description": "Search Yahoo Finance news for a public company ticker.",
        "default_prompt": "Use for recent public-company finance/business news only after you know the listed ticker. Input must be only the ticker symbol, such as AAPL, MSFT, or NVDA; do not pass a company name, natural-language sentence, exchange name, or private company. If the user gives only a company name, first call search_web with a query like \"Nvidia stock ticker\" to find the ticker, then call yahoo_finance_news with just that ticker. If no public ticker exists, do not call this tool. Use for news context, not investment advice, valuation, real-time quotes, or private-company research.",
    },
    TOOL_NAME_ASK_FOR_CLARIFICATION: {
        "id": TOOL_CLARIFY_INTENT,
        "display_name": "Clarify Intent",
        "description": "Present 2–4 complete alternative questions for user selection.",
        "default_prompt": (
            "Use only when ambiguity would materially change the answer. Return plausible, "
            "self-contained questions that can be submitted exactly as written; never frame "
            "them as 'Did you mean', 'Are you asking', 'Do you want', or 'Do I want'."
        ),
    },
    TOOL_NAME_GET_THREAD_SHAPE: {
        "id": TOOL_THREAD_SHAPE,
        "display_name": "Thread Shape",
        "description": "Snapshot of document inventory and QA history volume.",
        "default_prompt": "Use to choose between broad doc search, scoped search, or memory search. Call once per turn.",
    },
}
