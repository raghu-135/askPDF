"""
tool_registry.py - Single source of truth for user-facing tool metadata.

This powers:
  - Tool catalog in the UI
  - Tool playbook injected into the system prompt

Keep this file DRY and free of imports from agent.py to avoid cycles.
"""

TOOL_FRIENDLY_CONFIG = {
    "search_documents": {
        "id": "document_evidence",
        "display_name": "Document Evidence",
        "description": "Semantic search across uploaded documents and cached web snippets when the user needs evidence content. Use this when the target document is unknown, the question spans multiple documents, or cached web snippets may contain the answer. Do not use it just to answer first/latest/since/order questions; use search_thread_timeline when chronology is central.",
        "default_prompt": "Use for evidence content from uploaded documents or cached web snippets. Prefer search_document_by_id when a specific file_hash is known. Prefer search_thread_timeline when the user's wording depends on first/latest/earlier/since/before/after or mixed-source ordering.",
    },
    "search_document_by_id": {
        "id": "focused_document_evidence",
        "display_name": "Focused Document Evidence",
        "description": "Semantic search within one uploaded document identified by file_hash. Use this when the user names or clearly points to a specific document and thread shape provides the file_hash. Do not use it for cross-document comparison or timeline ordering unless paired with search_thread_timeline.",
        "default_prompt": "Use when a specific document is known and its file_hash is available. Keep the query focused on the requested fact. Use search_thread_timeline instead for document added-to-thread time or chronology questions.",
    },
    "search_conversation_history": {
        "id": "deep_memory",
        "display_name": "Deep Memory",
        "description": "Semantic search across past Q/A pairs in this thread when the user asks what was previously discussed or decided. Use this for topical recall where ordering is not the main question. Do not use it for first/latest/earlier/since/before/after questions; use search_thread_timeline for temporal reasoning.",
        "default_prompt": "Use for non-temporal recall of prior discussion, decisions, or answers about a topic. Avoid using it merely to reread recent turns already present in prefetch. Prefer search_thread_timeline for chronological questions.",
    },
    "search_thread_timeline": {
        "id": "thread_timeline",
        "display_name": "Thread Timeline",
        "description": "Search timestamped timeline events across conversation memory, document added-to-thread time, and cached web evidence. Use this for earliest/latest/first/earlier/since/before/after questions or when mixed-source ordering matters. It returns source-specific timestamps plus derived timeline_event_at and timeline_event_type; document timestamps mean added to this thread, not document publication time.",
        "default_prompt": "Use when the answer depends on chronology, recency, sequence, or comparing event times across conversation, documents, and cached web. Set order=oldest for first/earliest, order=newest for latest/recent, and sources to narrow the search when the user names a source class. Do not use it for ordinary semantic evidence lookup where time is irrelevant.",
    },
    "search_web": {
        "id": "live_web_recon",
        "display_name": "Internet Search",
        "description": "Live web search for external or time-sensitive information; cached to the thread.",
        "default_prompt": "Use when information is outside the uploaded documents or likely time-sensitive. Run in parallel with document search when enabled.",
    },
    "search_web_intent": {
        "id": "intent_web_lookup",
        "display_name": "Intent Web Search",
        "description": "Lightweight web search for intent disambiguation before orchestration.",
        "default_prompt": "Use only to identify unfamiliar terms or acronyms, detect time-sensitive intent, or disambiguate between plausible entities before rewriting. Do not use results as answer evidence; preserve the user's original scope and pass the clarified query to the Orchestrator.",
    },
    "wikipedia": {
        "id": "wikipedia_reference",
        "display_name": "Wikipedia",
        "description": "Lookup concise encyclopedia-style background on people, places, organizations, concepts, and historical topics.",
        "default_prompt": "Use for stable background, definitions, and entity overviews. Input should be a short entity/topic query, not a full multi-part question. Good for orientation before synthesis; do not use as the only source for current events, specialized papers, financial news, or claims that must come from uploaded documents.",
    },
    "wikidata": {
        "id": "wikidata_reference",
        "display_name": "Wikidata",
        "description": "Lookup structured entity facts from Wikidata.",
        "default_prompt": "Use for structured entity facts such as identifiers, entity type, relationships, dates, locations, creator/author, organization, occupation, and canonical metadata. Input should be an exact entity name or Wikidata QID, optionally with the fact needed. Prefer Wikipedia for narrative context; disclose if Wikidata returns sparse or ambiguous entity matches.",
    },
    "arxiv": {
        "id": "arxiv_research",
        "display_name": "arXiv",
        "description": "Search arXiv for scientific and technical papers.",
        "default_prompt": "Use for preprints and papers in computer science, math, physics, quantitative biology, quantitative finance, statistics, electrical engineering, economics, and related technical fields. Input may be a concise keyword query, exact paper title, author/topic, or arXiv identifier. Do not use for biomedical-only literature when PubMed is a better fit.",
    },
    "pub_med": {
        "id": "pubmed_research",
        "display_name": "PubMed",
        "description": "Search PubMed for biomedical and life-sciences literature.",
        "default_prompt": "Use for biomedical, clinical, medicine, genetics, public-health, and life-sciences literature. Input should be a concise PubMed-style query with key concepts, conditions, interventions, genes, or outcomes; avoid very long natural-language prompts because the API wrapper truncates long queries. Summarize findings cautiously and avoid medical advice.",
    },
    "pubmed": {
        "id": "pubmed_research",
        "display_name": "PubMed",
        "description": "Search PubMed for biomedical and life-sciences literature.",
        "default_prompt": "Use for biomedical, clinical, medicine, genetics, public-health, and life-sciences literature. Input should be a concise PubMed-style query with key concepts, conditions, interventions, genes, or outcomes; avoid very long natural-language prompts because the API wrapper truncates long queries. Summarize findings cautiously and avoid medical advice.",
    },
    "semanticscholar": {
        "id": "semantic_scholar_research",
        "display_name": "Semantic Scholar",
        "description": "Search Semantic Scholar for academic papers across disciplines.",
        "default_prompt": "Use for broad scholarly paper discovery across disciplines, especially when the field is not limited to arXiv or PubMed. Input should be a concise paper/topic/author query. Results commonly include title, abstract, venue, year, citations, IDs, authors, and open-access links when available; verify with arXiv/PubMed for field-specific depth.",
    },
    "semantic_scholar": {
        "id": "semantic_scholar_research",
        "display_name": "Semantic Scholar",
        "description": "Search Semantic Scholar for academic papers across disciplines.",
        "default_prompt": "Use for broad scholarly paper discovery across disciplines, especially when the field is not limited to arXiv or PubMed. Input should be a concise paper/topic/author query. Results commonly include title, abstract, venue, year, citations, IDs, authors, and open-access links when available; verify with arXiv/PubMed for field-specific depth.",
    },
    "stack_exchange": {
        "id": "stackexchange_reference",
        "display_name": "StackExchange",
        "description": "Search Stack Overflow / StackExchange style technical Q&A.",
        "default_prompt": "Use for programming, debugging, command-line, library usage, library/framework behavior, and practical implementation questions. Input should be a concise technical query with the language/library/error. Treat answers as community Q&A evidence, not authoritative docs; prefer official docs or uploaded project files for final implementation decisions.",
    },
    "yahoo_finance_news": {
        "id": "yahoo_finance_news",
        "display_name": "Yahoo Finance News",
        "description": "Search Yahoo Finance news for a public company ticker.",
        "default_prompt": "Use for recent public-company finance/business news only after you know the listed ticker. Input must be only the ticker symbol, such as AAPL, MSFT, or NVDA; do not pass a company name, natural-language sentence, exchange name, or private company. If the user gives only a company name, first call search_web with a query like \"Nvidia stock ticker\" to find the ticker, then call yahoo_finance_news with just that ticker. If no public ticker exists, do not call this tool. Use for news context, not investment advice, valuation, real-time quotes, or private-company research.",
    },
    "ask_for_clarification": {
        "id": "clarify_intent",
        "display_name": "Clarify Intent",
        "description": "Present 2–4 alternative interpretations for user selection.",
        "default_prompt": "Use only when ambiguity would materially change the answer.",
    },
    "get_thread_shape": {
        "id": "thread_shape",
        "display_name": "Thread Shape",
        "description": "Snapshot of document inventory and QA history volume.",
        "default_prompt": "Use to choose between broad doc search, scoped search, or memory search. Call once per turn.",
    },
}
