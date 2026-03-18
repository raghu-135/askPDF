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
        "description": "Semantic search across all uploaded documents and cached web results.",
        "default_prompt": "Use when the target document is unknown or the question spans multiple documents. If a specific document is known, prefer search_document_by_id.",
    },
    "search_document_by_id": {
        "id": "focused_document_evidence",
        "display_name": "Focused Document Evidence",
        "description": "Semantic search within a single document by file_hash.",
        "default_prompt": "Use when the user references a specific document and its file_hash is known.",
    },
    "search_conversation_history": {
        "id": "deep_memory",
        "display_name": "Deep Memory",
        "description": "Semantic search across past Q/A pairs in this thread.",
        "default_prompt": "Use for recalling prior discussion, decisions, or answers about a topic.",
    },
    "find_topic_anchor_in_history": {
        "id": "temporal_anchor",
        "display_name": "Temporal Anchor",
        "description": "Finds the earliest mention of a topic in conversation history.",
        "default_prompt": "Use for temporal references like 'first time we discussed X'; follow up with search_conversation_history if you need the full content.",
    },
    "search_web": {
        "id": "live_web_recon",
        "display_name": "Internet Search",
        "description": "Live web search for external or time-sensitive information; cached to the thread.",
        "default_prompt": "Use when information is outside the uploaded documents or likely time-sensitive. Run in parallel with document search when enabled.",
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
