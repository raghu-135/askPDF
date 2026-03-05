"""
prompt_defaults.py - Default values for customizable prompt sections

This module defines all user-customizable defaults for the Orchestrator and Intent Agent prompts.
The database can override these on a per-thread basis by storing only the customized values.
"""

# Default system role for the Orchestrator Agent
DEFAULT_SYSTEM_ROLE = "Expert AI Research Assistant specializing in analyzing uploaded documents and synthesizing accurate answers."

# Default tool instructions by tool ID
# These are merged with user overrides during prompt composition
DEFAULT_TOOL_INSTRUCTIONS = {
    "document_evidence": (
        "Use when the question spans multiple documents or the target document is unknown. "
        "If results are weak, retry with a rephrased or more specific query. "
        "When the document is known, prefer search_pdf_by_document instead."
    ),
    "focused_document_evidence": (
        "Prefer this over Document Evidence when the user explicitly refers to a specific document. "
        "Resolve the file_hash from the document list in the pre-fetched context or via list_uploaded_documents."
    ),
    "deep_memory": (
        "Use for thematic or semantic recall (e.g., 'what did we discuss about X?') across the full thread history. "
        "Retry with a rephrased query if initial results are irrelevant."
    ),
    "temporal_anchor": (
        "Use for temporal references like 'your first answer about X', 'what you said earlier regarding Y', "
        "'when we started discussing Z'. Returns precise turn anchors so you can ground time-relative claims accurately. "
        "Combine with search_conversation_history to retrieve the full content of that turn."
    ),
    "live_web_recon": (
        "MANDATORY when web search is enabled: call search_web for virtually every factual question to supplement "
        "PDF content with current, external knowledge. Run it IN PARALLEL with document searches — do not wait for "
        "PDF results first. Never skip it based on pre-fetched PDF evidence alone. Always cite the URL and title "
        "of web results in your answer."
    ),
    "clarify_intent": (
        "Use only when the question has multiple plausible interpretations and making an assumption risks answering "
        "the wrong question entirely. Each option must be a complete, self-contained question."
    ),
    "thread_shape": (
        "Use to calibrate retrieval strategy: check document chunk counts to decide between search_documents vs. "
        "search_pdf_by_document, and check QA history volume to decide whether semantic memory search is worthwhile. "
        "Only call once — the snapshot is current at the time of the call."
    ),
}

# Default custom instructions (empty by default — user can add)
DEFAULT_CUSTOM_INSTRUCTIONS = ""
