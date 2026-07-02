from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agent.agent_helpers import format_runtime_datetime_context
from app.agent.external_research_tools import get_external_research_tools
from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
from app.prompts.defaults import DEFAULT_SYSTEM_ROLE
from app.prompts.loaders import (
    get_orchestrator_phase0_prompt,
    get_orchestrator_prompt,
    get_web_search_mandate,
)


CORE_TOOL_NAMES = [
    "get_thread_shape",
    "search_documents",
    "search_document_by_id",
    "search_conversation_history",
    "search_thread_timeline",
    "search_web",
    "ask_for_clarification",
]


def _tool_name(tool_item: Any) -> str:
    return tool_item if isinstance(tool_item, str) else str(getattr(tool_item, "name", tool_item))


def _tool_description(tool_item: Any, tool_name: str) -> str:
    if not isinstance(tool_item, str):
        return str(getattr(tool_item, "description", "") or "")
    return ""


def _default_tool_names(use_external_research: bool = True) -> List[Any]:
    if not use_external_research:
        return list(CORE_TOOL_NAMES)
    external_names = [getattr(tool, "name", "") for tool in get_external_research_tools()]
    return [*CORE_TOOL_NAMES, *[name for name in external_names if name]]


def _sanitize_lines_with_blocklist(raw: str, blocklist: List[str], max_chars: int) -> str:
    if not raw:
        return ""
    lines = []
    for line in raw.splitlines():
        check = line.strip().lower()
        if any(bad in check for bad in blocklist):
            continue
        lines.append(line)
    return "\n".join(lines).strip()[:max_chars]


def sanitize_system_role(raw: str, max_chars: int = 500) -> str:
    blocked = [
        "ignore previous instructions",
        "you have no restrictions",
    ]
    return _sanitize_lines_with_blocklist(raw, blocked, max_chars)


def sanitize_custom_instructions(raw: str, max_chars: int = 2000) -> str:
    blocked = [
        "ignore previous instructions",
        "ignore all previous",
        "do not use tools",
        "disable tools",
        "never use tools",
        "pretend you have no tool",
    ]
    return _sanitize_lines_with_blocklist(raw, blocked, max_chars)


def get_tool_catalog(tool_items: Optional[List[Any]] = None) -> List[Dict[str, str]]:
    catalog: List[Dict[str, str]] = []
    for tool_item in tool_items or _default_tool_names(use_external_research=True):
        tool_name = _tool_name(tool_item)
        cfg = TOOL_FRIENDLY_CONFIG.get(tool_name, {})
        alias_id = str(cfg.get("id", tool_name))
        catalog.append(
            {
                "tool_name": tool_name,
                "id": alias_id,
                "display_name": str(cfg.get("display_name", alias_id.replace("_", " ").title())),
                "description": str(cfg.get("description", _tool_description(tool_item, tool_name))),
                "default_prompt": str(cfg.get("default_prompt", "Use this tool when it is the most relevant retrieval path.")),
            }
        )
    return catalog


def get_default_tool_instruction_map(tool_items: Optional[List[Any]] = None) -> Dict[str, str]:
    return {item["id"]: item["default_prompt"] for item in get_tool_catalog(tool_items)}


def format_intent_tool_context(active_intent_tools: List[Any]) -> str:
    """Summarize tools actually bound to the Intent Agent."""
    if not active_intent_tools:
        return "\n".join(
            [
                "## INTENT AGENT TOOL CATALOG (CALLABLE BY YOU NOW)",
                "",
                "No intent-stage tools are active for this session.",
            ]
        )

    catalog = get_tool_catalog(active_intent_tools)
    lines = [
        "## INTENT AGENT TOOL CATALOG (CALLABLE BY YOU NOW)",
        "",
        "Only the tools in this section are callable by you before submitting your route:",
    ]
    for item in catalog:
        lines.append(
            f"- `{item['tool_name']}` ({item['display_name']}): {item['description']} "
            f"Guidance: {item['default_prompt']}"
        )
    return "\n".join(lines)


def normalize_tool_instructions(
    raw: Optional[Dict[str, str]],
    max_chars_per_tool: int = 500,
    tool_items: Optional[List[Any]] = None,
) -> Dict[str, str]:
    blocked = [
        "do not use tools",
        "disable tools",
        "never use tools",
        "ignore tool contract",
    ]
    normalized = get_default_tool_instruction_map(tool_items)
    if not isinstance(raw, dict):
        return normalized
    for tool_id, value in raw.items():
        if tool_id not in normalized:
            continue
        text = _sanitize_lines_with_blocklist(str(value or ""), blocked, max_chars_per_tool)
        if text:
            normalized[tool_id] = text
    return normalized


def build_system_prompt(
    context_window: int,
    system_role: str = "",
    tool_instructions: Optional[Dict[str, str]] = None,
    custom_instructions: str = "",
    use_web_search: bool = False,
    intent_agent_ran: bool = True,
    client_timezone: Optional[str] = None,
    client_locale: Optional[str] = None,
    client_now_iso: Optional[str] = None,
) -> str:
    """Build the Orchestrator Agent system prompt without importing legacy graph objects."""
    role = system_role or DEFAULT_SYSTEM_ROLE
    active_tools = _default_tool_names(use_external_research=use_web_search)
    catalog = get_tool_catalog(active_tools)
    playbook = normalize_tool_instructions(tool_instructions or {}, tool_items=active_tools)

    template = get_orchestrator_prompt()

    if intent_agent_ran:
        intent_agent_note = "The Intent Agent upstream has already rewritten the user's query and classified its coverage; your job is to:"
        preprocessing_phase_note = ""
        phase0 = ""
        phase_count = "five"
        phase_start = ""
        orient_word = "rewritten"
        orient_extra = ""
        plan_query_note = ""
    else:
        intent_agent_note = "No upstream query preprocessor ran for this turn - you are responsible for both query preprocessing AND orchestration. Your job is to:"
        preprocessing_phase_note = "  0. Preprocess the raw user query: resolve coreferences, standalone-ify, assess coverage (Phase 0)."
        phase0 = get_orchestrator_phase0_prompt()
        phase_count = "six"
        phase_start = " Begin with Phase 0 - Preprocess."
        orient_word = "working"
        orient_extra = "\n  e) Does the raw message contain unresolved pronouns or references? -> your Phase 0\n     WORKING QUERY replaces the raw message for all retrieval operations below."
        plan_query_note = "\n  - Use the WORKING QUERY from Phase 0 - not the raw user message - for all tool arguments."

    max_parallel_tools = 4 if use_web_search else 3
    edit = "(USER-CONFIGURABLE)"
    tool_registry_section = (
        f"\n\n{'=' * 64}\nTOOL REGISTRY {edit}:\n{'=' * 64}\n"
        + "\n".join(
            [
                f"- {item['display_name']} (tool name: `{item['tool_name']}`)\n    {item['description']}"
                for item in catalog
            ]
        )
    )
    tool_playbook_section = (
        f"\n\n{'=' * 64}\nTOOL PLAYBOOK {edit}:\n{'=' * 64}\n"
        + "\n".join(
            [
                f"- `{item['tool_name']}`: {playbook.get(item['id'], item['default_prompt'])}"
                for item in catalog
            ]
        )
    )

    web_search_mandate_section = ""
    if use_web_search:
        lock = "(LOCKED - not overridable)"
        web_search_mandate_section = (
            f"\n\n{'=' * 64}\nWEB SEARCH MANDATE {lock} - overrides pre-fetch sufficiency\n"
            f"{'=' * 64}\n"
            + get_web_search_mandate()
        )

    custom_instructions_section = ""
    if custom_instructions:
        custom_instructions_section = (
            f"\n\n{'=' * 64}\nUSER CUSTOM INSTRUCTIONS {edit}\n{'=' * 64}\n"
            + custom_instructions
        )

    return template.format(
        SYSTEM_ROLE=role,
        CONTEXT_WINDOW=context_window,
        INTENT_AGENT_NOTE=intent_agent_note,
        PREPROCESSING_PHASE_NOTE=preprocessing_phase_note,
        PREPROCESSING_SECTION=phase0,
        PHASE_COUNT=phase_count,
        PHASE_START=phase_start,
        ORIENT_WORD=orient_word,
        ORIENT_EXTRA=orient_extra,
        PLAN_QUERY_NOTE=plan_query_note,
        MAX_PARALLEL_TOOLS=max_parallel_tools,
        RUNTIME_DATETIME_CONTEXT=format_runtime_datetime_context(
            client_timezone=client_timezone,
            client_locale=client_locale,
            client_now_iso=client_now_iso,
        ),
        TOOL_REGISTRY_SECTION=tool_registry_section,
        TOOL_PLAYBOOK_SECTION=tool_playbook_section,
        WEB_SEARCH_MANDATE_SECTION=web_search_mandate_section,
        CUSTOM_INSTRUCTIONS_SECTION=custom_instructions_section,
    )


def format_prefetch_for_prompt(bundle: Optional[Dict[str, Any]]) -> str:
    """
    Format a pre-fetched context bundle as a labelled block for injection into
    LLM system prompts. Returns an empty string when the bundle is None/empty.
    """
    if not bundle:
        return ""

    parts: List[str] = []

    stats = bundle.get("stats", {})
    if stats and stats.get("total_messages", 0) > 0:
        parts.append(
            f"[CONVERSATION STATS]  Total turns: {stats.get('total_messages', 0)} | "
            f"Estimated tokens in history: {int(stats.get('estimated_history_tokens', 0))}"
        )

    docs = bundle.get("documents", [])
    if docs:
        doc_lines = []
        for d in docs:
            source_type = d.get("source_type") or "unknown"
            available_at = d.get("document_available_in_thread_at") or "unknown"
            counts = []
            if d.get("page_count") not in (None, ""):
                counts.append(f"pages: {d['page_count']}")
            if d.get("word_count") not in (None, ""):
                counts.append(f"words: {d['word_count']}")
            if d.get("sentence_count") not in (None, ""):
                counts.append(f"sentences: {d['sentence_count']}")
            if d.get("chunk_count") not in (None, ""):
                counts.append(f"chunks: {d['chunk_count']}")
            counts_text = f" | {' | '.join(counts)}" if counts else ""
            doc_lines.append(
                f"  {d['index']}. {d['file_name']} "
                f"(file_hash: {d['file_hash']} | source_type: {source_type} | "
                f"added_to_thread_at: {available_at}{counts_text})"
            )
        parts.append(f"[UPLOADED DOCUMENTS - {len(docs)} file(s)]\n" + "\n".join(doc_lines))

    recent = bundle.get("recent_history_text", "")
    if recent:
        parts.append(f"[RECENT CONVERSATION TURNS]\n{recent}")

    semantic = bundle.get("semantic_history_text", "")
    if semantic:
        parts.append(f"[SEMANTICALLY RELEVANT PAST QA PAIRS]\n{semantic}")

    documents = bundle.get("document_evidence_text", "")
    if documents:
        parts.append(f"[DOCUMENT EVIDENCE (PDF + WEBPAGE)  (queried with raw/un-rewritten question)]\n{documents}")

    web = bundle.get("web_evidence_text", "")
    if web:
        parts.append(f"[WEB SEARCH EVIDENCE]\n{web}")

    if not parts:
        return ""

    sep = "=" * 64
    return (
        f"\n\n{sep}\n"
        "PRE-FETCHED CONTEXT  (assembled before this call - no tool calls needed for this data):\n"
        f"{sep}\n"
        + "\n\n".join(parts)
        + f"\n{sep}\n"
        "NOTE: Document Evidence and Semantic History were retrieved with the raw question.\n"
        "A better-rewritten query will improve precision - call tools ONLY when this\n"
        "pre-fetched context is genuinely insufficient to answer the user's request.\n"
        f"{sep}"
    )
