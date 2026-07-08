from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.agent.prompting import (
    format_runtime_datetime_context,
    get_tool_catalog,
    normalize_tool_instructions,
    sanitize_custom_instructions,
    sanitize_system_role,
)
from app.prompts.loaders import get_web_search_mandate, load_prompt


GRAPH_TOOL_NAMES = [
    "search_documents",
    "search_document_by_id",
    "search_conversation_history",
    "search_thread_timeline",
    "search_web",
    "ask_for_clarification",
]

QUESTION_PLACEHOLDER = "{{QUESTION}}"
PREFETCH_PLACEHOLDER = "{{PREFETCH_CONTEXT}}"
CONTEXT_PLACEHOLDER = "{{EVIDENCE_CONTEXT}}"
EVALUATOR_REPORT_PLACEHOLDER = "{{EVALUATOR_REPORT}}"


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_prompt(filename: str, values: Dict[str, Any]) -> str:
    return load_prompt(filename).format_map(_SafeFormatDict({k: str(v) for k, v in values.items()}))


def _format_prefetch_summary(bundle: Optional[Dict[str, Any]]) -> str:
    if not bundle:
        return ""
    parts: List[str] = []
    if bundle.get("documents"):
        docs = bundle["documents"]
        parts.append(
            "Documents available: "
            + ", ".join(
                str(d.get("file_name") or d.get("title") or d.get("file_hash") or "document")
                for d in docs[:5]
            )
        )
    if bundle.get("document_evidence_text"):
        parts.append("Document evidence:\n" + str(bundle["document_evidence_text"])[:4000])
    if bundle.get("semantic_history_text"):
        parts.append("Relevant prior conversation:\n" + str(bundle["semantic_history_text"])[:2500])
    if bundle.get("recent_history_text"):
        parts.append("Recent conversation:\n" + str(bundle["recent_history_text"])[:2500])
    if bundle.get("web_evidence_text"):
        parts.append("Cached web evidence:\n" + str(bundle["web_evidence_text"])[:3000])
    return "\n\n".join(parts) if parts else "No pre-fetched context available."


def _prompt_context(state_or_settings: Dict[str, Any]) -> Dict[str, Any]:
    use_web_search = bool(state_or_settings.get("use_web_search", False))
    active_tools = list(GRAPH_TOOL_NAMES if use_web_search else [name for name in GRAPH_TOOL_NAMES if name != "search_web"])
    catalog = get_tool_catalog(active_tools)
    playbook = normalize_tool_instructions(
        state_or_settings.get("tool_instructions") or {},
        tool_items=active_tools,
    )
    registry_lines = [
        "## Tool Registry",
        "",
        "These are the retrieval and clarification tools represented by this graph runtime:",
    ]
    registry_lines.extend(
        [
            f"- {item['display_name']} (tool name: `{item['tool_name']}`): {item['description']}"
            for item in catalog
        ]
    )
    playbook_lines = [
        "## Tool Playbook",
        "",
    ]
    playbook_lines.extend(
        [
            f"- `{item['tool_name']}`: {playbook.get(item['id'], item['default_prompt'])}"
            for item in catalog
        ]
    )
    web_search_mandate = ""
    if use_web_search:
        web_search_mandate = "## Web Search Mandate\n\n" + get_web_search_mandate()

    return {
        "RUNTIME_DATETIME_CONTEXT": format_runtime_datetime_context(
            client_timezone=state_or_settings.get("client_timezone"),
            client_locale=state_or_settings.get("client_locale"),
            client_now_iso=state_or_settings.get("client_now_iso"),
        ),
        "TOOL_REGISTRY_SECTION": "\n".join(registry_lines),
        "TOOL_PLAYBOOK_SECTION": "\n".join(playbook_lines),
        "WEB_SEARCH_MANDATE_SECTION": web_search_mandate,
        "USE_WEB_SEARCH": str(use_web_search),
        "CONTEXT_WINDOW": state_or_settings.get("context_window", ""),
        "QUESTION": state_or_settings.get("question") or QUESTION_PLACEHOLDER,
        "PREFETCH_CONTEXT": _format_prefetch_summary(state_or_settings.get("pre_fetch_bundle") or {})
        or PREFETCH_PLACEHOLDER,
    }


def build_router_prompt(state: Dict[str, Any]) -> str:
    return _render_prompt("agent_patterns/router_rag_router.md", _prompt_context(state))


def build_planner_prompt(state: Dict[str, Any]) -> str:
    return _render_prompt("agent_patterns/plan_execute_planner.md", _prompt_context(state))


def _json_preview(value: Any, *, limit: int = 4000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        text = str(value or "")
    return text[:limit]


def _evaluator_prompt_context(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **_prompt_context(state),
        "EVIDENCE_CONTEXT": str(state.get("evidence") or "")[:8000] or CONTEXT_PLACEHOLDER,
        "EXECUTION_PLAN": _json_preview(state.get("execution_plan") or []),
        "DOCUMENT_SOURCE_COUNT": len(state.get("document_sources") or []),
        "WEB_SOURCE_COUNT": len(state.get("web_sources") or []),
        "USED_CHAT_ID_COUNT": len(state.get("used_chat_ids") or []),
        "REPLAN_COUNT": state.get("replan_count", 0),
        "REPLANS": state.get("replans", 1),
    }


def build_evaluator_prompt(state: Dict[str, Any]) -> str:
    return _render_prompt("agent_patterns/evaluator_replanner_evaluator.md", _evaluator_prompt_context(state))


def build_replanner_prompt(state: Dict[str, Any]) -> str:
    values = {
        **_evaluator_prompt_context(state),
        "EVALUATOR_REPORT": _json_preview(state.get("evaluator_report") or {}, limit=5000)
        or EVALUATOR_REPORT_PLACEHOLDER,
    }
    return _render_prompt("agent_patterns/evaluator_replanner_replanner.md", values)


def build_final_answer_messages(state: Dict[str, Any], context: str) -> Dict[str, str]:
    system_role = sanitize_system_role(state.get("system_role", ""))
    custom_instructions = sanitize_custom_instructions(state.get("custom_instructions", ""))
    values = {
        **_prompt_context(state),
        "CONTEXT": context or CONTEXT_PLACEHOLDER,
        "SYSTEM_ROLE_SECTION": f"Assistant role:\n{system_role}" if system_role else "",
        "CUSTOM_INSTRUCTIONS_SECTION": f"Custom instructions:\n{custom_instructions}" if custom_instructions else "",
    }
    rendered = _render_prompt("agent_patterns/final_answer.md", values)
    system_marker = "## System Message"
    human_marker = "## Human Message"
    if system_marker not in rendered or human_marker not in rendered:
        return {
            "system": "You answer askPDF questions using the supplied context.",
            "human": f"Question:\n{values['QUESTION']}\n\nContext:\n{values['CONTEXT']}\n\nWrite the final answer.",
        }
    system_start = rendered.index(system_marker) + len(system_marker)
    human_start = rendered.index(human_marker)
    human_content_start = human_start + len(human_marker)
    return {
        "system": rendered[system_start:human_start].strip(),
        "human": rendered[human_content_start:].strip(),
    }


def build_agent_pattern_prompt_preview(
    *,
    pattern_id: Optional[str] = None,
    prompt_profile: Optional[str] = None,
    context_window: int,
    system_role: str = "",
    tool_instructions: Optional[Dict[str, str]] = None,
    custom_instructions: str = "",
    use_web_search: bool = False,
    client_timezone: Optional[str] = None,
    client_locale: Optional[str] = None,
    client_now_iso: Optional[str] = None,
) -> str:
    state = {
        "question": QUESTION_PLACEHOLDER,
        "pre_fetch_bundle": {},
        "context_window": context_window,
        "system_role": system_role,
        "tool_instructions": tool_instructions or {},
        "custom_instructions": custom_instructions,
        "use_web_search": use_web_search,
        "client_timezone": client_timezone,
        "client_locale": client_locale,
        "client_now_iso": client_now_iso,
    }
    final_messages = build_final_answer_messages(state, CONTEXT_PLACEHOLDER)
    sections: List[str] = []
    if prompt_profile == "evaluator_replanner":
        sections.append(
            "# Planner Node Prompt\n\n"
            "This is the system + human prompt for the planner LLM call. It decides route and initial worker inclusion only.\n\n"
            "## System Message\n\nYou are a strict planner for a scoped RAG workflow.\n\n"
            "## Human Message\n\n"
            + build_planner_prompt(state)
        )
        sections.append(
            "# Evidence Evaluator Prompt\n\n"
            "This is the system + human prompt for the evidence evaluator LLM call. It decides whether evidence is sufficient or one bounded replan is needed.\n\n"
            "## System Message\n\nYou are a strict evidence evaluator for a bounded RAG workflow.\n\n"
            "## Human Message\n\n"
            + build_evaluator_prompt(state)
        )
        sections.append(
            "# Replanner Prompt\n\n"
            "This is the system + human prompt for the replanner LLM call. It revises worker inclusion under the remaining replan budget.\n\n"
            "## System Message\n\nYou are a strict replanner for a bounded RAG workflow.\n\n"
            "## Human Message\n\n"
            + build_replanner_prompt(state)
        )
    elif prompt_profile == "planner":
        sections.append(
            "# Planner Node Prompt\n\n"
            "This is the system + human prompt for the planner LLM call. It decides route and worker inclusion only.\n\n"
            "## System Message\n\nYou are a strict planner for a scoped RAG workflow.\n\n"
            "## Human Message\n\n"
            + build_planner_prompt(state)
        )
    else:
        sections.append(
            "# Router Node Prompt\n\n"
            "This is the system + human prompt for the router LLM call. It chooses the next graph route only.\n\n"
            "## System Message\n\nYou are a strict router for a RAG workflow.\n\n"
            "## Human Message\n\n"
            + build_router_prompt(state)
        )
    sections.append(
        "# Final Answer Prompt\n\n"
        "This is the system + human prompt for the answer-writing LLM call. The UI System role and Custom instructions apply here.\n\n"
        "## System Message\n\n"
        + final_messages["system"]
        + "\n\n## Human Message\n\n"
        + final_messages["human"]
    )
    return "\n\n---\n\n".join(sections)
