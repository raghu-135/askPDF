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
from app.agent_workflows.enums import PromptProfile, ToolName
from app.agent_workflows.planning import WORKER_NODE_ORDER
from app.prompts.loaders import get_web_search_mandate, load_prompt


GRAPH_TOOL_NAMES = [
    ToolName.SEARCH_DOCUMENTS.value,
    ToolName.SEARCH_DOCUMENT_BY_ID.value,
    ToolName.SEARCH_CONVERSATION_HISTORY.value,
    ToolName.SEARCH_THREAD_TIMELINE.value,
    ToolName.SEARCH_WEB.value,
    ToolName.ASK_FOR_CLARIFICATION.value,
]

QUESTION_PLACEHOLDER = "{{QUESTION}}"
PREFETCH_PLACEHOLDER = "{{PREFETCH_CONTEXT}}"
CONTEXT_PLACEHOLDER = "{{EVIDENCE_CONTEXT}}"
EVALUATOR_REPORT_PLACEHOLDER = "{{EVALUATOR_REPORT}}"


WORKER_TYPE_DESCRIPTIONS = {
    "retrieval_worker": "uploaded document, PDF, page, section, quote, citation, excerpt, summary, or cached web snippet evidence",
    "memory_worker": "non-temporal recall of prior conversation, previous answers, or what we discussed",
    "timeline_worker": "chronology, latest/most recent/current, first/earliest/oldest, before/after/since, date/time, or event ordering",
    "web_worker": "live internet evidence, only when live web search is enabled",
}


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


def _format_available_worker_nodes(state_or_settings: Dict[str, Any]) -> str:
    workers = state_or_settings.get("available_worker_nodes")
    if workers is None:
        workers = [{"id": node_type, "type": node_type, "label": node_type} for node_type in WORKER_NODE_ORDER]
    if not isinstance(workers, list) or not workers:
        return "No worker nodes are available for this workflow."
    lines = [
        "Use these exact worker node ids in `execution_plan`; do not output worker type aliases unless the id is exactly the same.",
        "",
    ]
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        node_id = worker.get("id")
        node_type = worker.get("type")
        if not isinstance(node_id, str) or not isinstance(node_type, str):
            continue
        label = worker.get("label")
        label_suffix = f" ({label})" if isinstance(label, str) and label and label != node_id else ""
        description = WORKER_TYPE_DESCRIPTIONS.get(node_type, node_type)
        lines.append(f"- `{node_id}`{label_suffix}: {node_type}; {description}.")
    return "\n".join(lines)


def _prompt_context(state_or_settings: Dict[str, Any]) -> Dict[str, Any]:
    use_web_search = bool(state_or_settings.get("use_web_search", False))
    active_tools = list(GRAPH_TOOL_NAMES if use_web_search else [name for name in GRAPH_TOOL_NAMES if name != ToolName.SEARCH_WEB.value])
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
        "AVAILABLE_WORKER_NODES": _format_available_worker_nodes(state_or_settings),
    }


def build_router_prompt(state: Dict[str, Any]) -> str:
    return _render_prompt("agent_workflows/router_rag_router.md", _prompt_context(state))


def build_planner_prompt(state: Dict[str, Any]) -> str:
    return _render_prompt("agent_workflows/plan_execute_planner.md", _prompt_context(state))


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
    return _render_prompt("agent_workflows/evaluator_replanner_evaluator.md", _evaluator_prompt_context(state))


def build_replanner_prompt(state: Dict[str, Any]) -> str:
    values = {
        **_evaluator_prompt_context(state),
        "EVALUATOR_REPORT": _json_preview(state.get("evaluator_report") or {}, limit=5000)
        or EVALUATOR_REPORT_PLACEHOLDER,
    }
    return _render_prompt("agent_workflows/evaluator_replanner_replanner.md", values)


def build_final_answer_messages(state: Dict[str, Any], context: str) -> Dict[str, str]:
    system_role = sanitize_system_role(state.get("system_role", ""))
    custom_instructions = sanitize_custom_instructions(state.get("custom_instructions", ""))
    values = {
        **_prompt_context(state),
        "CONTEXT": context or CONTEXT_PLACEHOLDER,
        "SYSTEM_ROLE_SECTION": f"Assistant role:\n{system_role}" if system_role else "",
        "CUSTOM_INSTRUCTIONS_SECTION": f"Custom instructions:\n{custom_instructions}" if custom_instructions else "",
    }
    rendered = _render_prompt("agent_workflows/final_answer.md", values)
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


def build_agent_workflow_prompt_preview(
    *,
    workflow_id: Optional[str] = None,
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
    if prompt_profile is None:
        prompt_profile = {
            "plan_execute_rag_agent": PromptProfile.PLANNER.value,
            "evaluator_replanner_rag_agent": PromptProfile.EVALUATOR_REPLANNER.value,
        }.get(str(workflow_id or ""), PromptProfile.ROUTER.value)
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
    if prompt_profile == PromptProfile.EVALUATOR_REPLANNER.value:
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
    elif prompt_profile == PromptProfile.PLANNER.value:
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
