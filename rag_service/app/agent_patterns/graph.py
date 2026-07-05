from __future__ import annotations

import json
import logging
import re
import time
from datetime import timedelta
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agent.reasoning import normalize_ai_response
from app.agent.tool_contract import compact_tool_event, normalize_tool_result
from app.agent.tool_registry import get_tool_contract_id, validate_tool_call_allowed
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET, get_llm
from app.models.retry import invoke_with_retry
from app.agent.external_research_tools import search_web
from app.rag.agent_tools import search_conversation_history, search_documents, search_thread_timeline
from app.rag.chat_service import prefetch_context
from app.agent_patterns.prompting import build_final_answer_messages, build_planner_prompt, build_router_prompt
from app.agent_patterns.templates import PLAN_EXECUTE_WORKER_NODES
from app.agent_patterns.trace import (
    available_document_refs,
    compact_preview,
    compact_refs,
    normalize_warnings,
    prompt_summary,
    refs_from_artifacts,
    refs_from_documents,
    refs_from_messages,
    refs_from_timeline,
    refs_from_web,
    selected_and_skipped_workers,
)
from app.time_utils import iso_utc_z, utc_now


RouterRoute = Literal["document", "memory", "timeline", "web", "direct", "clarify"]

logger = logging.getLogger(__name__)


TEMPORAL_PLAN_RE = re.compile(
    r"\b("
    r"latest|most\s+recent|recent|newest|current|first|earliest|oldest|last|"
    r"since|before|after|earlier|later|timeline|chronolog(?:y|ical)|sequence|order|"
    r"when|date|time"
    r")\b",
    re.IGNORECASE,
)

MEMORY_PLAN_RE = re.compile(
    r"\b("
    r"previously|prior|earlier\s+(answer|conversation|chat|discussion)|"
    r"remember|discussed|talked\s+about|said\s+before|you\s+said|we\s+(said|discussed|talked)"
    r")\b",
    re.IGNORECASE,
)

DOCUMENT_PLAN_RE = re.compile(
    r"\b("
    r"document|pdf|paper|uploaded|upload|file|source|page|section|chapter|"
    r"quote|cite|citation|excerpt|summar(?:y|ize)|abstract"
    r")\b",
    re.IGNORECASE,
)


class RouterRagState(TypedDict, total=False):
    agent_run_id: Optional[str]
    thread_id: str
    question: str
    llm_model: str
    embedding_model: str
    context_window: int
    use_web_search: bool
    use_reranker: bool
    system_role: str
    tool_instructions: Dict[str, str]
    custom_instructions: str
    client_timezone: Optional[str]
    client_locale: Optional[str]
    client_now_iso: Optional[str]
    pre_fetch_bundle: Dict[str, Any]
    route: RouterRoute
    route_reason: str
    clarification_options: Optional[List[str]]
    evidence: str
    document_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    used_chat_ids: List[str]
    final_answer: str
    reasoning: str
    reasoning_available: bool
    reasoning_format: str
    node_events: List[Dict[str, Any]]
    tool_events: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    allowed_tool_ids: List[str]
    pattern_type: str
    execution_plan: List[str]


def _append_event(
    state: RouterRagState,
    node: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    started: Optional[float] = None,
    config: Optional[RunnableConfig] = None,
) -> List[Dict[str, Any]]:
    event = {"node": node, **(data or {})}
    if started is not None:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        completed_at = utc_now()
        event["elapsed_ms"] = elapsed_ms
        event.setdefault("start_time", iso_utc_z(completed_at - timedelta(milliseconds=elapsed_ms)))
        event.setdefault("end_time", iso_utc_z(completed_at))
    telemetry_sink = ((config or {}).get("configurable") or {}).get("telemetry_sink")
    if isinstance(telemetry_sink, dict):
        telemetry_sink.setdefault("node_events", []).append(dict(event))
    return [*state.get("node_events", []), event]


def _append_tool_event(
    state: RouterRagState,
    payload: Dict[str, Any],
    *,
    tool_input: Any = None,
    config: Optional[RunnableConfig] = None,
) -> List[Dict[str, Any]]:
    event = compact_tool_event(payload, tool_input=tool_input)
    telemetry_sink = ((config or {}).get("configurable") or {}).get("telemetry_sink")
    if isinstance(telemetry_sink, dict):
        telemetry_sink.setdefault("tool_events", []).append(dict(event))
    return [*state.get("tool_events", []), event]


def _tool_config(state: RouterRagState, config: RunnableConfig, *, caller_node: str, tool_name: str) -> RunnableConfig:
    validate_tool_call_allowed(tool_name, caller_node)
    contract_id = get_tool_contract_id(tool_name)
    allowed_tool_ids = state.get("allowed_tool_ids")
    if not isinstance(allowed_tool_ids, list) or contract_id not in allowed_tool_ids:
        raise ValueError(
            f"Tool {tool_name} with contract ID {contract_id} is not enabled for this agent run"
        )
    updated = dict(config or {})
    configurable = dict(updated.get("configurable") or {})
    configurable.update(
        {
            "agent_run_id": state.get("agent_run_id"),
            "caller_node": caller_node,
            "route": state.get("route"),
            "tool_name": tool_name,
        }
    )
    updated["configurable"] = configurable
    metadata = dict(updated.get("metadata") or {})
    metadata.update(
        {
            "agent_run_id": state.get("agent_run_id"),
            "caller_node": caller_node,
            "route": state.get("route"),
            "tool_name": tool_name,
        }
    )
    updated["metadata"] = metadata
    return updated


def _log_node_end(
    state: RouterRagState,
    node: str,
    started: float,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    payload = data or {}
    logger.info(
        "Router RAG node completed | run_id=%s thread_id=%s node=%s elapsed_ms=%.1f route=%s evidence_chars=%s document_sources=%s web_sources=%s used_chat_ids=%s",
        state.get("agent_run_id"),
        state.get("thread_id"),
        node,
        (time.perf_counter() - started) * 1000,
        payload.get("route", state.get("route")),
        payload.get("evidence_chars", len(str(state.get("evidence") or ""))),
        payload.get("document_source_count", len(state.get("document_sources") or [])),
        payload.get("web_source_count", len(state.get("web_sources") or [])),
        payload.get("used_chat_id_count", len(state.get("used_chat_ids") or [])),
    )


def _safe_json_object(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _llm_result_metadata(
    response: Any,
    *,
    model_name: Optional[str] = None,
    normalized_response: Optional[Dict[str, Any]] = None,
    retry_attempts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    usage = usage if isinstance(usage, dict) else {}
    response_metadata = getattr(response, "response_metadata", None)
    response_metadata = response_metadata if isinstance(response_metadata, dict) else {}
    token_usage = response_metadata.get("token_usage") if isinstance(response_metadata.get("token_usage"), dict) else {}
    output_tokens_details = token_usage.get("completion_tokens_details") if isinstance(token_usage.get("completion_tokens_details"), dict) else {}
    input_tokens_details = token_usage.get("prompt_tokens_details") if isinstance(token_usage.get("prompt_tokens_details"), dict) else {}
    content = getattr(response, "content", "")
    content_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=True) if content else ""
    normalized = normalized_response if isinstance(normalized_response, dict) else {}
    reasoning = str(normalized.get("reasoning") or "")

    token_counts = {
        "prompt": _first_int(usage.get("input_tokens"), token_usage.get("prompt_tokens")),
        "completion": _first_int(usage.get("output_tokens"), token_usage.get("completion_tokens")),
        "total": _first_int(usage.get("total_tokens"), token_usage.get("total_tokens")),
        "reasoning": _first_int(
            usage.get("reasoning_tokens"),
            token_usage.get("reasoning_tokens"),
            output_tokens_details.get("reasoning_tokens"),
        ),
        "cached": _first_int(
            usage.get("cached_tokens"),
            token_usage.get("cached_tokens"),
            input_tokens_details.get("cached_tokens"),
        ),
    }
    token_counts = {key: value for key, value in token_counts.items() if value is not None}

    summary = {
        "model_name": model_name or response_metadata.get("model_name") or response_metadata.get("model"),
        "response_chars": len(content_text),
        "token_counts": token_counts,
        "retry_count": len(retry_attempts or []),
        "retry_attempts": retry_attempts or [],
        "reasoning_available": normalized.get("reasoning_available"),
        "reasoning_format": normalized.get("reasoning_format"),
        "reasoning_chars": len(reasoning) if reasoning else None,
        "reasoning_preview": compact_preview(reasoning, limit=1800) if reasoning else None,
    }
    return {key: value for key, value in summary.items() if value not in (None, "", {}, [])}


def _llm_retry_observer() -> tuple[List[Dict[str, Any]], Callable[[Dict[str, Any]], None]]:
    attempts: List[Dict[str, Any]] = []

    def observe(event: Dict[str, Any]) -> None:
        attempts.append(dict(event))

    return attempts, observe


def _format_prefetch_summary(bundle: Dict[str, Any]) -> str:
    parts = []
    if bundle.get("recent_history_text"):
        parts.append("Recent conversation:\n" + bundle["recent_history_text"])
    if bundle.get("semantic_history_text"):
        parts.append("Semantic memory:\n" + bundle["semantic_history_text"])
    if bundle.get("document_evidence_text"):
        parts.append("Document evidence:\n" + bundle["document_evidence_text"])
    documents = bundle.get("documents") or []
    if documents:
        names = [f"- {doc.get('file_name')} ({doc.get('file_hash')})" for doc in documents[:12]]
        parts.append("Available documents:\n" + "\n".join(names))
    return "\n\n".join(parts).strip() or "No pre-fetched context is available."


def _combine_evidence(existing: Any, addition: Any, *, label: str) -> str:
    existing_text = str(existing or "").strip()
    addition_text = str(addition or "").strip()
    if not addition_text:
        return existing_text
    labeled = f"[{label}]\n{addition_text}"
    return "\n\n".join(part for part in (existing_text, labeled) if part).strip()


def _should_skip_worker(state: RouterRagState, worker_node: str) -> bool:
    plan = state.get("execution_plan")
    if not isinstance(plan, list):
        return False
    return worker_node not in plan


def _skipped_worker_update(
    state: RouterRagState,
    config: RunnableConfig,
    worker_node: str,
    started: float,
    reason: str,
) -> Dict[str, Any]:
    data = {
        "status": "skipped",
        "skipped": True,
        "skip_reason": reason,
        "input_preview": {"question": compact_preview(state.get("question"))},
        "input_refs": _state_evidence_refs(state),
        "output_refs": _state_evidence_refs(state),
    }
    _log_node_end(state, worker_node, started, data)
    return {"node_events": _append_event(state, worker_node, data, started=started, config=config)}


def infer_required_plan_steps(question: Optional[str]) -> List[str]:
    """Return worker nodes that should be present for obvious query intent cues."""

    text = str(question or "")
    required: List[str] = []
    if TEMPORAL_PLAN_RE.search(text):
        required.append("timeline_worker")
    if MEMORY_PLAN_RE.search(text) and "memory_worker" not in required and "timeline_worker" not in required:
        required.append("memory_worker")
    if DOCUMENT_PLAN_RE.search(text) and "retrieval_worker" not in required:
        required.append("retrieval_worker")
    return required


def _ordered_plan_steps(steps: List[str]) -> List[str]:
    return [node for node in PLAN_EXECUTE_WORKER_NODES if node in steps]


def _fallback_clarification_options() -> List[str]:
    return [
        "Do I want an answer based on the uploaded document evidence?",
        "Do I want an answer based on what we discussed earlier in this thread?",
        "Do I want an answer based on the timeline or order of events in this thread?",
    ]


def _prefetch_refs(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return compact_refs(
        {
            "recent_messages": refs_from_messages(bundle.get("recent_message_refs")),
            "semantic_matches": refs_from_messages(bundle.get("semantic_memory_refs") or bundle.get("used_chat_ids")),
            "document_matches": refs_from_documents(bundle.get("document_sources")),
            "web_sources": refs_from_web(bundle.get("web_sources")),
            "available_documents": available_document_refs(bundle.get("documents")),
        }
    )


def _state_evidence_refs(state: RouterRagState) -> Dict[str, Any]:
    return compact_refs(
        {
            "document_matches": refs_from_documents(state.get("document_sources")),
            "web_sources": refs_from_web(state.get("web_sources")),
            "messages": refs_from_messages(state.get("used_chat_ids")),
        }
    )


def normalize_execution_plan(
    parsed: Dict[str, Any],
    *,
    use_web_search: bool,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    allowed_routes = {"execute", "direct", "clarify"}
    route = parsed.get("route") if parsed.get("route") in allowed_routes else "execute"
    required_steps = infer_required_plan_steps(question)
    normalization_notes: List[str] = []
    if route == "direct" and required_steps:
        route = "execute"
        normalization_notes.append("direct_route_clamped_to_execute")
    raw_steps = parsed.get("execution_plan") or parsed.get("steps") or []
    steps: List[str] = []
    if isinstance(raw_steps, list):
        for step in raw_steps:
            if isinstance(step, str):
                node = step
            elif isinstance(step, dict):
                node = step.get("node") or step.get("worker") or step.get("id")
            else:
                continue
            if node in PLAN_EXECUTE_WORKER_NODES and node not in steps:
                steps.append(node)
    if not use_web_search and "web_worker" in steps:
        steps = [step for step in steps if step != "web_worker"]
        normalization_notes.append("web_worker_removed_when_web_search_disabled")
    if route == "execute":
        for required_step in required_steps:
            if required_step not in steps:
                steps.append(required_step)
    if route == "execute" and not steps:
        steps = ["retrieval_worker"]
        normalization_notes.append("empty_execute_plan_defaulted_to_retrieval_worker")
    steps = _ordered_plan_steps(steps)
    if route != "execute":
        steps = []
    clarification_options = parsed.get("clarification_options")
    if route == "clarify" and not isinstance(clarification_options, list):
        clarification_options = _fallback_clarification_options()
    return {
        "route": route,
        "route_reason": str(parsed.get("reason") or parsed.get("route_reason") or ""),
        "execution_plan": steps,
        "clarification_options": clarification_options if route == "clarify" else None,
        "normalization_notes": normalization_notes,
    }


class NodeRegistry:
    """Registry of safe backend node implementations for compiled v2 patterns."""

    def __init__(self):
        self._nodes: Dict[str, Callable[..., Any]] = {
            "context_loader": self.context_loader,
            "planner": self.planner,
            "router": self.router,
            "retrieval_worker": self.retrieval_worker,
            "memory_worker": self.memory_worker,
            "timeline_worker": self.timeline_worker,
            "web_worker": self.web_worker,
            "direct_answer": self.direct_answer,
            "synthesizer": self.synthesizer,
            "finalizer": self.finalizer,
        }

    def get(self, node_type: str) -> Callable[..., Any]:
        if node_type not in self._nodes:
            raise ValueError(f"Unknown node type: {node_type}")
        return self._nodes[node_type]

    async def context_loader(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        bundle = await prefetch_context(
            thread_id=state["thread_id"],
            raw_question=state["question"],
            embed_model_name=state["embedding_model"],
            context_window=state.get("context_window", DEFAULT_TOKEN_BUDGET),
            use_web_search=state.get("use_web_search", False),
            use_reranker=state.get("use_reranker", True),
        )
        data = {
            "status": "completed",
            "document_source_count": len(bundle.get("document_sources", [])),
            "web_source_count": len(bundle.get("web_sources", [])),
            "used_chat_id_count": len(bundle.get("used_chat_ids", [])),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "settings": {
                    "context_window": state.get("context_window"),
                    "use_web_search": state.get("use_web_search"),
                    "use_reranker": state.get("use_reranker"),
                },
            },
            "output_refs": _prefetch_refs(bundle),
            "output_preview": {
                "recent_history": compact_preview(bundle.get("recent_history_text")),
                "semantic_history": compact_preview(bundle.get("semantic_history_text")),
                "document_evidence": compact_preview(bundle.get("document_evidence_text")),
            },
        }
        _log_node_end(state, "context_loader", started, data)
        return {
            "pre_fetch_bundle": bundle,
            "document_sources": list(bundle.get("document_sources", [])),
            "web_sources": list(bundle.get("web_sources", [])),
            "used_chat_ids": list(bundle.get("used_chat_ids", [])),
            "node_events": _append_event(state, "context_loader", data, started=started, config=config),
        }

    async def planner(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        prompt = build_planner_prompt(state)
        retry_attempts, retry_observer = _llm_retry_observer()
        response = await invoke_with_retry(
            llm.ainvoke,
            [
                SystemMessage(content="You are a strict planner for a scoped RAG workflow."),
                HumanMessage(content=prompt),
            ],
            retry_observer=retry_observer,
        )
        parsed = _safe_json_object(str(getattr(response, "content", "") or ""))
        normalized = normalize_execution_plan(
            parsed,
            use_web_search=bool(state.get("use_web_search", False)),
            question=state.get("question"),
        )
        worker_summary = selected_and_skipped_workers(
            normalized["execution_plan"],
            PLAN_EXECUTE_WORKER_NODES,
        )
        data = {
            "status": "completed",
            "route": normalized["route"],
            "route_reason": normalized["route_reason"],
            "execution_plan": normalized["execution_plan"],
            "input_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "prefetch": compact_preview(_format_prefetch_summary(state.get("pre_fetch_bundle") or {})),
            },
            "prompt_summary": prompt_summary(
                "Planner Node Prompt",
                "You are a strict planner for a scoped RAG workflow.",
                prompt,
            ),
            "llm_result_summary": {
                "parsed": bool(parsed),
                "route": normalized["route"],
                "route_reason": normalized["route_reason"],
                "execution_plan": normalized["execution_plan"],
                "clarification_option_count": len(normalized.get("clarification_options") or []),
                "normalization_notes": normalized.get("normalization_notes") or [],
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    retry_attempts=retry_attempts,
                ),
                **worker_summary,
            },
            "output_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
            "output_preview": worker_summary,
        }
        _log_node_end(state, "planner", started, data)
        return {
            **normalized,
            "node_events": _append_event(state, "planner", data, started=started, config=config),
        }

    async def router(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        prompt = build_router_prompt(state)
        retry_attempts, retry_observer = _llm_retry_observer()
        response = await invoke_with_retry(
            llm.ainvoke,
            [
                SystemMessage(content="You are a strict router for a RAG workflow."),
                HumanMessage(content=prompt),
            ],
            retry_observer=retry_observer,
        )
        parsed = _safe_json_object(str(getattr(response, "content", "") or ""))
        allowed_routes = {"document", "memory", "timeline", "direct", "clarify"}
        if state.get("use_web_search", False):
            allowed_routes.add("web")
        route = parsed.get("route") if parsed.get("route") in allowed_routes else "document"
        clarification_options = parsed.get("clarification_options")
        if route == "clarify" and not isinstance(clarification_options, list):
            clarification_options = _fallback_clarification_options()
        route_reason = str(parsed.get("reason") or "")
        data = {
            "status": "completed",
            "route": route,
            "route_reason": route_reason,
            "input_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "prefetch": compact_preview(_format_prefetch_summary(state.get("pre_fetch_bundle") or {})),
            },
            "prompt_summary": prompt_summary(
                "Router Node Prompt",
                "You are a strict router for a RAG workflow.",
                prompt,
            ),
            "llm_result_summary": {
                "parsed": bool(parsed),
                "route": route,
                "route_reason": route_reason,
                "clarification_option_count": len(clarification_options or []) if route == "clarify" else 0,
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    retry_attempts=retry_attempts,
                ),
            },
            "output_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
        }
        _log_node_end(state, "router", started, data)
        return {
            "route": route,
            "route_reason": route_reason,
            "clarification_options": clarification_options if route == "clarify" else None,
            "node_events": _append_event(state, "router", data, started=started, config=config),
        }

    async def retrieval_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        if _should_skip_worker(state, "retrieval_worker"):
            return _skipped_worker_update(state, config, "retrieval_worker", started, "not_selected_by_plan")
        tool_name = "search_documents"
        tool_config = _tool_config(state, config, caller_node="retrieval_worker", tool_name=tool_name)
        tool_input = {"query": state["question"], "max_results": 10}
        raw = await search_documents.ainvoke(
            tool_input,
            config=tool_config,
        )
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        document_sources = [*state.get("document_sources", []), *artifacts.get("document_sources", [])]
        web_sources = [*state.get("web_sources", []), *artifacts.get("web_sources", [])]
        evidence = _combine_evidence(state.get("evidence"), payload.get("content", ""), label="Document evidence")
        data = {
            "status": "completed" if payload.get("ok", True) else "failed",
            "warnings": normalize_warnings(payload.get("warnings")),
            "input_refs": _state_evidence_refs(state),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "previous_evidence": compact_preview(state.get("evidence")),
            },
            "evidence_chars": len(str(evidence or "")),
            "document_source_count": len(document_sources),
            "web_source_count": len(web_sources),
            "output_refs": compact_refs(
                {
                    **_state_evidence_refs(
                        {
                            **state,
                            "document_sources": document_sources,
                            "web_sources": web_sources,
                        }
                    ),
                    **refs_from_artifacts(artifacts),
                }
            ),
            "output_preview": {"evidence": compact_preview(evidence)},
        }
        _log_node_end(state, "retrieval_worker", started, data)
        return {
            "evidence": evidence,
            "document_sources": document_sources,
            "web_sources": web_sources,
            "node_events": _append_event(state, "retrieval_worker", data, started=started, config=config),
            "tool_events": _append_tool_event(state, payload, tool_input=tool_input, config=config),
        }

    async def memory_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        if _should_skip_worker(state, "memory_worker"):
            return _skipped_worker_update(state, config, "memory_worker", started, "not_selected_by_plan")
        tool_name = "search_conversation_history"
        tool_config = _tool_config(state, config, caller_node="memory_worker", tool_name=tool_name)
        tool_input = {"query": state["question"], "max_results": 10}
        raw = await search_conversation_history.ainvoke(
            tool_input,
            config=tool_config,
        )
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        evidence = _combine_evidence(state.get("evidence"), payload.get("content", ""), label="Memory evidence")
        used_chat_ids = [*state.get("used_chat_ids", []), *artifacts.get("used_chat_ids", [])]
        data = {
            "status": "completed" if payload.get("ok", True) else "failed",
            "warnings": normalize_warnings(payload.get("warnings")),
            "input_refs": _state_evidence_refs(state),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "previous_evidence": compact_preview(state.get("evidence")),
            },
            "evidence_chars": len(str(evidence or "")),
            "used_chat_id_count": len(used_chat_ids),
            "output_refs": compact_refs(
                {
                    **_state_evidence_refs({**state, "used_chat_ids": used_chat_ids}),
                    **refs_from_artifacts(artifacts),
                }
            ),
            "output_preview": {"evidence": compact_preview(evidence)},
        }
        _log_node_end(state, "memory_worker", started, data)
        return {
            "evidence": evidence,
            "used_chat_ids": used_chat_ids,
            "node_events": _append_event(state, "memory_worker", data, started=started, config=config),
            "tool_events": _append_tool_event(state, payload, tool_input=tool_input, config=config),
        }

    async def timeline_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        if _should_skip_worker(state, "timeline_worker"):
            return _skipped_worker_update(state, config, "timeline_worker", started, "not_selected_by_plan")
        tool_name = "search_thread_timeline"
        tool_config = _tool_config(state, config, caller_node="timeline_worker", tool_name=tool_name)
        tool_input = {"query": state["question"], "sources": "all", "order": "relevance", "max_results": 10}
        raw = await search_thread_timeline.ainvoke(
            tool_input,
            config=tool_config,
        )
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        evidence = _combine_evidence(state.get("evidence"), payload.get("content", ""), label="Timeline evidence")
        data = {
            "status": "completed" if payload.get("ok", True) else "failed",
            "warnings": normalize_warnings(payload.get("warnings")),
            "input_refs": _state_evidence_refs(state),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "previous_evidence": compact_preview(state.get("evidence")),
            },
            "evidence_chars": len(str(evidence or "")),
            "timeline_event_count": len(artifacts.get("timeline_events", []) or []),
            "output_refs": compact_refs(
                {
                    **_state_evidence_refs(state),
                    "timeline_events": refs_from_timeline(artifacts.get("timeline_events")),
                }
            ),
            "output_preview": {"evidence": compact_preview(evidence)},
        }
        _log_node_end(state, "timeline_worker", started, data)
        return {
            "evidence": evidence,
            "node_events": _append_event(state, "timeline_worker", data, started=started, config=config),
            "tool_events": _append_tool_event(state, payload, tool_input=tool_input, config=config),
        }

    async def web_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        if isinstance(state.get("execution_plan"), list) and not state.get("use_web_search", False):
            return _skipped_worker_update(state, config, "web_worker", started, "web_search_disabled")
        if _should_skip_worker(state, "web_worker"):
            return _skipped_worker_update(state, config, "web_worker", started, "not_selected_by_plan")
        tool_name = "search_web"
        tool_config = _tool_config(state, config, caller_node="web_worker", tool_name=tool_name)
        tool_input = state["question"]
        raw = await search_web.ainvoke(tool_input, config=tool_config)
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        web_sources = [*state.get("web_sources", []), *artifacts.get("web_sources", [])]
        evidence = _combine_evidence(state.get("evidence"), payload.get("content", ""), label="Web evidence")
        data = {
            "status": "completed" if payload.get("ok", True) else "failed",
            "warnings": normalize_warnings(payload.get("warnings")),
            "input_refs": _state_evidence_refs(state),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "previous_evidence": compact_preview(state.get("evidence")),
            },
            "evidence_chars": len(str(evidence or "")),
            "web_source_count": len(web_sources),
            "output_refs": compact_refs(
                {
                    **_state_evidence_refs({**state, "web_sources": web_sources}),
                    **refs_from_artifacts(artifacts),
                }
            ),
            "output_preview": {"evidence": compact_preview(evidence)},
        }
        _log_node_end(state, "web_worker", started, data)
        return {
            "evidence": evidence,
            "web_sources": web_sources,
            "node_events": _append_event(state, "web_worker", data, started=started, config=config),
            "tool_events": _append_tool_event(state, payload, tool_input=tool_input, config=config),
        }

    async def direct_answer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._answer_from_context(state, config, node_name="direct_answer")

    async def synthesizer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._answer_from_context(state, config, node_name="synthesizer")

    async def _answer_from_context(self, state: RouterRagState, config: RunnableConfig, *, node_name: str) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        context = state.get("evidence") or _format_prefetch_summary(state.get("pre_fetch_bundle") or {})
        context_source = "worker_evidence" if state.get("evidence") else "prefetch"
        messages = build_final_answer_messages(state, context)
        retry_attempts, retry_observer = _llm_retry_observer()
        response = await invoke_with_retry(
            llm.ainvoke,
            [
                SystemMessage(content=messages["system"]),
                HumanMessage(content=messages["human"]),
            ],
            retry_observer=retry_observer,
        )
        normalized = normalize_ai_response(response)
        data = {
            "status": "completed",
            "input_refs": _state_evidence_refs(state) or _prefetch_refs(state.get("pre_fetch_bundle") or {}),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "context_source": context_source,
                "context": compact_preview(context),
            },
            "prompt_summary": prompt_summary(
                "Final Answer Prompt",
                messages["system"],
                messages["human"],
            ),
            "llm_result_summary": {
                "answer_chars": len(normalized["answer"] or ""),
                "reasoning_available": bool(normalized["reasoning_available"]),
                "reasoning_format": normalized["reasoning_format"],
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    normalized_response=normalized,
                    retry_attempts=retry_attempts,
                ),
            },
            "answer_chars": len(normalized["answer"] or ""),
            "evidence_chars": len(str(context or "")),
            "output_refs": _state_evidence_refs(state) or _prefetch_refs(state.get("pre_fetch_bundle") or {}),
            "output_preview": {"answer": compact_preview(normalized["answer"])},
        }
        _log_node_end(state, node_name, started, data)
        return {
            "final_answer": normalized["answer"],
            "reasoning": normalized["reasoning"],
            "reasoning_available": normalized["reasoning_available"],
            "reasoning_format": normalized["reasoning_format"],
            "node_events": _append_event(state, node_name, data, started=started, config=config),
        }

    async def finalizer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        if state.get("clarification_options") and not state.get("final_answer"):
            answer = "I need a bit more clarification. Did you mean:\n" + "\n".join(
                f"- {option}" for option in state["clarification_options"]
            )
            data = {
                "status": "completed",
                "answer_chars": len(answer),
                "output_preview": {
                    "answer": compact_preview(answer),
                    "clarification_options": state.get("clarification_options"),
                },
                "llm_result_summary": {
                    "clarification_option_count": len(state.get("clarification_options") or []),
                },
            }
            _log_node_end(state, "finalizer", started, data)
            return {
                "final_answer": answer,
                "reasoning": "",
                "reasoning_available": False,
                "reasoning_format": "none",
                "node_events": _append_event(state, "finalizer", data, started=started, config=config),
            }
        data = {
            "status": "completed",
            "answer_chars": len(state.get("final_answer") or ""),
            "output_refs": _state_evidence_refs(state),
            "output_preview": {"answer": compact_preview(state.get("final_answer"))},
        }
        _log_node_end(state, "finalizer", started, data)
        return {"node_events": _append_event(state, "finalizer", data, started=started, config=config)}


def router_route(state: RouterRagState) -> str:
    return state.get("route") or "document"


def planner_route(state: RouterRagState) -> str:
    route = state.get("route")
    return route if route in {"execute", "direct", "clarify"} else "execute"


class TemplateCompiler:
    """Compile validated v2 template specs into LangGraph StateGraph instances."""

    def __init__(self, registry: Optional[NodeRegistry] = None):
        self.registry = registry or NodeRegistry()

    def compile(self, spec: Dict[str, Any]):
        from app.agent_patterns.validator import TemplateValidator

        TemplateValidator().validate(spec)
        graph_spec = (spec.get("config") or {}).get("graph") or {}
        workflow = StateGraph(RouterRagState)
        for node in graph_spec.get("nodes", []):
            workflow.add_node(node["id"], self.registry.get(node["type"]))

        for edge in graph_spec.get("edges", []):
            source = edge.get("from")
            target = edge.get("to")
            if edge.get("conditional"):
                route_fn = planner_route if source == "planner" else router_route
                workflow.add_conditional_edges(source, route_fn, edge["routes"])
                continue
            source_ref = START if source == "START" else source
            target_ref = END if target == "END" else target
            workflow.add_edge(source_ref, target_ref)

        return workflow.compile()
