from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agent.reasoning import normalize_ai_response
from app.agent.prompting import sanitize_custom_instructions, sanitize_system_role
from app.agent.tool_contract import compact_tool_event, normalize_tool_result
from app.agent.tool_registry import get_tool_contract_id, validate_tool_call_allowed
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET, get_llm
from app.models.retry import invoke_with_retry
from app.agent.external_research_tools import search_web
from app.rag.agent_tools import search_conversation_history, search_documents, search_thread_timeline
from app.rag.chat_service import prefetch_context
from app.agent_patterns.templates import PLAN_EXECUTE_WORKER_NODES


RouterRoute = Literal["document", "memory", "timeline", "web", "direct", "clarify"]

logger = logging.getLogger(__name__)


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
        event["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    telemetry_sink = ((config or {}).get("configurable") or {}).get("telemetry_sink")
    if isinstance(telemetry_sink, dict):
        telemetry_sink.setdefault("node_events", []).append(dict(event))
    return [*state.get("node_events", []), event]


def _append_tool_event(
    state: RouterRagState,
    payload: Dict[str, Any],
    *,
    config: Optional[RunnableConfig] = None,
) -> List[Dict[str, Any]]:
    event = compact_tool_event(payload)
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
    data = {"skipped": True, "skip_reason": reason}
    _log_node_end(state, worker_node, started, data)
    return {"node_events": _append_event(state, worker_node, data, started=started, config=config)}


def normalize_execution_plan(parsed: Dict[str, Any], *, use_web_search: bool) -> Dict[str, Any]:
    allowed_routes = {"execute", "direct", "clarify"}
    route = parsed.get("route") if parsed.get("route") in allowed_routes else "execute"
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
    if not use_web_search:
        steps = [step for step in steps if step != "web_worker"]
    if route == "execute" and not steps:
        steps = ["retrieval_worker"]
    if route != "execute":
        steps = []
    clarification_options = parsed.get("clarification_options")
    if route == "clarify" and not isinstance(clarification_options, list):
        clarification_options = [
            "Ask about uploaded document evidence.",
            "Ask about prior conversation context.",
        ]
    return {
        "route": route,
        "route_reason": str(parsed.get("reason") or parsed.get("route_reason") or ""),
        "execution_plan": steps,
        "clarification_options": clarification_options if route == "clarify" else None,
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
            "document_source_count": len(bundle.get("document_sources", [])),
            "web_source_count": len(bundle.get("web_sources", [])),
            "used_chat_id_count": len(bundle.get("used_chat_ids", [])),
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
        prompt = (
            "Create a bounded retrieval plan for this askPDF question.\n"
            "Routes:\n"
            "- execute: run one or more retrieval workers, then synthesize.\n"
            "- direct: pre-fetched context is enough for a concise answer.\n"
            "- clarify: the question is ambiguous and needs 2-4 options.\n\n"
            "Worker nodes available for execute:\n"
            "- retrieval_worker: uploaded document or cached web snippet evidence.\n"
            "- memory_worker: prior conversation memory.\n"
            "- timeline_worker: chronology, first/latest, before/after, since, event ordering.\n"
            "- web_worker: live internet evidence, only when live web search is enabled.\n\n"
            f"Live web search enabled: {bool(state.get('use_web_search', False))}\n"
            "Do not include web_worker when live web search is disabled.\n"
            "Return only JSON with keys route, reason, execution_plan, clarification_options.\n"
            "execution_plan must be an array of worker node IDs and must be empty unless route is execute.\n"
            "clarification_options must be null unless route is clarify.\n\n"
            f"Question:\n{state['question']}\n\n"
            f"Pre-fetched context:\n{_format_prefetch_summary(state.get('pre_fetch_bundle') or {})}"
        )
        response = await invoke_with_retry(
            llm.ainvoke,
            [
                SystemMessage(content="You are a strict planner for a scoped RAG workflow."),
                HumanMessage(content=prompt),
            ],
        )
        parsed = _safe_json_object(str(getattr(response, "content", "") or ""))
        normalized = normalize_execution_plan(parsed, use_web_search=bool(state.get("use_web_search", False)))
        data = {
            "route": normalized["route"],
            "route_reason": normalized["route_reason"],
            "execution_plan": normalized["execution_plan"],
        }
        _log_node_end(state, "planner", started, data)
        return {
            **normalized,
            "node_events": _append_event(state, "planner", data, started=started, config=config),
        }

    async def router(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        prompt = (
            "Route this askPDF question to exactly one route.\n"
            "Routes:\n"
            "- document: answer needs uploaded document evidence or cached web snippets.\n"
            "- memory: answer needs prior conversation memory.\n"
            "- timeline: answer depends on chronology, first/latest, before/after, since, or event ordering.\n"
            "- web: answer needs live internet evidence and web search is enabled.\n"
            "- direct: pre-fetched context is enough for a concise answer.\n"
            "- clarify: the question is ambiguous and needs 2-4 options.\n\n"
            f"Live web search enabled: {bool(state.get('use_web_search', False))}\n"
            "Do not choose web when live web search is disabled; choose document, memory, direct, or clarify instead.\n\n"
            "Return only JSON with keys route, reason, clarification_options.\n"
            "clarification_options must be null unless route is clarify.\n\n"
            f"Question:\n{state['question']}\n\n"
            f"Pre-fetched context:\n{_format_prefetch_summary(state.get('pre_fetch_bundle') or {})}"
        )
        response = await invoke_with_retry(
            llm.ainvoke,
            [
                SystemMessage(content="You are a strict router for a RAG workflow."),
                HumanMessage(content=prompt),
            ],
        )
        parsed = _safe_json_object(str(getattr(response, "content", "") or ""))
        allowed_routes = {"document", "memory", "timeline", "direct", "clarify"}
        if state.get("use_web_search", False):
            allowed_routes.add("web")
        route = parsed.get("route") if parsed.get("route") in allowed_routes else "document"
        clarification_options = parsed.get("clarification_options")
        if route == "clarify" and not isinstance(clarification_options, list):
            clarification_options = [
                "Ask about the uploaded document content.",
                "Ask about prior conversation context.",
            ]
        route_reason = str(parsed.get("reason") or "")
        data = {"route": route, "route_reason": route_reason}
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
        raw = await search_documents.ainvoke(
            {"query": state["question"], "max_results": 10},
            config=tool_config,
        )
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        document_sources = [*state.get("document_sources", []), *artifacts.get("document_sources", [])]
        web_sources = [*state.get("web_sources", []), *artifacts.get("web_sources", [])]
        evidence = _combine_evidence(state.get("evidence"), payload.get("content", ""), label="Document evidence")
        data = {
            "evidence_chars": len(str(evidence or "")),
            "document_source_count": len(document_sources),
            "web_source_count": len(web_sources),
        }
        _log_node_end(state, "retrieval_worker", started, data)
        return {
            "evidence": evidence,
            "document_sources": document_sources,
            "web_sources": web_sources,
            "node_events": _append_event(state, "retrieval_worker", data, started=started, config=config),
            "tool_events": _append_tool_event(state, payload, config=config),
        }

    async def memory_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        if _should_skip_worker(state, "memory_worker"):
            return _skipped_worker_update(state, config, "memory_worker", started, "not_selected_by_plan")
        tool_name = "search_conversation_history"
        tool_config = _tool_config(state, config, caller_node="memory_worker", tool_name=tool_name)
        raw = await search_conversation_history.ainvoke(
            {"query": state["question"], "max_results": 10},
            config=tool_config,
        )
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        evidence = _combine_evidence(state.get("evidence"), payload.get("content", ""), label="Memory evidence")
        used_chat_ids = [*state.get("used_chat_ids", []), *artifacts.get("used_chat_ids", [])]
        data = {
            "evidence_chars": len(str(evidence or "")),
            "used_chat_id_count": len(used_chat_ids),
        }
        _log_node_end(state, "memory_worker", started, data)
        return {
            "evidence": evidence,
            "used_chat_ids": used_chat_ids,
            "node_events": _append_event(state, "memory_worker", data, started=started, config=config),
            "tool_events": _append_tool_event(state, payload, config=config),
        }

    async def timeline_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        if _should_skip_worker(state, "timeline_worker"):
            return _skipped_worker_update(state, config, "timeline_worker", started, "not_selected_by_plan")
        tool_name = "search_thread_timeline"
        tool_config = _tool_config(state, config, caller_node="timeline_worker", tool_name=tool_name)
        raw = await search_thread_timeline.ainvoke(
            {"query": state["question"], "sources": "all", "order": "relevance", "max_results": 10},
            config=tool_config,
        )
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        evidence = _combine_evidence(state.get("evidence"), payload.get("content", ""), label="Timeline evidence")
        data = {
            "evidence_chars": len(str(evidence or "")),
            "timeline_event_count": len(artifacts.get("timeline_events", []) or []),
        }
        _log_node_end(state, "timeline_worker", started, data)
        return {
            "evidence": evidence,
            "node_events": _append_event(state, "timeline_worker", data, started=started, config=config),
            "tool_events": _append_tool_event(state, payload, config=config),
        }

    async def web_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        if isinstance(state.get("execution_plan"), list) and not state.get("use_web_search", False):
            return _skipped_worker_update(state, config, "web_worker", started, "web_search_disabled")
        if _should_skip_worker(state, "web_worker"):
            return _skipped_worker_update(state, config, "web_worker", started, "not_selected_by_plan")
        tool_name = "search_web"
        tool_config = _tool_config(state, config, caller_node="web_worker", tool_name=tool_name)
        raw = await search_web.ainvoke(state["question"], config=tool_config)
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        web_sources = [*state.get("web_sources", []), *artifacts.get("web_sources", [])]
        evidence = _combine_evidence(state.get("evidence"), payload.get("content", ""), label="Web evidence")
        data = {
            "evidence_chars": len(str(evidence or "")),
            "web_source_count": len(web_sources),
        }
        _log_node_end(state, "web_worker", started, data)
        return {
            "evidence": evidence,
            "web_sources": web_sources,
            "node_events": _append_event(state, "web_worker", data, started=started, config=config),
            "tool_events": _append_tool_event(state, payload, config=config),
        }

    async def direct_answer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._answer_from_context(state, config, node_name="direct_answer")

    async def synthesizer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._answer_from_context(state, config, node_name="synthesizer")

    async def _answer_from_context(self, state: RouterRagState, config: RunnableConfig, *, node_name: str) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        context = state.get("evidence") or _format_prefetch_summary(state.get("pre_fetch_bundle") or {})
        system_parts = [
            "You answer askPDF questions using the supplied context. Cite document or web sources when they are present.",
        ]
        system_role = sanitize_system_role(state.get("system_role", ""))
        custom_instructions = sanitize_custom_instructions(state.get("custom_instructions", ""))
        if system_role:
            system_parts.append(system_role)
        if custom_instructions:
            system_parts.append(custom_instructions)
        response = await invoke_with_retry(
            llm.ainvoke,
            [
                SystemMessage(content="\n\n".join(system_parts)),
                HumanMessage(
                    content=(
                        f"Question:\n{state['question']}\n\n"
                        f"Context:\n{context}\n\n"
                        "Write the final answer. If the context is insufficient, say what is missing."
                    )
                ),
            ],
        )
        normalized = normalize_ai_response(response)
        data = {
            "answer_chars": len(normalized["answer"] or ""),
            "evidence_chars": len(str(context or "")),
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
            data = {"answer_chars": len(answer)}
            _log_node_end(state, "finalizer", started, data)
            return {
                "final_answer": answer,
                "reasoning": "",
                "reasoning_available": False,
                "reasoning_format": "none",
                "node_events": _append_event(state, "finalizer", data, started=started, config=config),
            }
        data = {"answer_chars": len(state.get("final_answer") or "")}
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
