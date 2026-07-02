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
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET, get_llm
from app.models.retry import invoke_with_retry
from app.rag.agent_tools import search_conversation_history, search_documents
from app.rag.chat_service import prefetch_context


RouterRoute = Literal["document", "memory", "direct", "clarify"]

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
    errors: List[Dict[str, Any]]


def _append_event(state: RouterRagState, node: str, data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return [*state.get("node_events", []), {"node": node, **(data or {})}]


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


def _tool_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = _safe_json_object(raw)
        if parsed:
            return parsed
        return {"content": raw}
    return {"content": str(raw or "")}


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


class NodeRegistry:
    """Registry of safe backend node implementations for compiled v2 patterns."""

    def __init__(self):
        self._nodes: Dict[str, Callable[..., Any]] = {
            "context_loader": self.context_loader,
            "router": self.router,
            "retrieval_worker": self.retrieval_worker,
            "memory_worker": self.memory_worker,
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
            "node_events": _append_event(state, "context_loader", data),
        }

    async def router(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        prompt = (
            "Route this askPDF question to exactly one route.\n"
            "Routes:\n"
            "- document: answer needs uploaded document evidence or cached web snippets.\n"
            "- memory: answer needs prior conversation memory.\n"
            "- direct: pre-fetched context is enough for a concise answer.\n"
            "- clarify: the question is ambiguous and needs 2-4 options.\n\n"
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
        route = parsed.get("route") if parsed.get("route") in {"document", "memory", "direct", "clarify"} else "document"
        clarification_options = parsed.get("clarification_options")
        if route == "clarify" and not isinstance(clarification_options, list):
            clarification_options = [
                "Ask about the uploaded document content.",
                "Ask about prior conversation context.",
            ]
        data = {"route": route}
        _log_node_end(state, "router", started, data)
        return {
            "route": route,
            "route_reason": str(parsed.get("reason") or ""),
            "clarification_options": clarification_options if route == "clarify" else None,
            "node_events": _append_event(state, "router", data),
        }

    async def retrieval_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        raw = await search_documents.ainvoke(
            {"query": state["question"], "max_results": 10},
            config=config,
        )
        payload = _tool_payload(raw)
        document_sources = [*state.get("document_sources", []), *payload.get("__document_sources__", [])]
        web_sources = [*state.get("web_sources", []), *payload.get("__web_sources__", [])]
        evidence = payload.get("content", "")
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
            "node_events": _append_event(state, "retrieval_worker", data),
        }

    async def memory_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        raw = await search_conversation_history.ainvoke(
            {"query": state["question"], "max_results": 10},
            config=config,
        )
        payload = _tool_payload(raw)
        evidence = payload.get("content", "")
        used_chat_ids = [*state.get("used_chat_ids", []), *payload.get("__used_chat_ids__", [])]
        data = {
            "evidence_chars": len(str(evidence or "")),
            "used_chat_id_count": len(used_chat_ids),
        }
        _log_node_end(state, "memory_worker", started, data)
        return {
            "evidence": evidence,
            "used_chat_ids": used_chat_ids,
            "node_events": _append_event(state, "memory_worker", data),
        }

    async def direct_answer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._answer_from_context(state, node_name="direct_answer")

    async def synthesizer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._answer_from_context(state, node_name="synthesizer")

    async def _answer_from_context(self, state: RouterRagState, *, node_name: str) -> Dict[str, Any]:
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
            "node_events": _append_event(state, node_name, data),
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
                "node_events": _append_event(state, "finalizer", data),
            }
        data = {"answer_chars": len(state.get("final_answer") or "")}
        _log_node_end(state, "finalizer", started, data)
        return {"node_events": _append_event(state, "finalizer", data)}


def router_route(state: RouterRagState) -> str:
    return state.get("route") or "document"


class TemplateCompiler:
    """Compile validated v2 template specs into LangGraph StateGraph instances."""

    def __init__(self, registry: Optional[NodeRegistry] = None):
        self.registry = registry or NodeRegistry()

    def compile(self, spec: Dict[str, Any]):
        graph_spec = (spec.get("config") or {}).get("graph") or {}
        workflow = StateGraph(RouterRagState)
        for node in graph_spec.get("nodes", []):
            workflow.add_node(node["id"], self.registry.get(node["type"]))

        for edge in graph_spec.get("edges", []):
            source = edge.get("from")
            target = edge.get("to")
            if edge.get("conditional"):
                workflow.add_conditional_edges(source, router_route, edge["routes"])
                continue
            source_ref = START if source == "START" else source
            target_ref = END if target == "END" else target
            workflow.add_edge(source_ref, target_ref)

        return workflow.compile()
