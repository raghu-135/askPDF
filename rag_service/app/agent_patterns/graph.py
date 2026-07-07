from __future__ import annotations

import json
import logging
import re
import hashlib
import time
from copy import deepcopy
from datetime import timedelta
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agent.reasoning import normalize_ai_response
from app.agent.tool_contract import compact_tool_event, normalize_tool_result
from app.agent.tool_registry import get_tool_contract_id, validate_tool_call_allowed
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET, get_llm
from app.models.retry import invoke_with_retry
from app.agent.external_research_tools import search_web
from app.rag.agent_tools import search_conversation_history, search_documents, search_thread_timeline
from app.rag.chat_service import prefetch_context
from app.agent_patterns.prompting import (
    build_evaluator_prompt,
    build_final_answer_messages,
    build_planner_prompt,
    build_replanner_prompt,
    build_router_prompt,
)
from app.agent_patterns.node_catalog import get_node_type_metadata, node_type_capabilities, node_type_default_max_visits
from app.agent_patterns.route_registry import route_function_allowed_for_node_type, route_function_runtime_supported
from app.agent_patterns.templates import PLAN_EXECUTE_WORKER_NODES, WEB_APPROVAL_GATE_ID
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
FINAL_REVIEW_GATE_ID = "human_review_gate"
NODE_RUNTIME_CONFIG_KEY = "agent_pattern_node_runtime"


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


def _node_runtime(config: Optional[RunnableConfig]) -> Dict[str, Any]:
    configurable = ((config or {}).get("configurable") or {})
    runtime = configurable.get(NODE_RUNTIME_CONFIG_KEY)
    return runtime if isinstance(runtime, dict) else {}


def _runtime_node_id(config: Optional[RunnableConfig], fallback: str) -> str:
    runtime = _node_runtime(config)
    node_id = runtime.get("node_id")
    return str(node_id) if isinstance(node_id, str) and node_id else fallback


def _runtime_node_type(config: Optional[RunnableConfig], fallback: str) -> str:
    runtime = _node_runtime(config)
    node_type = runtime.get("node_type")
    return str(node_type) if isinstance(node_type, str) and node_type else fallback


def _runtime_node_capabilities(config: Optional[RunnableConfig]) -> List[str]:
    runtime = _node_runtime(config)
    capabilities = runtime.get("capabilities")
    return [str(item) for item in capabilities] if isinstance(capabilities, list) else []


def _runtime_visit_index(config: Optional[RunnableConfig]) -> Optional[int]:
    runtime = _node_runtime(config)
    try:
        value = int(runtime.get("visit_index"))
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _with_node_runtime_config(
    config: Optional[RunnableConfig],
    *,
    node_id: str,
    node_type: str,
    capabilities: List[str],
    visit_index: int,
) -> RunnableConfig:
    updated = dict(config or {})
    configurable = dict(updated.get("configurable") or {})
    configurable[NODE_RUNTIME_CONFIG_KEY] = {
        "node_id": node_id,
        "node_type": node_type,
        "capabilities": list(capabilities),
        "visit_index": visit_index,
    }
    updated["configurable"] = configurable
    metadata = dict(updated.get("metadata") or {})
    metadata.update(
        {
            "node_id": node_id,
            "node_type": node_type,
            "node_capabilities": list(capabilities),
            "node_visit_index": visit_index,
        }
    )
    updated["metadata"] = metadata
    return updated


def _loop_policy(state: RouterRagState) -> Dict[str, Any]:
    policy = state.get("loop_policy")
    return policy if isinstance(policy, dict) else {}


def _node_visit_counts(state: RouterRagState) -> Dict[str, int]:
    counts = state.get("node_visit_counts")
    if not isinstance(counts, dict):
        return {}
    normalized: Dict[str, int] = {}
    for key, value in counts.items():
        if not isinstance(key, str):
            continue
        try:
            normalized[key] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _node_visit_sequence(state: RouterRagState) -> List[Dict[str, Any]]:
    sequence = state.get("node_visit_sequence")
    return [item for item in sequence if isinstance(item, dict)] if isinstance(sequence, list) else []


def _node_visit_limit(state: RouterRagState, *, node_id: str, node_type: str) -> Optional[int]:
    policy = _loop_policy(state)
    if not policy:
        return None
    node_limits = policy.get("node_visit_limits") if isinstance(policy.get("node_visit_limits"), dict) else {}
    if node_id in node_limits:
        try:
            return max(1, int(node_limits[node_id]))
        except (TypeError, ValueError):
            return 1
    try:
        default_limit = int(policy.get("default_max_node_visits", node_type_default_max_visits(node_type)))
    except (TypeError, ValueError):
        default_limit = node_type_default_max_visits(node_type)
    return max(1, min(default_limit, node_type_default_max_visits(node_type)))


def _total_visit_limit(state: RouterRagState) -> Optional[int]:
    policy = _loop_policy(state)
    if not policy:
        return None
    try:
        value = int(policy.get("max_total_visits"))
    except (TypeError, ValueError):
        return None
    return max(1, value)


def _check_visit_budget(state: RouterRagState, *, node_id: str, node_type: str, visit_index: int) -> None:
    limit = _node_visit_limit(state, node_id=node_id, node_type=node_type)
    if limit is not None and visit_index > limit:
        raise ValueError(f"Node {node_id} exceeded visit limit {limit}")
    total_limit = _total_visit_limit(state)
    if total_limit is not None and len(_node_visit_sequence(state)) + 1 > total_limit:
        raise ValueError(f"Graph exceeded total visit limit {total_limit}")


def _with_visit_accounting(
    update: Dict[str, Any],
    state: RouterRagState,
    *,
    node_id: str,
    node_type: str,
    visit_index: int,
) -> Dict[str, Any]:
    counts = _node_visit_counts(state)
    counts[node_id] = max(counts.get(node_id, 0), visit_index)
    sequence = [
        *_node_visit_sequence(state),
        {"node": node_id, "node_type": node_type, "visit_index": visit_index},
    ]
    return {
        **update,
        "node_visit_counts": counts,
        "node_visit_sequence": sequence,
    }


def _hitl_interrupt_counts(state: RouterRagState) -> Dict[str, int]:
    counts = state.get("hitl_interrupt_counts")
    if not isinstance(counts, dict):
        return {}
    normalized: Dict[str, int] = {}
    for key, value in counts.items():
        if not isinstance(key, str):
            continue
        try:
            normalized[key] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _hitl_interrupt_limit(policy: Dict[str, Any], gate_policy: Dict[str, Any]) -> Optional[int]:
    raw = gate_policy.get("max_interrupts_per_run", policy.get("max_interrupts_per_run"))
    if raw in (None, ""):
        return None
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _hitl_interrupt_count_key(gate_id: str, visit_index: Optional[int]) -> str:
    if isinstance(visit_index, int) and visit_index >= 1:
        return f"{gate_id}:visit:{visit_index}"
    return gate_id


def _hitl_visit_interrupt_count(counts: Dict[str, int], *, gate_id: str, visit_index: Optional[int]) -> int:
    visit_key = _hitl_interrupt_count_key(gate_id, visit_index)
    has_visit_counts = any(key.startswith(f"{gate_id}:visit:") for key in counts)
    if visit_key != gate_id and has_visit_counts:
        return counts.get(visit_key, 0)
    return counts.get(gate_id, 0)


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
    human_review_decision: Dict[str, Any]
    hitl_policy: Dict[str, Any]
    hitl_decisions: List[Dict[str, Any]]
    hitl_gate_route: str
    hitl_gate_routes: Dict[str, Any]
    hitl_selected_options: Dict[str, List[str]]
    skipped_nodes: List[str]
    node_events: List[Dict[str, Any]]
    tool_events: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    allowed_tool_ids: List[str]
    pattern_type: str
    loop_policy: Dict[str, Any]
    node_visit_counts: Dict[str, int]
    node_visit_sequence: List[Dict[str, Any]]
    evidence_packets: List[Dict[str, Any]]
    hitl_interrupt_counts: Dict[str, int]
    execution_plan: List[str]
    replans: int
    replan_count: int
    replan_reason: str
    replan_history: List[Dict[str, Any]]
    evaluator_report: Dict[str, Any]
    evidence_gaps: List[str]
    evaluation_confidence: float
    evaluator_route: str


def _append_event(
    state: RouterRagState,
    node: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    started: Optional[float] = None,
    config: Optional[RunnableConfig] = None,
) -> List[Dict[str, Any]]:
    event = {
        "node": _runtime_node_id(config, node),
        "node_type": _runtime_node_type(config, node),
        **(data or {}),
    }
    visit_index = _runtime_visit_index(config)
    if visit_index is not None:
        event["visit_index"] = visit_index
    if started is not None:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        completed_at = utc_now()
        event["elapsed_ms"] = elapsed_ms
        event.setdefault("start_time", iso_utc_z(completed_at - timedelta(milliseconds=elapsed_ms)))
        event.setdefault("end_time", iso_utc_z(completed_at))
    telemetry_sink = ((config or {}).get("configurable") or {}).get("telemetry_sink")
    if isinstance(telemetry_sink, dict):
        telemetry_sink.setdefault("node_events", []).append(dict(event))
    trace_recorder = ((config or {}).get("configurable") or {}).get("trace_recorder")
    if trace_recorder is not None and hasattr(trace_recorder, "record_node_event"):
        trace_recorder.record_node_event(dict(event))
    return [*state.get("node_events", []), event]


def _append_tool_event(
    state: RouterRagState,
    payload: Dict[str, Any],
    *,
    tool_input: Any = None,
    config: Optional[RunnableConfig] = None,
) -> List[Dict[str, Any]]:
    event = compact_tool_event(payload, tool_input=tool_input)
    caller_node_type = event.get("caller_node_type") or _runtime_node_type(config, str(event.get("caller_node") or ""))
    if caller_node_type:
        event["caller_node_type"] = caller_node_type
    visit_index = _runtime_visit_index(config)
    if visit_index is not None:
        event["caller_visit_index"] = visit_index
    telemetry_sink = ((config or {}).get("configurable") or {}).get("telemetry_sink")
    if isinstance(telemetry_sink, dict):
        telemetry_sink.setdefault("tool_events", []).append(dict(event))
    trace_recorder = ((config or {}).get("configurable") or {}).get("trace_recorder")
    if trace_recorder is not None and hasattr(trace_recorder, "record_tool_event"):
        trace_recorder.record_tool_event(dict(event))
    return [*state.get("tool_events", []), event]


def _error_summary(exc: Exception, *, code: str) -> Dict[str, Any]:
    return {
        "code": code,
        "type": type(exc).__name__,
        "message": compact_preview(str(exc), limit=700),
        "raw_message": compact_preview(str(exc), limit=700),
        "retryable": True,
    }


def _append_failed_node_event(
    state: RouterRagState,
    config: RunnableConfig,
    node: str,
    started: float,
    exc: Exception,
    *,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "status": "failed",
        "error": _error_summary(exc, code=f"{node}_failed"),
        "input_preview": {"question": compact_preview(state.get("question"))},
        **(data or {}),
    }
    _append_event(state, node, payload, started=started, config=config)


async def _invoke_llm_for_node(
    func: Callable[..., Any],
    messages: List[Any],
    *,
    state: RouterRagState,
    config: RunnableConfig,
    node: str,
    started: float,
    retry_observer: Callable[[Dict[str, Any]], None],
    retry_attempts: List[Dict[str, Any]],
    model_name: Optional[str],
    failure_data: Optional[Dict[str, Any]] = None,
) -> Any:
    try:
        return await invoke_with_retry(func, messages, retry_observer=retry_observer)
    except Exception as exc:
        llm_failure = {
            "llm_result_summary": {
                "llm": {
                    "model_name": model_name,
                    "retry_count": len(retry_attempts),
                    "retry_attempts": retry_attempts,
                }
            }
        }
        _append_failed_node_event(
            state,
            config,
            node,
            started,
            exc,
            data={**(failure_data or {}), **llm_failure},
        )
        raise


async def _invoke_tool_for_node(
    tool: Any,
    tool_input: Any,
    *,
    state: RouterRagState,
    config: RunnableConfig,
    node: str,
    started: float,
) -> Any:
    try:
        return await tool.ainvoke(tool_input, config=config)
    except Exception as exc:
        _append_failed_node_event(
            state,
            config,
            node,
            started,
            exc,
            data={"input_preview": {"tool_input": tool_input}},
        )
        raise


def _tool_config(state: RouterRagState, config: RunnableConfig, *, caller_node: str, tool_name: str) -> RunnableConfig:
    caller_node_id = _runtime_node_id(config, caller_node)
    caller_node_type = _runtime_node_type(config, caller_node)
    caller_capabilities = _runtime_node_capabilities(config) or node_type_capabilities(caller_node_type)
    validate_tool_call_allowed(
        tool_name,
        caller_node_id,
        caller_node_type=caller_node_type,
        caller_capabilities=caller_capabilities,
    )
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
            "caller_node": caller_node_id,
            "caller_node_type": caller_node_type,
            "caller_capabilities": caller_capabilities,
            "route": state.get("route"),
            "tool_name": tool_name,
        }
    )
    updated["configurable"] = configurable
    metadata = dict(updated.get("metadata") or {})
    metadata.update(
        {
            "agent_run_id": state.get("agent_run_id"),
            "caller_node": caller_node_id,
            "caller_node_type": caller_node_type,
            "caller_capabilities": caller_capabilities,
            "route": state.get("route"),
            "tool_name": tool_name,
        }
    )
    updated["metadata"] = metadata
    return updated


def _tool_config_for_node(
    state: RouterRagState,
    config: RunnableConfig,
    *,
    caller_node: str,
    tool_name: str,
    started: float,
) -> RunnableConfig:
    try:
        return _tool_config(state, config, caller_node=caller_node, tool_name=tool_name)
    except Exception as exc:
        _append_failed_node_event(
            state,
            config,
            caller_node,
            started,
            exc,
            data={"input_preview": {"tool_name": tool_name}},
        )
        raise


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


def _combine_evidence(existing: Any, addition: Any, *, label: str, limit: Optional[int] = None) -> str:
    existing_text = str(existing or "").strip()
    addition_text = str(addition or "").strip()
    if not addition_text:
        return existing_text
    labeled = f"[{label}]\n{addition_text}"
    combined = "\n\n".join(part for part in (existing_text, labeled) if part).strip()
    if isinstance(limit, int) and limit > 0 and len(combined) > limit:
        return combined[-limit:].lstrip()
    return combined


EVIDENCE_PACKET_LIMIT = 12
EVIDENCE_PACKET_CONTENT_LIMIT = 2_000
EVIDENCE_TEXT_LIMIT = EVIDENCE_PACKET_LIMIT * (EVIDENCE_PACKET_CONTENT_LIMIT + 128)
FINAL_CONTEXT_CHAR_LIMIT = EVIDENCE_TEXT_LIMIT


def _context_policy(state: RouterRagState) -> Dict[str, Any]:
    policy = state.get("context_policy")
    return policy if isinstance(policy, dict) else {}


def _context_policy_int(state: RouterRagState, key: str, default: int) -> int:
    try:
        value = int(_context_policy(state).get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _evidence_packet_limit(state: RouterRagState) -> int:
    return _context_policy_int(state, "evidence_packet_limit", EVIDENCE_PACKET_LIMIT)


def _evidence_packet_content_limit(state: RouterRagState) -> int:
    return _context_policy_int(state, "evidence_packet_content_limit", EVIDENCE_PACKET_CONTENT_LIMIT)


def _evidence_text_limit(state: RouterRagState) -> int:
    packet_limit = _evidence_packet_limit(state)
    content_limit = _evidence_packet_content_limit(state)
    if "context_policy" not in state:
        return EVIDENCE_TEXT_LIMIT
    return max(1, packet_limit * (content_limit + 128))


def _final_context_char_limit(state: RouterRagState) -> int:
    return _context_policy_int(state, "final_context_char_limit", _evidence_text_limit(state) or FINAL_CONTEXT_CHAR_LIMIT)


def _evidence_dedupe_enabled(state: RouterRagState) -> bool:
    value = _context_policy(state).get("evidence_dedupe", True)
    return value is not False


def _evidence_compression_mode(state: RouterRagState) -> str:
    mode = _context_policy(state).get("evidence_compression", "compact")
    return str(mode or "compact")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        return json.dumps(str(value), ensure_ascii=True)


def _normalized_evidence_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _short_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:16]


def _packet_fingerprint(*, kind: str, content: str, refs: Dict[str, Any]) -> str:
    return _short_hash(
        {
            "kind": kind,
            "content": _normalized_evidence_text(content),
            "refs": refs or {},
        }
    )


def _dedupe_evidence_packets(packets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped_reversed: List[Dict[str, Any]] = []
    for packet in reversed(packets):
        fingerprint = str(packet.get("fingerprint") or "")
        if not fingerprint:
            fingerprint = _packet_fingerprint(
                kind=str(packet.get("kind") or packet.get("producer_node_type") or "evidence"),
                content=str(packet.get("content") or ""),
                refs=packet.get("refs") if isinstance(packet.get("refs"), dict) else {},
            )
            packet = {**packet, "fingerprint": fingerprint}
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped_reversed.append(packet)
    return list(reversed(deduped_reversed))


def _compact_context_text(text: str, *, limit: int, mode: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if mode == "compact":
        lines: List[str] = []
        seen_lines: set[str] = set()
        for raw_line in value.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                if lines and lines[-1]:
                    lines.append("")
                continue
            key = line.lower()
            if key in seen_lines:
                continue
            seen_lines.add(key)
            lines.append(line)
        value = "\n".join(lines).strip()
    if isinstance(limit, int) and limit > 0 and len(value) > limit:
        return value[-limit:].lstrip()
    return value


def _evidence_packets(state: RouterRagState) -> List[Dict[str, Any]]:
    packets = state.get("evidence_packets")
    normalized = [item for item in packets if isinstance(item, dict)] if isinstance(packets, list) else []
    if _evidence_dedupe_enabled(state):
        normalized = _dedupe_evidence_packets(normalized)
    return normalized


def _evidence_context_from_packets(state: RouterRagState) -> str:
    parts = []
    for packet in _evidence_packets(state)[-_evidence_packet_limit(state):]:
        content = compact_preview(packet.get("content"), limit=_evidence_packet_content_limit(state))
        if not content:
            continue
        kind = str(packet.get("kind") or packet.get("producer_node_type") or "evidence")
        producer = str(packet.get("producer_node_id") or packet.get("producer_node_type") or "unknown")
        parts.append(f"[{kind} evidence from {producer}]\n{content}")
    return _compact_context_text(
        "\n\n".join(parts).strip(),
        limit=_final_context_char_limit(state),
        mode=_evidence_compression_mode(state),
    )


def _final_context_from_state(state: RouterRagState) -> tuple[str, str]:
    policy = _context_policy(state)
    if policy.get("final_prompt_assembly") == "evidence_packets":
        packet_context = _evidence_context_from_packets(state)
        if packet_context:
            return packet_context, "evidence_packets"
    if state.get("evidence"):
        return _compact_context_text(
            str(state.get("evidence") or ""),
            limit=_final_context_char_limit(state),
            mode=_evidence_compression_mode(state),
        ), "worker_evidence"
    return _compact_context_text(
        _format_prefetch_summary(state.get("pre_fetch_bundle") or {}),
        limit=_final_context_char_limit(state),
        mode=_evidence_compression_mode(state),
    ), "prefetch"


def _append_evidence_packet(
    state: RouterRagState,
    config: RunnableConfig,
    *,
    kind: str,
    content: Any,
    refs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    text = compact_preview(content, limit=_evidence_packet_content_limit(state))
    if not text:
        return _evidence_packets(state)
    node_id = _runtime_node_id(config, kind)
    node_type = _runtime_node_type(config, kind)
    visit_index = _runtime_visit_index(config) or 1
    refs = refs or {}
    fingerprint = _packet_fingerprint(kind=kind, content=text, refs=refs)
    existing_packets = _evidence_packets(state)
    if _evidence_dedupe_enabled(state):
        existing_packets = [packet for packet in existing_packets if packet.get("fingerprint") != fingerprint]
    packet = {
        "id": f"{node_id}:visit:{visit_index}:{kind}:{len(existing_packets) + 1}",
        "producer_node_id": node_id,
        "producer_node_type": node_type,
        "visit_index": visit_index,
        "kind": kind,
        "content": text,
        "content_hash": _short_hash(_normalized_evidence_text(text)),
        "fingerprint": fingerprint,
        "refs": refs,
        "created_at": iso_utc_z(utc_now()),
    }
    return [*existing_packets, packet][-_evidence_packet_limit(state):]


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


def _hitl_gates_from_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    return policy.get("gates") if isinstance(policy.get("gates"), dict) else {}


def _normalize_hitl_gate_policy(gate_id: str, gate_policy: Any) -> Dict[str, Any]:
    gate = dict(gate_policy) if isinstance(gate_policy, dict) else {}
    if gate_id == WEB_APPROVAL_GATE_ID:
        gate.setdefault("mode", "approval")
        gate.setdefault("phase", "before")
        gate.setdefault("target", {"node_id": "web_worker", "node_type": "web_worker"})
        gate.setdefault("interrupt_type", "tool_approval")
        gate.setdefault("title", "Approve web search?")
        gate.setdefault(
            "prompt",
            "This answer needs live web research. Approve web search or continue without it.",
        )
        gate.setdefault("allowed_actions", ["approve", "continue_without"])
        gate.setdefault("default_action", "continue_without")
        gate.setdefault("routes", {"approve": "web_worker", "continue_without": "synthesizer"})
    if gate_id == FINAL_REVIEW_GATE_ID:
        gate.setdefault("mode", "review")
        gate.setdefault("phase", "after")
        gate.setdefault("target", {"node_id": "finalizer", "node_type": "finalizer"})
        gate.setdefault("interrupt_type", "final_answer_review")
        gate.setdefault("title", "Review final answer")
        gate.setdefault("prompt", "Approve this answer before it is saved to the thread.")
        gate.setdefault("allowed_actions", ["approve", "edit", "continue_without", "reject"])
        gate.setdefault("default_action", "approve")
        gate.setdefault("routes", {"approve": "END", "edit": "END", "continue_without": "END"})
        gate.setdefault("editable_fields", ["final_answer"])
    gate.setdefault("mode", "approval")
    gate.setdefault("phase", "before")
    if not isinstance(gate.get("routes"), dict):
        gate["routes"] = {}
    if not isinstance(gate.get("allowed_actions"), list):
        gate["allowed_actions"] = ["approve_selected", "continue_without"] if gate.get("mode") == "choice" else ["approve", "continue_without"]
    if not isinstance(gate.get("default_action"), str):
        gate["default_action"] = "approve_selected" if gate.get("mode") == "choice" else "approve"
    return gate


def _normalize_hitl_actions(gate: Dict[str, Any]) -> List[str]:
    allowed = gate.get("allowed_actions")
    if not isinstance(allowed, list) or not all(isinstance(action, str) for action in allowed):
        allowed = ["approve_selected", "continue_without"] if gate.get("mode") == "choice" else ["approve", "continue_without"]
    allowed = [action for action in allowed if action in {"approve", "approve_selected", "continue_without", "reject", "edit"}]
    return allowed or ["approve", "continue_without"]


def _hitl_option_ids(gate: Dict[str, Any]) -> List[str]:
    options = gate.get("options") if isinstance(gate.get("options"), list) else []
    return [
        str(option.get("id"))
        for option in options
        if isinstance(option, dict) and isinstance(option.get("id"), str) and option.get("id")
    ]


def _hitl_selected_option_ids(decision: Dict[str, Any], gate: Dict[str, Any]) -> List[str]:
    valid_ids = _hitl_option_ids(gate)
    selected = decision.get("selected_option_ids")
    if isinstance(selected, str):
        selected = [selected]
    if not isinstance(selected, list):
        selected = []
    normalized = [str(item) for item in selected if str(item) in valid_ids]
    selection_mode = str(gate.get("selection_mode") or "single")
    if selection_mode == "single" and len(normalized) > 1:
        normalized = normalized[:1]
    return normalized


def _hitl_option_targets(gate: Dict[str, Any], selected_option_ids: List[str]) -> List[str]:
    options = gate.get("options") if isinstance(gate.get("options"), list) else []
    selected = set(selected_option_ids)
    targets: List[str] = []
    for option in options:
        if not isinstance(option, dict) or option.get("id") not in selected:
            continue
        target = option.get("target_node_id")
        if isinstance(target, str) and target not in targets:
            targets.append(target)
    return targets


def with_web_approval_hitl_policy(policy: Any) -> Dict[str, Any]:
    """Return a policy with the reusable before-web approval gate enabled."""

    normalized = deepcopy(policy) if isinstance(policy, dict) else {}
    normalized["enabled"] = True
    gates = dict(normalized.get("gates") or {})
    gates[WEB_APPROVAL_GATE_ID] = _normalize_hitl_gate_policy(
        WEB_APPROVAL_GATE_ID,
        gates.get(WEB_APPROVAL_GATE_ID),
    )
    normalized["gates"] = gates
    return normalized


def normalize_hitl_policy_for_thread_settings(policy: Any, thread_settings: Any = None) -> Dict[str, Any]:
    """Normalize legacy thread-level HITL toggles into the reusable policy contract."""

    normalized = deepcopy(policy) if isinstance(policy, dict) else {}
    if isinstance(thread_settings, dict) and bool(thread_settings.get("hitl_web_approval")):
        return with_web_approval_hitl_policy(normalized)
    return normalized


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


def _risk_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"low", "medium", "high"} else "medium"


def _bounded_string_list(value: Any, *, limit: int = 5, chars: int = 240) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value[:limit]:
        text = compact_preview(str(item), limit=chars)
        if text:
            result.append(text)
    return result


def _bounded_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _replan_budget(state: RouterRagState) -> int:
    try:
        return max(1, int(state.get("replans", 1)))
    except (TypeError, ValueError):
        return 1


def _current_replan_count(state: RouterRagState) -> int:
    try:
        return max(0, int(state.get("replan_count", 0)))
    except (TypeError, ValueError):
        return 0


def normalize_evaluator_report(parsed: Dict[str, Any], state: RouterRagState) -> Dict[str, Any]:
    sufficient = parsed.get("sufficient")
    if not isinstance(sufficient, bool):
        sufficient = bool(state.get("evidence")) and bool(state.get("document_sources") or state.get("web_sources") or state.get("used_chat_ids"))
    confidence = _bounded_confidence(parsed.get("confidence"))
    missing_evidence = _bounded_string_list(parsed.get("missing_evidence"))
    recommended_next_steps = _bounded_string_list(parsed.get("recommended_next_steps"))
    report = {
        "sufficient": sufficient,
        "confidence": confidence,
        "missing_evidence": missing_evidence,
        "citation_risk": _risk_level(parsed.get("citation_risk")),
        "contradiction_risk": _risk_level(parsed.get("contradiction_risk")),
        "recommended_next_steps": recommended_next_steps,
        "reason": compact_preview(str(parsed.get("reason") or ""), limit=500),
    }
    return report


def _normalize_replanner_execution_plan(
    parsed: Dict[str, Any],
    *,
    use_web_search: bool,
    allowed_tool_ids: Any,
) -> Dict[str, Any]:
    normalization_notes: List[str] = []
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
    allowed_ids = set(allowed_tool_ids if isinstance(allowed_tool_ids, list) else [])
    if "live_web_recon" not in allowed_ids and "web_worker" in steps:
        steps = [step for step in steps if step != "web_worker"]
        normalization_notes.append("web_worker_removed_when_tool_disallowed")
    return {
        "execution_plan": _ordered_plan_steps(steps),
        "reason": str(parsed.get("reason") or parsed.get("route_reason") or ""),
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
            "evidence_evaluator": self.evidence_evaluator,
            "replanner": self.replanner,
            "direct_answer": self.direct_answer,
            "synthesizer": self.synthesizer,
            "finalizer": self.finalizer,
            "hitl_gate": self.hitl_gate,
        }

    def get(self, node_type: str) -> Callable[..., Any]:
        if node_type not in self._nodes:
            raise ValueError(f"Unknown node type: {node_type}")
        return self._nodes[node_type]

    def get_for_spec(self, node_spec: Dict[str, Any]) -> Callable[..., Any]:
        node_type = str(node_spec.get("type") or "")
        node_id = str(node_spec.get("id") or node_type)
        metadata = get_node_type_metadata(node_type)
        capabilities = list(metadata.get("capabilities") or node_type_capabilities(node_type))
        node_impl = self.get(node_type)

        async def _bound_node(state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
            visit_index = _node_visit_counts(state).get(node_id, 0) + 1
            _check_visit_budget(state, node_id=node_id, node_type=node_type, visit_index=visit_index)
            runtime_config = _with_node_runtime_config(
                config,
                node_id=node_id,
                node_type=node_type,
                capabilities=capabilities,
                visit_index=visit_index,
            )
            if node_type == "hitl_gate":
                update = await self.hitl_gate(state, runtime_config, node_id=node_id)
            else:
                update = await node_impl(state, runtime_config)
            return _with_visit_accounting(
                update,
                state,
                node_id=node_id,
                node_type=node_type,
                visit_index=visit_index,
            )

        return _bound_node

    async def context_loader(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            bundle = await prefetch_context(
                thread_id=state["thread_id"],
                raw_question=state["question"],
                embed_model_name=state["embedding_model"],
                context_window=state.get("context_window", DEFAULT_TOKEN_BUDGET),
                use_web_search=state.get("use_web_search", False),
                use_reranker=state.get("use_reranker", True),
            )
        except Exception as exc:
            _append_failed_node_event(state, config, "context_loader", started, exc)
            raise
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
        prompt_details = prompt_summary(
            "Planner Node Prompt",
            "You are a strict planner for a scoped RAG workflow.",
            prompt,
        )
        response = await _invoke_llm_for_node(
            llm.ainvoke,
            [
                SystemMessage(content="You are a strict planner for a scoped RAG workflow."),
                HumanMessage(content=prompt),
            ],
            state=state,
            config=config,
            node="planner",
            started=started,
            retry_observer=retry_observer,
            retry_attempts=retry_attempts,
            model_name=state.get("llm_model"),
            failure_data={
                "input_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
                "input_preview": {
                    "question": compact_preview(state.get("question")),
                    "prefetch": compact_preview(_format_prefetch_summary(state.get("pre_fetch_bundle") or {})),
                },
                "prompt_summary": prompt_details,
            },
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
            "prompt_summary": prompt_details,
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
        prompt_details = prompt_summary(
            "Router Node Prompt",
            "You are a strict router for a RAG workflow.",
            prompt,
        )
        response = await _invoke_llm_for_node(
            llm.ainvoke,
            [
                SystemMessage(content="You are a strict router for a RAG workflow."),
                HumanMessage(content=prompt),
            ],
            state=state,
            config=config,
            node="router",
            started=started,
            retry_observer=retry_observer,
            retry_attempts=retry_attempts,
            model_name=state.get("llm_model"),
            failure_data={
                "input_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
                "input_preview": {
                    "question": compact_preview(state.get("question")),
                    "prefetch": compact_preview(_format_prefetch_summary(state.get("pre_fetch_bundle") or {})),
                },
                "prompt_summary": prompt_details,
            },
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
            "prompt_summary": prompt_details,
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
        tool_input = {"query": state["question"], "max_results": 10}
        tool_config = _tool_config_for_node(
            state,
            config,
            caller_node="retrieval_worker",
            tool_name=tool_name,
            started=started,
        )
        raw = await _invoke_tool_for_node(
            search_documents,
            tool_input,
            state=state,
            config=tool_config,
            node="retrieval_worker",
            started=started,
        )
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        document_sources = [*state.get("document_sources", []), *artifacts.get("document_sources", [])]
        web_sources = [*state.get("web_sources", []), *artifacts.get("web_sources", [])]
        evidence = _combine_evidence(
            state.get("evidence"),
            payload.get("content", ""),
            label="Document evidence",
            limit=_evidence_text_limit(state),
        )
        evidence_packets = _append_evidence_packet(
            state,
            config,
            kind="document",
            content=payload.get("content", ""),
            refs=refs_from_artifacts(artifacts),
        )
        data = {
            "status": "completed" if payload.get("ok", True) else "failed",
            "warnings": normalize_warnings(payload.get("warnings")),
            "input_refs": _state_evidence_refs(state),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "previous_evidence": compact_preview(state.get("evidence")),
            },
            "evidence_chars": len(str(evidence or "")),
            "evidence_packet_count": len(evidence_packets),
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
            "evidence_packets": evidence_packets,
            "node_events": _append_event(state, "retrieval_worker", data, started=started, config=config),
            "tool_events": _append_tool_event(state, payload, tool_input=tool_input, config=config),
        }

    async def memory_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        if _should_skip_worker(state, "memory_worker"):
            return _skipped_worker_update(state, config, "memory_worker", started, "not_selected_by_plan")
        tool_name = "search_conversation_history"
        tool_input = {"query": state["question"], "max_results": 10}
        tool_config = _tool_config_for_node(
            state,
            config,
            caller_node="memory_worker",
            tool_name=tool_name,
            started=started,
        )
        raw = await _invoke_tool_for_node(
            search_conversation_history,
            tool_input,
            state=state,
            config=tool_config,
            node="memory_worker",
            started=started,
        )
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        evidence = _combine_evidence(
            state.get("evidence"),
            payload.get("content", ""),
            label="Memory evidence",
            limit=_evidence_text_limit(state),
        )
        evidence_packets = _append_evidence_packet(
            state,
            config,
            kind="memory",
            content=payload.get("content", ""),
            refs=refs_from_artifacts(artifacts),
        )
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
            "evidence_packet_count": len(evidence_packets),
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
            "evidence_packets": evidence_packets,
            "used_chat_ids": used_chat_ids,
            "node_events": _append_event(state, "memory_worker", data, started=started, config=config),
            "tool_events": _append_tool_event(state, payload, tool_input=tool_input, config=config),
        }

    async def timeline_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        if _should_skip_worker(state, "timeline_worker"):
            return _skipped_worker_update(state, config, "timeline_worker", started, "not_selected_by_plan")
        tool_name = "search_thread_timeline"
        tool_input = {"query": state["question"], "sources": "all", "order": "relevance", "max_results": 10}
        tool_config = _tool_config_for_node(
            state,
            config,
            caller_node="timeline_worker",
            tool_name=tool_name,
            started=started,
        )
        raw = await _invoke_tool_for_node(
            search_thread_timeline,
            tool_input,
            state=state,
            config=tool_config,
            node="timeline_worker",
            started=started,
        )
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        evidence = _combine_evidence(
            state.get("evidence"),
            payload.get("content", ""),
            label="Timeline evidence",
            limit=_evidence_text_limit(state),
        )
        evidence_packets = _append_evidence_packet(
            state,
            config,
            kind="timeline",
            content=payload.get("content", ""),
            refs=refs_from_artifacts(artifacts),
        )
        data = {
            "status": "completed" if payload.get("ok", True) else "failed",
            "warnings": normalize_warnings(payload.get("warnings")),
            "input_refs": _state_evidence_refs(state),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "previous_evidence": compact_preview(state.get("evidence")),
            },
            "evidence_chars": len(str(evidence or "")),
            "evidence_packet_count": len(evidence_packets),
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
            "evidence_packets": evidence_packets,
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
        tool_config = _tool_config_for_node(
            state,
            config,
            caller_node="web_worker",
            tool_name=tool_name,
            started=started,
        )
        tool_input = state["question"]
        raw = await _invoke_tool_for_node(
            search_web,
            tool_input,
            state=state,
            config=tool_config,
            node="web_worker",
            started=started,
        )
        payload = normalize_tool_result(raw, tool_name=tool_name, config=tool_config)
        artifacts = payload.get("artifacts") or {}
        web_sources = [*state.get("web_sources", []), *artifacts.get("web_sources", [])]
        evidence = _combine_evidence(
            state.get("evidence"),
            payload.get("content", ""),
            label="Web evidence",
            limit=_evidence_text_limit(state),
        )
        evidence_packets = _append_evidence_packet(
            state,
            config,
            kind="web",
            content=payload.get("content", ""),
            refs=refs_from_artifacts(artifacts),
        )
        data = {
            "status": "completed" if payload.get("ok", True) else "failed",
            "warnings": normalize_warnings(payload.get("warnings")),
            "input_refs": _state_evidence_refs(state),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "previous_evidence": compact_preview(state.get("evidence")),
            },
            "evidence_chars": len(str(evidence or "")),
            "evidence_packet_count": len(evidence_packets),
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
            "evidence_packets": evidence_packets,
            "web_sources": web_sources,
            "node_events": _append_event(state, "web_worker", data, started=started, config=config),
            "tool_events": _append_tool_event(state, payload, tool_input=tool_input, config=config),
        }

    async def evidence_evaluator(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        prompt = build_evaluator_prompt(state)
        retry_attempts, retry_observer = _llm_retry_observer()
        prompt_details = prompt_summary(
            "Evidence Evaluator Prompt",
            "You are a strict evidence evaluator for a bounded RAG workflow.",
            prompt,
        )
        response = await _invoke_llm_for_node(
            llm.ainvoke,
            [
                SystemMessage(content="You are a strict evidence evaluator for a bounded RAG workflow."),
                HumanMessage(content=prompt),
            ],
            state=state,
            config=config,
            node="evidence_evaluator",
            started=started,
            retry_observer=retry_observer,
            retry_attempts=retry_attempts,
            model_name=state.get("llm_model"),
            failure_data={
                "input_refs": _state_evidence_refs(state),
                "input_preview": {
                    "question": compact_preview(state.get("question")),
                    "execution_plan": state.get("execution_plan"),
                    "evidence": compact_preview(state.get("evidence")),
                },
                "prompt_summary": prompt_details,
            },
        )
        parsed = _safe_json_object(str(getattr(response, "content", "") or ""))
        report = normalize_evaluator_report(parsed, state)
        replan_count = _current_replan_count(state)
        replans = _replan_budget(state)
        if report["sufficient"]:
            next_route = "answer"
            event_name = "evaluation.completed"
        elif replan_count < replans:
            next_route = "replan"
            event_name = "replan.requested"
        else:
            next_route = "answer_budget_exhausted"
            event_name = "replan.budget_exhausted"

        evidence_update = state.get("evidence")
        if next_route == "answer_budget_exhausted":
            gaps = "; ".join(report.get("missing_evidence") or []) or "The evaluator found unresolved evidence gaps."
            evidence_update = _combine_evidence(
                state.get("evidence"),
                (
                    "The evidence evaluator found insufficient evidence, and the replan budget is exhausted. "
                    f"Answer only from available context and explicitly state unresolved gaps: {gaps}"
                ),
                label="Evaluator warning",
                limit=_evidence_text_limit(state),
            )

        data = {
            "status": "completed",
            "route": state.get("route"),
            "route_reason": state.get("route_reason"),
            "evaluator_route": next_route,
            "evaluator_report": report,
            "evaluation_confidence": report["confidence"],
            "evidence_gaps": report["missing_evidence"],
            "replan_count": replan_count,
            "replans": replans,
            "event_name": event_name,
            "input_refs": _state_evidence_refs(state),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "execution_plan": state.get("execution_plan"),
                "evidence": compact_preview(state.get("evidence")),
            },
            "prompt_summary": prompt_details,
            "llm_result_summary": {
                "parsed": bool(parsed),
                "evaluator_route": next_route,
                "evaluator_report": report,
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    retry_attempts=retry_attempts,
                ),
            },
            "output_refs": _state_evidence_refs({**state, "evidence": evidence_update}),
            "output_preview": {
                "evaluator_route": next_route,
                "evaluator_report": report,
            },
        }
        _log_node_end(state, "evidence_evaluator", started, data)
        return {
            "evaluator_route": next_route,
            "evaluator_report": report,
            "evidence_gaps": report["missing_evidence"],
            "evaluation_confidence": report["confidence"],
            "evidence": evidence_update,
            "node_events": _append_event(state, "evidence_evaluator", data, started=started, config=config),
        }

    async def replanner(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        prompt = build_replanner_prompt(state)
        retry_attempts, retry_observer = _llm_retry_observer()
        prompt_details = prompt_summary(
            "Replanner Prompt",
            "You are a strict replanner for a bounded RAG workflow.",
            prompt,
        )
        response = await _invoke_llm_for_node(
            llm.ainvoke,
            [
                SystemMessage(content="You are a strict replanner for a bounded RAG workflow."),
                HumanMessage(content=prompt),
            ],
            state=state,
            config=config,
            node="replanner",
            started=started,
            retry_observer=retry_observer,
            retry_attempts=retry_attempts,
            model_name=state.get("llm_model"),
            failure_data={
                "input_refs": _state_evidence_refs(state),
                "input_preview": {
                    "question": compact_preview(state.get("question")),
                    "current_execution_plan": state.get("execution_plan"),
                    "evaluator_report": state.get("evaluator_report"),
                },
                "prompt_summary": prompt_details,
            },
        )
        parsed = _safe_json_object(str(getattr(response, "content", "") or ""))
        normalized = _normalize_replanner_execution_plan(
            parsed,
            use_web_search=bool(state.get("use_web_search", False)),
            allowed_tool_ids=state.get("allowed_tool_ids"),
        )
        replan_count = _current_replan_count(state) + 1
        history_item = {
            "replan_count": replan_count,
            "reason": compact_preview(normalized["reason"], limit=500),
            "execution_plan": normalized["execution_plan"],
            "evaluator_report": state.get("evaluator_report") or {},
        }
        replan_history = [
            *(state.get("replan_history") if isinstance(state.get("replan_history"), list) else []),
            history_item,
        ][-5:]
        data = {
            "status": "completed",
            "route": state.get("route"),
            "route_reason": state.get("route_reason"),
            "execution_plan": normalized["execution_plan"],
            "replan_count": replan_count,
            "replan_reason": normalized["reason"],
            "event_name": "replan.requested" if normalized["execution_plan"] else "replan.skipped",
            "input_refs": _state_evidence_refs(state),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "current_execution_plan": state.get("execution_plan"),
                "evaluator_report": state.get("evaluator_report"),
            },
            "prompt_summary": prompt_details,
            "llm_result_summary": {
                "parsed": bool(parsed),
                "execution_plan": normalized["execution_plan"],
                "normalization_notes": normalized.get("normalization_notes") or [],
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    retry_attempts=retry_attempts,
                ),
            },
            "output_refs": _state_evidence_refs(state),
            "output_preview": {
                "execution_plan": normalized["execution_plan"],
                "replan_count": replan_count,
                "replan_reason": compact_preview(normalized["reason"]),
            },
        }
        _log_node_end(state, "replanner", started, data)
        return {
            "execution_plan": normalized["execution_plan"],
            "replan_count": replan_count,
            "replan_reason": normalized["reason"],
            "replan_history": replan_history,
            "node_events": _append_event(state, "replanner", data, started=started, config=config),
        }

    async def direct_answer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._answer_from_context(state, config, node_name="direct_answer")

    async def synthesizer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._answer_from_context(state, config, node_name="synthesizer")

    async def _answer_from_context(self, state: RouterRagState, config: RunnableConfig, *, node_name: str) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        context, context_source = _final_context_from_state(state)
        if state.get("evaluator_report"):
            context = _combine_evidence(
                context,
                json.dumps(state.get("evaluator_report") or {}, ensure_ascii=True, sort_keys=True),
                label="Evaluator report",
                limit=_evidence_text_limit(state),
            )
        messages = build_final_answer_messages(state, context)
        retry_attempts, retry_observer = _llm_retry_observer()
        prompt_details = prompt_summary(
            "Final Answer Prompt",
            messages["system"],
            messages["human"],
        )
        response = await _invoke_llm_for_node(
            llm.ainvoke,
            [
                SystemMessage(content=messages["system"]),
                HumanMessage(content=messages["human"]),
            ],
            state=state,
            config=config,
            node=node_name,
            started=started,
            retry_observer=retry_observer,
            retry_attempts=retry_attempts,
            model_name=state.get("llm_model"),
            failure_data={
                "input_refs": _state_evidence_refs(state) or _prefetch_refs(state.get("pre_fetch_bundle") or {}),
                "input_preview": {
                    "question": compact_preview(state.get("question")),
                    "context_source": context_source,
                    "context": compact_preview(context),
                },
                "prompt_summary": prompt_details,
            },
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
            "prompt_summary": prompt_details,
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

    async def hitl_gate(
        self,
        state: RouterRagState,
        config: RunnableConfig,
        *,
        node_id: str = WEB_APPROVAL_GATE_ID,
    ) -> Dict[str, Any]:
        """Pause at a reusable human-in-the-loop gate declared by hitl_policy."""

        started = time.perf_counter()
        policy = state.get("hitl_policy") if isinstance(state.get("hitl_policy"), dict) else {}
        gates = _hitl_gates_from_policy(policy)
        gate_policy = _normalize_hitl_gate_policy(node_id, gates.get(node_id))
        enabled = bool(policy.get("enabled")) and gate_policy.get("enabled", True) is not False
        if not enabled:
            routes = dict(state.get("hitl_gate_routes") or {})
            routes[node_id] = "approve"
            return _skipped_worker_update(state, config, node_id, started, "hitl_policy_disabled") | {
                "hitl_gate_route": "approve",
                "hitl_gate_routes": routes,
            }

        mode = str(gate_policy.get("mode") or "approval")
        phase = str(gate_policy.get("phase") or "before")
        target = gate_policy.get("target") if isinstance(gate_policy.get("target"), dict) else {}
        target_node_id = target.get("node_id")
        target_node_type = target.get("node_type")
        interrupt_type = str(gate_policy.get("interrupt_type") or gate_policy.get("type") or ("option_review" if mode == "choice" else "human_review"))
        allowed_actions = _normalize_hitl_actions(gate_policy)
        default_action = str(gate_policy.get("default_action") or "continue_without")
        if default_action not in allowed_actions:
            default_action = "continue_without" if "continue_without" in allowed_actions else allowed_actions[0]
        routes_by_action = gate_policy.get("routes") if isinstance(gate_policy.get("routes"), dict) else {}
        visit_index = _runtime_visit_index(config)
        interrupt_count_key = _hitl_interrupt_count_key(node_id, visit_index)

        interrupt_counts = _hitl_interrupt_counts(state)
        interrupt_limit = _hitl_interrupt_limit(policy, gate_policy)
        if interrupt_limit is not None and _hitl_visit_interrupt_count(interrupt_counts, gate_id=node_id, visit_index=visit_index) >= interrupt_limit:
            route = "continue_without" if "continue_without" in allowed_actions or "continue_without" in routes_by_action else default_action
            gate_routes = dict(state.get("hitl_gate_routes") or {})
            gate_routes[node_id] = route
            update: Dict[str, Any] = {
                "hitl_gate_route": route,
                "hitl_gate_routes": gate_routes,
                "hitl_interrupt_counts": interrupt_counts,
            }
            if route == "continue_without":
                update["evidence"] = _combine_evidence(
                    state.get("evidence"),
                    (
                        "The configured human review interrupt limit was reached. "
                        "Continue without additional gated actions unless already approved by available context."
                    ),
                    label="HITL decision",
                    limit=_evidence_text_limit(state),
                )
            return _skipped_worker_update(state, config, node_id, started, "hitl_interrupt_limit_exhausted") | update

        options = gate_policy.get("options") if isinstance(gate_policy.get("options"), list) else []
        option_ids = _hitl_option_ids(gate_policy)
        input_summary = {
            "question": compact_preview(state.get("question")),
            "route": state.get("route"),
            "route_reason": compact_preview(state.get("route_reason")),
            "document_source_count": len(state.get("document_sources") or []),
            "web_source_count": len(state.get("web_sources") or []),
            "used_chat_id_count": len(state.get("used_chat_ids") or []),
            "evidence": compact_preview(state.get("evidence")),
        }
        proposed_tool = None
        if target_node_id == "web_worker" or node_id == WEB_APPROVAL_GATE_ID:
            proposed_tool = {
                "name": "search_web",
                "caller_node": "web_worker",
                "input": compact_preview(state.get("question"), limit=1000),
            }

        decision = interrupt(
            {
                "gate_id": node_id,
                "node_id": node_id,
                "target_node_id": target_node_id,
                "target_node_type": target_node_type,
                "visit_index": visit_index,
                "interrupt_count_key": interrupt_count_key,
                "phase": phase,
                "mode": mode,
                "type": interrupt_type,
                "title": gate_policy.get("title") or ("Choose approved options" if mode == "choice" else "Human review requested"),
                "prompt": gate_policy.get("prompt")
                or gate_policy.get("body")
                or ("Select which options may run." if mode == "choice" else "Review this step before the graph continues."),
                "allowed_actions": allowed_actions,
                "default_action": default_action,
                "selection_mode": gate_policy.get("selection_mode") if mode == "choice" else None,
                "options": options if mode == "choice" else None,
                "checkpoint_resume": True,
                "reject_behavior": "resume" if "reject" in dict(gate_policy.get("routes") or {}) else gate_policy.get("reject_behavior"),
                "input_summary": input_summary,
                "proposed_tool": proposed_tool,
                "proposed_final_answer": compact_preview(state.get("final_answer"), limit=2000) if mode == "review" else None,
                "editable_fields": gate_policy.get("editable_fields") if mode == "review" else None,
            }
        )
        decision = decision if isinstance(decision, dict) else {"action": str(decision or default_action)}
        action = str(decision.get("action") or default_action)
        if action not in allowed_actions:
            action = default_action
        selected_option_ids = _hitl_selected_option_ids(decision, gate_policy) if mode == "choice" else []
        if action == "approve_selected" and not selected_option_ids and option_ids:
            selected_option_ids = [option_ids[0]]

        if mode == "choice" and action == "approve_selected":
            route: Any = selected_option_ids[0] if selected_option_ids else "continue_without"
        elif action == "approve":
            route = "approve"
        elif action in routes_by_action:
            route = action
        else:
            route = "continue_without" if action in {"continue_without", "reject"} else action

        gate_routes = dict(state.get("hitl_gate_routes") or {})
        gate_routes[node_id] = route
        selected_options_by_gate = dict(state.get("hitl_selected_options") or {})
        if selected_option_ids:
            selected_options_by_gate[node_id] = selected_option_ids

        selected_targets = _hitl_option_targets(gate_policy, selected_option_ids)
        execution_plan = state.get("execution_plan")
        execution_plan_update = None
        if selected_targets and all(target in PLAN_EXECUTE_WORKER_NODES for target in selected_targets):
            execution_plan_update = [target for target in PLAN_EXECUTE_WORKER_NODES if target in selected_targets]

        update: Dict[str, Any] = {
            "hitl_gate_route": route,
            "hitl_gate_routes": gate_routes,
            "hitl_selected_options": selected_options_by_gate,
            "hitl_interrupt_counts": {
                **interrupt_counts,
                node_id: interrupt_counts.get(node_id, 0) + 1,
                interrupt_count_key: interrupt_counts.get(interrupt_count_key, 0) + 1,
            },
            "hitl_decisions": [
                *(state.get("hitl_decisions") if isinstance(state.get("hitl_decisions"), list) else []),
                {
                    "gate_id": node_id,
                    "node_id": node_id,
                    "target_node_id": target_node_id,
                    "visit_index": visit_index,
                    "interrupt_count_key": interrupt_count_key,
                    "phase": phase,
                    "mode": mode,
                    "type": interrupt_type,
                    "action": action,
                    "selected_option_ids": selected_option_ids,
                    "decision": {
                        key: value
                        for key, value in decision.items()
                        if key not in {"resume_token"}
                    },
                },
            ],
        }
        if execution_plan_update is not None:
            update["execution_plan"] = execution_plan_update
        elif isinstance(execution_plan, list):
            update["execution_plan"] = execution_plan

        if mode == "review":
            update["human_review_decision"] = {
                key: value
                for key, value in decision.items()
                if key not in {"resume_token"}
            }
            edited_payload = decision.get("edited_payload") if isinstance(decision.get("edited_payload"), dict) else {}
            edited_answer = edited_payload.get("final_answer") or edited_payload.get("answer")
            if action == "edit" and isinstance(edited_answer, str) and edited_answer.strip():
                update["final_answer"] = edited_answer.strip()

        if route == "continue_without" or action == "reject":
            update["evidence"] = _combine_evidence(
                state.get("evidence"),
                (
                    "A human reviewer chose to continue without one or more gated options. "
                    "Do not claim skipped tools, branches, or live evidence were checked; answer only from available context."
                ),
                label="HITL decision",
                limit=_evidence_text_limit(state),
            )

        data = {
            "status": "completed",
            "action": action,
            "route": state.get("route"),
            "route_reason": state.get("route_reason"),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "route": state.get("route"),
                "route_reason": compact_preview(state.get("route_reason")),
                "options": options if mode == "choice" else None,
                "proposed_final_answer": compact_preview(state.get("final_answer")) if mode == "review" else None,
            },
            "output_preview": {
                "decision": update["hitl_decisions"][-1],
                "next": routes_by_action.get(route) or (selected_targets[0] if selected_targets else route),
                "final_answer": compact_preview(update.get("final_answer") or state.get("final_answer")) if mode == "review" else None,
            },
        }
        _log_node_end(state, node_id, started, data)
        return {
            **update,
            "node_events": _append_event(state, node_id, data, started=started, config=config),
        }

def router_route(state: RouterRagState) -> str:
    return state.get("route") or "document"


def planner_route(state: RouterRagState) -> str:
    route = state.get("route")
    return route if route in {"execute", "direct", "clarify"} else "execute"


def evaluator_route(state: RouterRagState) -> str:
    route = state.get("evaluator_route")
    return route if route in {"answer", "replan", "answer_budget_exhausted"} else "answer"


def hitl_gate_route(state: RouterRagState) -> str:
    route = state.get("hitl_gate_route")
    return route if route in {"approve", "continue_without"} else "continue_without"


def hitl_gate_route_for(gate_id: str) -> Callable[[RouterRagState], Any]:
    def _route(state: RouterRagState) -> Any:
        routes = state.get("hitl_gate_routes") if isinstance(state.get("hitl_gate_routes"), dict) else {}
        route = routes.get(gate_id, state.get("hitl_gate_route"))
        return route if isinstance(route, str) and route else "continue_without"

    return _route


def _route_function_for_edge(
    edge: Dict[str, Any],
    *,
    source: str,
    node_types: Dict[str, str],
) -> Callable[[RouterRagState], Any]:
    route_fn_id = edge.get("route_fn")
    source_type = node_types.get(source)
    if isinstance(route_fn_id, str) and route_fn_id:
        if source_type and not route_function_allowed_for_node_type(route_fn_id, source_type):
            raise ValueError(f"Route function {route_fn_id} is not allowed from node type {source_type}")
        if not route_function_runtime_supported(route_fn_id):
            raise ValueError(f"Route function {route_fn_id} is not runtime-supported in V1")
        if route_fn_id == "hitl_gate_route":
            return hitl_gate_route_for(str(source))
        if route_fn_id == "planner_route":
            return planner_route
        if route_fn_id == "evaluator_route":
            return evaluator_route
        if route_fn_id == "router_route":
            return router_route
        raise ValueError(f"Unknown route function: {route_fn_id}")

    raise ValueError(f"Conditional edge from {source} must declare route_fn")


class TemplateCompiler:
    """Compile validated v2 template specs into LangGraph StateGraph instances."""

    def __init__(self, registry: Optional[NodeRegistry] = None):
        self.registry = registry or NodeRegistry()

    def compile(
        self,
        spec: Dict[str, Any],
        *,
        checkpointer: Any = None,
    ):
        from app.agent_patterns.validator import TemplateValidator

        graph_spec = ((spec.get("config") or {}).get("graph") or {}) if isinstance(spec, dict) else {}
        if not graph_spec.get("hitl_compiled"):
            TemplateValidator().validate(spec)
            spec = self.materialize_spec(spec)
            graph_spec = (spec.get("config") or {}).get("graph") or {}
        workflow = StateGraph(RouterRagState)
        node_types: Dict[str, str] = {}
        for node in graph_spec.get("nodes", []):
            node_types[node["id"]] = node["type"]
            workflow.add_node(node["id"], self.registry.get_for_spec(node))

        for edge in graph_spec.get("edges", []):
            source = edge.get("from")
            target = edge.get("to")
            if edge.get("conditional"):
                route_fn = _route_function_for_edge(
                    edge,
                    source=str(source),
                    node_types=node_types,
                )
                routes = {
                    key: END if value == "END" else value
                    for key, value in dict(edge["routes"]).items()
                }
                workflow.add_conditional_edges(source, route_fn, routes)
                continue
            source_ref = START if source == "START" else source
            target_ref = END if target == "END" else target
            workflow.add_edge(source_ref, target_ref)

        return workflow.compile(checkpointer=checkpointer)

    def materialize_spec(
        self,
        spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        materialized = deepcopy(spec)
        config = materialized.get("config") if isinstance(materialized.get("config"), dict) else {}
        graph_spec = config.get("graph") if isinstance(config.get("graph"), dict) else {}
        hitl_policy = config.get("hitl_policy") if isinstance(config.get("hitl_policy"), dict) else {}
        explicit_graph = self._with_explicit_route_functions(graph_spec)
        compiled_graph = self._with_hitl_policy_gates(
            explicit_graph,
            hitl_policy=hitl_policy,
        )
        config["graph"] = self._with_catalog_node_metadata(compiled_graph)
        config["loop_policy"] = self._with_materialized_loop_policy(
            config.get("loop_policy"),
            graph_spec=config["graph"],
        )
        materialized["config"] = config
        return materialized

    def _with_materialized_loop_policy(self, loop_policy: Any, *, graph_spec: Dict[str, Any]) -> Dict[str, Any]:
        policy = dict(loop_policy) if isinstance(loop_policy, dict) else {}
        nodes = [node for node in graph_spec.get("nodes", []) if isinstance(node, dict)]
        node_count = len(nodes)
        try:
            max_total_visits = int(policy.get("max_total_visits", 0))
        except (TypeError, ValueError):
            max_total_visits = 0
        if node_count and max_total_visits < node_count:
            policy["max_total_visits"] = node_count
        node_visit_limits = policy.get("node_visit_limits")
        if isinstance(node_visit_limits, dict):
            policy["node_visit_limits"] = dict(node_visit_limits)
        elif node_visit_limits is not None:
            policy["node_visit_limits"] = {}
        return policy

    def _with_explicit_route_functions(self, graph_spec: Dict[str, Any]) -> Dict[str, Any]:
        nodes = [dict(node) for node in graph_spec.get("nodes", []) if isinstance(node, dict)]
        node_types = {
            str(node.get("id")): str(node.get("type"))
            for node in nodes
            if isinstance(node.get("id"), str) and isinstance(node.get("type"), str)
        }
        route_by_type = {
            "router": "router_route",
            "planner": "planner_route",
            "evidence_evaluator": "evaluator_route",
            "hitl_gate": "hitl_gate_route",
        }
        edges = []
        for raw_edge in graph_spec.get("edges", []):
            if not isinstance(raw_edge, dict):
                continue
            edge = dict(raw_edge)
            if edge.get("conditional") and not edge.get("route_fn"):
                route_fn = route_by_type.get(node_types.get(str(edge.get("from")), ""))
                if route_fn:
                    edge["route_fn"] = route_fn
            edges.append(edge)
        return {**graph_spec, "nodes": nodes, "edges": edges}

    def _with_catalog_node_metadata(self, graph_spec: Dict[str, Any]) -> Dict[str, Any]:
        nodes = []
        for raw_node in graph_spec.get("nodes", []):
            if not isinstance(raw_node, dict):
                continue
            node = dict(raw_node)
            node_type = node.get("type")
            metadata = get_node_type_metadata(str(node_type)) if isinstance(node_type, str) else {}
            display_name = metadata.get("display_name")
            category = metadata.get("category")
            if isinstance(display_name, str) and display_name:
                node["label"] = display_name
            if isinstance(category, str) and category:
                node["category"] = category
            for key in (
                "capabilities",
                "allowed_route_functions",
                "allowed_tool_contract_ids",
                "state_reads",
                "state_writes",
                "prompt_slots",
                "context_policy",
                "observability",
                "max_instances",
            ):
                if key in metadata:
                    node[key] = deepcopy(metadata[key])
            nodes.append(node)
        return {**graph_spec, "nodes": nodes}

    def _with_hitl_policy_gates(self, graph_spec: Dict[str, Any], *, hitl_policy: Dict[str, Any]) -> Dict[str, Any]:
        nodes = [dict(node) for node in graph_spec.get("nodes", []) if isinstance(node, dict)]
        edges = [dict(edge) for edge in graph_spec.get("edges", []) if isinstance(edge, dict)]
        if graph_spec.get("hitl_compiled") or not bool(hitl_policy.get("enabled")):
            return {**graph_spec, "nodes": nodes, "edges": edges}

        node_types = {
            str(node.get("id")): str(node.get("type"))
            for node in nodes
            if isinstance(node.get("id"), str) and isinstance(node.get("type"), str)
        }
        existing_node_ids = set(node_types)
        gates = _hitl_gates_from_policy(hitl_policy)
        for gate_id, raw_gate in gates.items():
            if not isinstance(gate_id, str) or gate_id in existing_node_ids:
                continue
            gate = _normalize_hitl_gate_policy(gate_id, raw_gate)
            if gate.get("enabled", True) is False:
                continue
            phase = str(gate.get("phase") or "before")
            if phase == "inside_tool":
                continue
            target_node_id = self._resolve_hitl_target_node_id(gate, node_types)
            if not target_node_id:
                continue

            nodes.append({"id": gate_id, "type": "hitl_gate"})
            existing_node_ids.add(gate_id)
            routes = self._hitl_gate_routes(gate, target_node_id, edges, phase=phase)
            if phase == "before":
                edges = self._insert_before_gate(edges, gate_id, target_node_id)
            elif phase == "after":
                edges = self._insert_after_gate(edges, gate_id, target_node_id)
            else:
                continue
            edges.append({"from": gate_id, "conditional": True, "route_fn": "hitl_gate_route", "routes": routes})

        return {"nodes": nodes, "edges": edges, "hitl_compiled": True}

    def _resolve_hitl_target_node_id(self, gate: Dict[str, Any], node_types: Dict[str, str]) -> Optional[str]:
        target = gate.get("target") if isinstance(gate.get("target"), dict) else {}
        node_id = target.get("node_id")
        if isinstance(node_id, str) and node_id in node_types:
            return node_id
        node_type = target.get("node_type")
        if isinstance(node_type, str):
            matches = [candidate for candidate, candidate_type in node_types.items() if candidate_type == node_type]
            if len(matches) == 1:
                return matches[0]
        return None

    def _default_bypass_target(self, target_node_id: str, edges: List[Dict[str, Any]]) -> str:
        for edge in edges:
            if edge.get("from") == target_node_id and isinstance(edge.get("to"), str):
                return str(edge["to"])
        return "END"

    def _hitl_gate_routes(self, gate: Dict[str, Any], target_node_id: str, edges: List[Dict[str, Any]], *, phase: str) -> Dict[str, str]:
        configured = dict(gate.get("routes") or {})
        mode = str(gate.get("mode") or "approval")
        bypass = self._default_bypass_target(target_node_id, edges)
        routes: Dict[str, str] = {}
        if mode == "choice":
            options = gate.get("options") if isinstance(gate.get("options"), list) else []
            for option in options:
                if not isinstance(option, dict):
                    continue
                option_id = option.get("id")
                option_target = option.get("target_node_id")
                if isinstance(option_id, str) and isinstance(option_target, str):
                    routes[option_id] = option_target
            routes["continue_without"] = configured.get("continue_without") or bypass
            if "reject" in configured:
                routes["reject"] = configured["reject"]
            return routes
        routes["approve"] = configured.get("approve") or (target_node_id if phase == "before" else bypass)
        routes["continue_without"] = configured.get("continue_without") or bypass
        if "reject" in configured:
            routes["reject"] = configured["reject"]
        if "edit" in configured:
            routes["edit"] = configured["edit"]
        return routes

    def _insert_before_gate(self, edges: List[Dict[str, Any]], gate_id: str, target_node_id: str) -> List[Dict[str, Any]]:
        updated: List[Dict[str, Any]] = []
        for edge in edges:
            edge = dict(edge)
            if edge.get("conditional") and isinstance(edge.get("routes"), dict):
                routes = dict(edge["routes"])
                changed = False
                for route, target in list(routes.items()):
                    if target == target_node_id:
                        routes[route] = gate_id
                        changed = True
                if changed:
                    edge["routes"] = routes
                updated.append(edge)
                continue
            if edge.get("to") == target_node_id:
                edge["to"] = gate_id
            updated.append(edge)
        return updated

    def _insert_after_gate(self, edges: List[Dict[str, Any]], gate_id: str, target_node_id: str) -> List[Dict[str, Any]]:
        updated: List[Dict[str, Any]] = []
        for edge in edges:
            edge = dict(edge)
            if edge.get("from") == target_node_id:
                edge["from"] = gate_id
            updated.append(edge)
        updated.append({"from": target_node_id, "to": gate_id})
        return updated
