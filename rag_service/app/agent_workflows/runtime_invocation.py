from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from app.agent.tool_registry import get_tool_contract_id, validate_tool_call_allowed
from app.agent_workflows.enums import NodeEventStatus
from app.agent_workflows.events import append_node_event, append_tool_event
from app.agent_workflows.evidence import state_evidence_refs
from app.agent_workflows.node_catalog import node_type_capabilities
from app.agent_workflows.state import (
    RouterRagState,
    runtime_node_capabilities,
    runtime_node_id,
    runtime_node_type,
    runtime_visit_index,
)
from app.agent_workflows.trace import compact_preview
from app.models.retry import invoke_with_retry
from app.time_utils import iso_utc_z, utc_now


logger = logging.getLogger(__name__)


def append_event(
    state: RouterRagState,
    node: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    started: Optional[float] = None,
    config: Optional[RunnableConfig] = None,
) -> List[Dict[str, Any]]:
    return append_node_event(
        state,
        node,
        data,
        started=started,
        config=config,
        runtime_node_id=runtime_node_id,
        runtime_node_type=runtime_node_type,
        runtime_visit_index=runtime_visit_index,
        utc_now=utc_now,
        iso_utc_z=iso_utc_z,
    )


def append_tool_event_for_node(
    state: RouterRagState,
    payload: Dict[str, Any],
    *,
    tool_input: Any = None,
    config: Optional[RunnableConfig] = None,
) -> List[Dict[str, Any]]:
    return append_tool_event(
        state,
        payload,
        tool_input=tool_input,
        config=config,
        runtime_node_type=runtime_node_type,
        runtime_visit_index=runtime_visit_index,
    )


def error_summary(exc: Exception, *, code: str) -> Dict[str, Any]:
    return {
        "code": code,
        "type": type(exc).__name__,
        "message": compact_preview(str(exc), limit=700),
        "raw_message": compact_preview(str(exc), limit=700),
        "retryable": True,
    }


def append_failed_node_event(
    state: RouterRagState,
    config: RunnableConfig,
    node: str,
    started: float,
    exc: Exception,
    *,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "status": NodeEventStatus.FAILED.value,
        "error": error_summary(exc, code=f"{node}_failed"),
        "input_preview": {"question": compact_preview(state.get("question"))},
        **(data or {}),
    }
    append_event(state, node, payload, started=started, config=config)


async def invoke_llm_for_node(
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
        response = await invoke_with_retry(func, messages, retry_observer=retry_observer)
        trace_recorder = ((config or {}).get("configurable") or {}).get("trace_recorder")
        if trace_recorder is not None and hasattr(trace_recorder, "record_llm_detail"):
            trace_recorder.record_llm_detail(
                node_id=runtime_node_id(config, node),
                node_type=runtime_node_type(config, node),
                visit_index=runtime_visit_index(config) or 1,
                messages=messages,
                response=response,
            )
        return response
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
        append_failed_node_event(
            state,
            config,
            node,
            started,
            exc,
            data={**(failure_data or {}), **llm_failure},
        )
        raise


async def invoke_tool_for_node(
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
        append_failed_node_event(
            state,
            config,
            node,
            started,
            exc,
            data={"input_preview": {"tool_input": tool_input}},
        )
        raise


def tool_config(state: RouterRagState, config: RunnableConfig, *, caller_node: str, tool_name: str) -> RunnableConfig:
    caller_node_id = runtime_node_id(config, caller_node)
    caller_node_type = runtime_node_type(config, caller_node)
    caller_capabilities = runtime_node_capabilities(config) or node_type_capabilities(caller_node_type)
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
    if caller_node_type == "durable_memory_worker":
        bundle = state.get("pre_fetch_bundle") or {}
        configurable.update({
            "prefetched_durable_memories": bundle.get("durable_memories") or [],
            "prefetched_durable_memory_scopes": bundle.get("durable_memory_scopes") or [],
            "prefetched_durable_memory_scope_policy": bundle.get("durable_memory_scope_policy") or {},
            "prefetched_durable_memory_debug": bundle.get("durable_memory_retrieval_debug") or {},
            "prefetched_durable_memory_query_vector": bundle.get("_shared_query_vector"),
        })
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


def tool_config_for_node(
    state: RouterRagState,
    config: RunnableConfig,
    *,
    caller_node: str,
    tool_name: str,
    started: float,
) -> RunnableConfig:
    try:
        return tool_config(state, config, caller_node=caller_node, tool_name=tool_name)
    except Exception as exc:
        append_failed_node_event(
            state,
            config,
            caller_node,
            started,
            exc,
            data={"input_preview": {"tool_name": tool_name}},
        )
        raise


def log_node_end(
    state: RouterRagState,
    node: str,
    started: float,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    payload = data or {}
    logger.info(
        "Agent workflow node completed | run_id=%s thread_id=%s node=%s elapsed_ms=%.1f route=%s evidence_chars=%s document_sources=%s web_sources=%s used_chat_ids=%s",
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


def safe_json_object(raw: str) -> Dict[str, Any]:
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


def first_int(*values: Any) -> Optional[int]:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def llm_result_metadata(
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
        "prompt": first_int(usage.get("input_tokens"), token_usage.get("prompt_tokens")),
        "completion": first_int(usage.get("output_tokens"), token_usage.get("completion_tokens")),
        "total": first_int(usage.get("total_tokens"), token_usage.get("total_tokens")),
        "reasoning": first_int(
            usage.get("reasoning_tokens"),
            token_usage.get("reasoning_tokens"),
            output_tokens_details.get("reasoning_tokens"),
        ),
        "cached": first_int(
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


def llm_retry_observer() -> tuple[List[Dict[str, Any]], Callable[[Dict[str, Any]], None]]:
    attempts: List[Dict[str, Any]] = []

    def observe(event: Dict[str, Any]) -> None:
        attempts.append(dict(event))

    return attempts, observe


def should_skip_worker(state: RouterRagState, worker_node: str) -> bool:
    plan = state.get("execution_plan")
    if not isinstance(plan, list):
        return False
    return worker_node not in plan


def skipped_worker_update(
    state: RouterRagState,
    config: RunnableConfig,
    worker_node: str,
    started: float,
    reason: str,
) -> Dict[str, Any]:
    data = {
        "status": NodeEventStatus.SKIPPED.value,
        "skipped": True,
        "skip_reason": reason,
        "input_preview": {"question": compact_preview(state.get("question"))},
        "input_refs": state_evidence_refs(state),
        "output_refs": state_evidence_refs(state),
    }
    log_node_end(state, worker_node, started, data)
    return {"node_events": append_event(state, worker_node, data, started=started, config=config)}
