from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agent.reasoning import normalize_ai_response
from app.agent_workflows.evidence import (
    combine_evidence,
    evidence_text_limit,
    final_context_from_state,
    prefetch_refs,
    state_evidence_refs,
)
from app.agent_workflows.enums import NodeEventStatus, WorkflowNodeType
from app.agent_workflows.prompting import build_final_answer_messages
from app.agent_workflows.runtime_invocation import (
    append_event,
    invoke_llm_for_node,
    llm_result_metadata,
    llm_retry_observer,
    log_node_end,
)
from app.agent_workflows.state import RouterRagState
from app.agent_workflows.trace import compact_preview, prompt_summary
from app.db.enums import ReasoningFormat
from app.models.llm_server_client import get_llm as _default_get_llm


def _get_llm(model_name: str) -> Any:
    graph_module = sys.modules.get("app.agent_workflows.graph")
    get_llm_fn = getattr(graph_module, "get_llm", _default_get_llm)
    return get_llm_fn(model_name)


async def direct_answer_node(state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
    return await answer_from_context_node(state, config, node_name=WorkflowNodeType.DIRECT_ANSWER.value)


async def synthesizer_node(state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
    return await answer_from_context_node(state, config, node_name=WorkflowNodeType.SYNTHESIZER.value)


async def answer_from_context_node(state: RouterRagState, config: RunnableConfig, *, node_name: str) -> Dict[str, Any]:
    started = time.perf_counter()
    llm = _get_llm(state["llm_model"])
    context, context_source = final_context_from_state(state)
    if state.get("evaluator_report"):
        context = combine_evidence(
            context,
            json.dumps(state.get("evaluator_report") or {}, ensure_ascii=True, sort_keys=True),
            label="Evaluator report",
            limit=evidence_text_limit(state),
        )
    parallel_summary = state.get("parallel_summary") if isinstance(state.get("parallel_summary"), dict) else {}
    if parallel_summary.get("partial_evidence"):
        coverage = {
            key: parallel_summary.get(key)
            for key in ("planned", "completed", "skipped", "failed", "timed_out")
        }
        context = combine_evidence(
            context,
            json.dumps(coverage, ensure_ascii=True, sort_keys=True),
            label="Retrieval coverage; answer cautiously where evidence is incomplete",
            limit=evidence_text_limit(state),
        )
    messages = build_final_answer_messages(state, context)
    retry_attempts, retry_observer = llm_retry_observer()
    prompt_details = prompt_summary(
        "Final Answer Prompt",
        messages["system"],
        messages["human"],
    )
    response = await invoke_llm_for_node(
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
            "input_refs": state_evidence_refs(state) or prefetch_refs(state.get("pre_fetch_bundle") or {}),
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
        "status": NodeEventStatus.COMPLETED.value,
        "input_refs": state_evidence_refs(state) or prefetch_refs(state.get("pre_fetch_bundle") or {}),
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
            "llm": llm_result_metadata(
                response,
                model_name=state.get("llm_model"),
                normalized_response=normalized,
                retry_attempts=retry_attempts,
            ),
        },
        "answer_chars": len(normalized["answer"] or ""),
        "evidence_chars": len(str(context or "")),
        "output_refs": state_evidence_refs(state) or prefetch_refs(state.get("pre_fetch_bundle") or {}),
        "output_preview": {"answer": compact_preview(normalized["answer"])},
    }
    log_node_end(state, node_name, started, data)
    return {
        "final_answer": normalized["answer"],
        "reasoning": normalized["reasoning"],
        "reasoning_available": normalized["reasoning_available"],
        "reasoning_format": normalized["reasoning_format"],
        "node_events": append_event(state, node_name, data, started=started, config=config),
    }


async def finalizer_node(state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    if state.get("clarification_options") and not state.get("final_answer"):
        answer = "I need a bit more clarification. Did you mean:\n" + "\n".join(
            f"- {option}" for option in state["clarification_options"]
        )
        data = {
            "status": NodeEventStatus.COMPLETED.value,
            "answer_chars": len(answer),
            "output_preview": {
                "answer": compact_preview(answer),
                "clarification_options": state.get("clarification_options"),
            },
            "llm_result_summary": {
                "clarification_option_count": len(state.get("clarification_options") or []),
            },
        }
        log_node_end(state, WorkflowNodeType.FINALIZER.value, started, data)
        return {
            "final_answer": answer,
            "reasoning": "",
            "reasoning_available": False,
            "reasoning_format": ReasoningFormat.NONE.value,
            "node_events": append_event(state, WorkflowNodeType.FINALIZER.value, data, started=started, config=config),
        }
    data = {
        "status": NodeEventStatus.COMPLETED.value,
        "answer_chars": len(state.get("final_answer") or ""),
        "output_refs": state_evidence_refs(state),
        "output_preview": {"answer": compact_preview(state.get("final_answer"))},
    }
    log_node_end(state, WorkflowNodeType.FINALIZER.value, started, data)
    return {"node_events": append_event(state, WorkflowNodeType.FINALIZER.value, data, started=started, config=config)}
