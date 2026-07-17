from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agent_workflows.enums import NodeEventStatus


@dataclass(frozen=True)
class JsonDecisionNodeSpec:
    node_name: str
    prompt_section: str
    system_message: str
    prompt: str
    failure_data: Dict[str, Any]


def build_decision_node_event_data(
    *,
    leading_fields: Dict[str, Any],
    input_refs: Dict[str, Any],
    input_preview: Dict[str, Any],
    prompt_summary: Dict[str, Any],
    llm_result_summary: Dict[str, Any],
    output_refs: Dict[str, Any],
    output_preview: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = {
        "status": NodeEventStatus.COMPLETED.value,
        **leading_fields,
        "input_refs": input_refs,
        "input_preview": input_preview,
        "prompt_summary": prompt_summary,
        "llm_result_summary": llm_result_summary,
        "output_refs": output_refs,
    }
    if output_preview is not None:
        data["output_preview"] = output_preview
    return data


async def invoke_json_decision_node(
    state: Dict[str, Any],
    config: RunnableConfig,
    *,
    started: float,
    spec: JsonDecisionNodeSpec,
    llm: Any,
    llm_retry_observer: Callable[[], Tuple[List[Dict[str, Any]], Callable[[Dict[str, Any]], None]]],
    prompt_summary: Callable[[str, str, str], Dict[str, Any]],
    invoke_llm_for_node: Callable[..., Awaitable[Any]],
    safe_json_object: Callable[[str], Dict[str, Any]],
) -> tuple[Any, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    retry_attempts, retry_observer = llm_retry_observer()
    prompt_details = prompt_summary(spec.prompt_section, spec.system_message, spec.prompt)
    response = await invoke_llm_for_node(
        llm.ainvoke,
        [
            SystemMessage(content=spec.system_message),
            HumanMessage(content=spec.prompt),
        ],
        state=state,
        config=config,
        node=spec.node_name,
        started=started,
        retry_observer=retry_observer,
        retry_attempts=retry_attempts,
        model_name=state.get("llm_model"),
        failure_data={**spec.failure_data, "prompt_summary": prompt_details},
    )
    parsed = safe_json_object(str(getattr(response, "content", "") or ""))
    return response, parsed, prompt_details, retry_attempts
