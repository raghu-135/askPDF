from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agent_workflows.enums import NodeEventStatus


DECISION_REPAIR_PREVIEW_CHARS = 8_000


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


async def invoke_validated_json_decision_node(
    state: Dict[str, Any],
    config: RunnableConfig,
    *,
    started: float,
    spec: JsonDecisionNodeSpec,
    validate: Callable[[Dict[str, Any]], List[str]],
    review_when: Optional[Callable[[Dict[str, Any]], bool]] = None,
    llm: Any,
    llm_retry_observer: Callable[[], Tuple[List[Dict[str, Any]], Callable[[Dict[str, Any]], None]]],
    prompt_summary: Callable[[str, str, str], Dict[str, Any]],
    invoke_llm_for_node: Callable[..., Awaitable[Any]],
    safe_json_object: Callable[[str], Dict[str, Any]],
) -> tuple[Any, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Invoke a typed JSON decision and perform one bounded contract-repair turn.

    The validator owns structural policy; the model still owns semantic routing. This
    mirrors the structured-output retry pattern without assuming every configured
    OpenAI-compatible provider implements native JSON Schema response formats.
    """

    response, parsed, prompt_details, retry_attempts = await invoke_json_decision_node(
        state,
        config,
        started=started,
        spec=spec,
        llm=llm,
        llm_retry_observer=llm_retry_observer,
        prompt_summary=prompt_summary,
        invoke_llm_for_node=invoke_llm_for_node,
        safe_json_object=safe_json_object,
    )
    initial_errors = validate(parsed)
    repair_data = {
        "attempted": False,
        "mode": None,
        "initial_errors": initial_errors,
        "remaining_errors": initial_errors,
    }
    needs_coverage_review = not initial_errors and review_when is not None and review_when(parsed)
    if not initial_errors and not needs_coverage_review:
        return response, parsed, prompt_details, retry_attempts, repair_data

    review_instruction = (
        "The previous response did not satisfy the typed planning contract. Correct the structure and "
        if initial_errors
        else "The previous response satisfies the structural contract. Perform one bounded coverage review and "
    )
    repair_prompt = (
        spec.prompt
        + "\n\n## Structured Output Repair\n\n"
        + review_instruction
        + "review whether the selected workers cover every explicit source, scope, and evidence requirement "
        + "in the question. Do not select workers by keyword alone. Return only the corrected JSON object.\n\n"
        + ("Validation errors:\n- " + "\n- ".join(initial_errors) + "\n\n" if initial_errors else "")
        + "\n\nPrevious response:\n"
        + json.dumps(parsed, ensure_ascii=True, sort_keys=True, default=str)[:DECISION_REPAIR_PREVIEW_CHARS]
    )
    repaired_response, repaired, repaired_prompt_details, repaired_attempts = await invoke_json_decision_node(
        state,
        config,
        started=started,
        spec=JsonDecisionNodeSpec(
            node_name=spec.node_name,
            prompt_section=spec.prompt_section,
            system_message=spec.system_message,
            prompt=repair_prompt,
            failure_data=spec.failure_data,
        ),
        llm=llm,
        llm_retry_observer=llm_retry_observer,
        prompt_summary=prompt_summary,
        invoke_llm_for_node=invoke_llm_for_node,
        safe_json_object=safe_json_object,
    )
    remaining_errors = validate(repaired)
    repair_data = {
        "attempted": True,
        "mode": "contract_repair" if initial_errors else "coverage_review",
        "initial_errors": initial_errors,
        "remaining_errors": remaining_errors,
    }
    if remaining_errors:
        return response, parsed, prompt_details, [*retry_attempts, *repaired_attempts], repair_data
    return (
        repaired_response,
        repaired,
        repaired_prompt_details,
        [*retry_attempts, *repaired_attempts],
        repair_data,
    )
