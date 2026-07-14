from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from langchain_core.runnables import RunnableConfig


@dataclass(frozen=True)
class ToolWorkerSpec:
    node_name: str
    tool_name: str
    evidence_kind: str
    evidence_label: str
    tool: Any
    tool_input: Callable[[Dict[str, Any]], Any]
    skip_reason: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None
    state_update: Optional[
        Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, list[Dict[str, Any]]], Dict[str, Any]]
    ] = None


async def run_tool_worker(
    state: Dict[str, Any],
    config: RunnableConfig,
    *,
    started: float,
    spec: ToolWorkerSpec,
    should_skip_worker: Callable[[Dict[str, Any], str], bool],
    skipped_worker_update: Callable[[Dict[str, Any], RunnableConfig, str, float, str], Dict[str, Any]],
    tool_config_for_node: Callable[..., RunnableConfig],
    invoke_tool_for_node: Callable[..., Awaitable[Any]],
    normalize_tool_result: Callable[..., Dict[str, Any]],
    combine_evidence: Callable[..., str],
    evidence_text_limit: Callable[[Dict[str, Any]], int],
    append_evidence_packet: Callable[..., list[Dict[str, Any]]],
    refs_from_artifacts: Callable[[Any], Dict[str, Any]],
    state_evidence_refs: Callable[[Dict[str, Any]], Dict[str, Any]],
    compact_refs: Callable[[Dict[str, Any]], Dict[str, Any]],
    compact_preview: Callable[..., str],
    normalize_warnings: Callable[[Any], list[Any]],
    log_node_end: Callable[[Dict[str, Any], str, float, Optional[Dict[str, Any]]], None],
    append_event: Callable[..., list[Dict[str, Any]]],
    append_tool_event: Callable[..., list[Dict[str, Any]]],
) -> Dict[str, Any]:
    reason = spec.skip_reason(state) if spec.skip_reason is not None else None
    if reason:
        return skipped_worker_update(state, config, spec.node_name, started, reason)
    if should_skip_worker(state, spec.node_name):
        return skipped_worker_update(state, config, spec.node_name, started, "not_selected_by_plan")

    tool_input = spec.tool_input(state)
    tool_config = tool_config_for_node(
        state,
        config,
        caller_node=spec.node_name,
        tool_name=spec.tool_name,
        started=started,
    )
    raw = await invoke_tool_for_node(
        spec.tool,
        tool_input,
        state=state,
        config=tool_config,
        node=spec.node_name,
        started=started,
    )
    payload = normalize_tool_result(raw, tool_name=spec.tool_name, config=tool_config)
    artifacts = payload.get("artifacts") or {}
    evidence = combine_evidence(
        state.get("evidence"),
        payload.get("content", ""),
        label=spec.evidence_label,
        limit=evidence_text_limit(state),
    )
    evidence_packets = append_evidence_packet(
        state,
        config,
        kind=spec.evidence_kind,
        content=payload.get("content", ""),
        refs=refs_from_artifacts(artifacts),
    )

    update = spec.state_update(state, payload, artifacts, evidence, evidence_packets) if spec.state_update else {}
    output_state = {**state, **update, "evidence": evidence, "evidence_packets": evidence_packets}
    data = {
        "status": "completed" if payload.get("ok", True) else "failed",
        "warnings": normalize_warnings(payload.get("warnings")),
        "input_refs": state_evidence_refs(state),
        "input_preview": {
            "question": compact_preview(state.get("question")),
            "previous_evidence": compact_preview(state.get("evidence")),
        },
        "evidence_chars": len(str(evidence or "")),
        "evidence_packet_count": len(evidence_packets),
    }
    if "document_sources" in update:
        data["document_source_count"] = len(update["document_sources"])
    if "web_sources" in update:
        data["web_source_count"] = len(update["web_sources"])
    if "used_chat_ids" in update:
        data["used_chat_id_count"] = len(update["used_chat_ids"])
    if "timeline_event_count" in update:
        data["timeline_event_count"] = update["timeline_event_count"]

    data["output_refs"] = compact_refs(
        {
            **state_evidence_refs(output_state),
            **refs_from_artifacts(artifacts),
        }
    )
    if "timeline_refs" in update:
        data["output_refs"] = compact_refs({**state_evidence_refs(state), **update["timeline_refs"]})
    data["output_preview"] = {"evidence": compact_preview(evidence)}

    log_node_end(state, spec.node_name, started, data)
    return {
        "evidence": evidence,
        "evidence_packets": evidence_packets,
        **{key: value for key, value in update.items() if key not in {"timeline_event_count", "timeline_refs"}},
        "node_events": append_event(state, spec.node_name, data, started=started, config=config),
        "tool_events": append_tool_event(state, payload, tool_input=tool_input, config=config),
    }
