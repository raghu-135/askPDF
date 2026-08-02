from __future__ import annotations

from typing import Any, Dict, List, Mapping

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from app.agent_workflows.enums import DebugGraphNodeStatus, GraphSentinel, NodeEventStatus, TraceStatus, WorkflowNodeType
from app.agent_workflows.trace_otel import (
    _first_number,
    _node_display_name,
    _node_metadata,
    _observability_metadata,
)
from app.agent_workflows.trace_payloads import _interrupt_summary
from app.agent_workflows.trace_sanitization import _as_dict, _as_list, _bounded_value


def _as_string_list(value: Any) -> List[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _summary_node(span: Mapping[str, Any]) -> Dict[str, Any]:
    attributes = _as_dict(span.get("attributes"))
    raw = _as_dict(span.get("raw"))
    warnings = [
        _as_dict(event.get("attributes")).get("warning.code")
        for event in _as_list(span.get("events"))
        if isinstance(event, dict) and event.get("name") == "warning"
    ]
    error = next(
        (
            _as_dict(event.get("attributes"))
            for event in _as_list(span.get("events"))
            if isinstance(event, dict) and event.get("name") == "exception"
        ),
        None,
    )
    node_id = str(attributes.get("askpdf.node.id") or attributes.get(SpanAttributes.GRAPH_NODE_ID) or span.get("name") or "unknown_node")
    return {
        "id": node_id,
        "type": attributes.get("askpdf.node.type"),
        "label": attributes.get("askpdf.node.name"),
        "visitIndex": attributes.get("askpdf.node.visit_index"),
        "status": span.get("status"),
        "skipped": span.get("status") == NodeEventStatus.SKIPPED.value,
        "durationMs": span.get("duration_ms"),
        "route": attributes.get("askpdf.route"),
        "routeReason": attributes.get("askpdf.route_reason"),
        "evaluatorRoute": attributes.get("askpdf.evaluator_route"),
        "evaluationConfidence": attributes.get("askpdf.evaluation_confidence"),
        "replanCount": attributes.get("askpdf.replan_count"),
        "usedMemoryIdCount": attributes.get("askpdf.used_memory_id_count"),
        "executionPlan": _as_string_list(attributes.get("askpdf.execution_plan")),
        "warningCodes": [str(warning) for warning in warnings if warning],
        "error": error,
        "span": dict(span),
        "raw": raw,
    }


def _summary_tool(span: Mapping[str, Any]) -> Dict[str, Any]:
    attributes = _as_dict(span.get("attributes"))
    raw = _as_dict(span.get("raw"))
    warning_events = [
        _as_dict(event.get("attributes")).get("warning.code")
        for event in _as_list(span.get("events"))
        if isinstance(event, dict) and event.get("name") == "warning"
    ]
    return {
        "name": str(attributes.get(SpanAttributes.TOOL_NAME) or span.get("name") or "tool"),
        "id": attributes.get(SpanAttributes.TOOL_ID),
        "category": attributes.get("askpdf.tool.category"),
        "displayName": attributes.get(SpanAttributes.TOOL_DESCRIPTION) or span.get("name"),
        "callerNode": attributes.get("askpdf.caller_node"),
        "callerNodeType": attributes.get("askpdf.caller_node_type"),
        "callerVisitIndex": attributes.get("askpdf.caller_visit_index"),
        "ok": span.get("status") != TraceStatus.ERROR.value,
        "durationMs": span.get("duration_ms"),
        "sourceCount": attributes.get("askpdf.source_count"),
        "warningCodes": [str(warning) for warning in warning_events if warning],
        "span": dict(span),
        "raw": raw,
    }


def _build_summary_from_trace(trace: Dict[str, Any], resolved_spec: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = _as_dict(trace.get("metrics"))
    spans = _as_list(trace.get("spans"))
    node_spans = [
        span
        for span in spans
        if _as_dict(_as_dict(span).get("attributes")).get("askpdf.node.id")
        and _as_dict(span).get("kind") != OpenInferenceSpanKindValues.LLM.value
    ]
    tool_spans = [span for span in spans if _as_dict(span).get("kind") == OpenInferenceSpanKindValues.TOOL.value]
    errors = [
        _as_dict(event.get("attributes"))
        for span in spans
        for event in _as_list(_as_dict(span).get("events"))
        if isinstance(event, dict) and event.get("name") == "exception"
    ]
    nodes = [_summary_node(span) for span in node_spans if isinstance(span, dict)]
    tools = [_summary_tool(span) for span in tool_spans if isinstance(span, dict)]
    used_node_count = len({node.get("id") for node in nodes if node.get("id") and not node.get("skipped")})
    warning_count = sum(len(node.get("warningCodes") or []) for node in nodes) + sum(len(tool.get("warningCodes") or []) for tool in tools)
    error_count = int(metrics.get("error_count") or 0)
    if errors:
        error_count = max(error_count, len(errors))
    memory_refs: List[Dict[str, Any]] = []
    memory_scopes: List[Dict[str, Any]] = []
    memory_scope_policies: List[Dict[str, Any]] = []
    memory_applied_overrides: List[Dict[str, Any]] = []
    memory_suppressed_ids: List[str] = []
    for tool in tools:
        raw = _as_dict(tool.get("raw"))
        refs = _as_dict(raw.get("artifact_refs"))
        artifacts = _as_dict(raw.get("artifacts"))
        memory_refs.extend([item for item in _as_list(refs.get("memories")) if isinstance(item, dict)])
        memory_scopes.extend([item for item in _as_list(artifacts.get("memory_scopes")) if isinstance(item, dict)])
        policy = artifacts.get("memory_scope_policy")
        if isinstance(policy, dict):
            memory_scope_policies.append(policy)
        memory_applied_overrides.extend([
            item for item in _as_list(artifacts.get("memory_applied_overrides"))
            if isinstance(item, dict)
        ])
        memory_suppressed_ids.extend(str(item) for item in _as_list(artifacts.get("memory_suppressed_ids")) if item)
    config = _as_dict(resolved_spec.get("config"))
    evaluator_nodes = [node for node in nodes if node.get("id") == WorkflowNodeType.EVIDENCE_EVALUATOR.value]
    replanner_nodes = [node for node in nodes if node.get("id") == WorkflowNodeType.REPLANNER.value]
    last_evaluator = evaluator_nodes[-1] if evaluator_nodes else {}
    last_evaluator_raw = _as_dict(last_evaluator.get("raw"))
    replan_count = max(
        [_first_number(node.get("replanCount"), _as_dict(node.get("raw")).get("replan_count")) for node in replanner_nodes + evaluator_nodes]
        or [0]
    )
    return {
        "status": trace.get("status"),
        "route": _as_dict(trace.get("attributes")).get("askpdf.route") or metrics.get("route"),
        "routeReason": _as_dict(trace.get("attributes")).get("askpdf.route_reason"),
        "evaluatorRoute": last_evaluator.get("evaluatorRoute") or last_evaluator_raw.get("evaluator_route"),
        "evaluationConfidence": last_evaluator.get("evaluationConfidence") or last_evaluator_raw.get("evaluation_confidence"),
        "evaluatorReport": _bounded_value(last_evaluator_raw.get("evaluator_report")),
        "evidenceGaps": _as_string_list(last_evaluator_raw.get("evidence_gaps")),
        "replanCount": replan_count,
        "durationMs": trace.get("duration_ms"),
        "metrics": metrics,
        "nodes": nodes,
        "tools": tools,
        "usedNodeCount": used_node_count,
        "availableNodeCount": len(_as_list(_as_dict(config.get("graph")).get("nodes"))) or None,
        "usedToolCount": int(metrics.get("tool_event_count") or len(tools)),
        "availableToolCount": len(set(_as_list(config.get("allowed_tool_ids")))) or None,
        "warningCount": int(metrics.get("tool_warning_count") or warning_count),
        "errorCount": error_count,
        "errors": [error for error in errors if error],
        "memory": {
            "recalledMemoryIds": [str(item.get("memory_id")) for item in memory_refs if item.get("memory_id")],
            "searchedScopes": memory_scopes,
            "scopePolicies": memory_scope_policies,
            "appliedOverrides": memory_applied_overrides,
            "suppressedMemoryIds": list(dict.fromkeys(memory_suppressed_ids)),
            "recalledCount": len({str(item.get("memory_id")) for item in memory_refs if item.get("memory_id")}),
        },
        **_interrupt_summary(trace),
    }


def _execution_plan_from_summary(summary: Mapping[str, Any]) -> List[str]:
    for node in _as_list(summary.get("nodes")):
        plan = node.get("executionPlan")
        if isinstance(plan, list) and plan:
            return _as_string_list(plan)
        raw_plan = _as_dict(node.get("raw")).get("execution_plan")
        if isinstance(raw_plan, list) and raw_plan:
            return _as_string_list(raw_plan)
    return []


def _graph_node_status(
    node_id: str,
    summary_node: Mapping[str, Any],
    tool_summaries: List[Dict[str, Any]],
    execution_plan: List[str],
) -> str:
    if summary_node.get("error") or any(not tool.get("ok", True) for tool in tool_summaries):
        return DebugGraphNodeStatus.ERROR.value
    if summary_node.get("skipped"):
        return NodeEventStatus.SKIPPED.value
    if summary_node:
        return DebugGraphNodeStatus.ACTIVE.value
    if node_id in execution_plan:
        return DebugGraphNodeStatus.PLANNED.value
    return DebugGraphNodeStatus.INACTIVE.value


def _has_active_node(node_id: str, nodes_by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    status = _as_dict(nodes_by_id.get(node_id)).get("status")
    return status in {
        DebugGraphNodeStatus.ACTIVE.value,
        DebugGraphNodeStatus.PLANNED.value,
        NodeEventStatus.SKIPPED.value,
        DebugGraphNodeStatus.ERROR.value,
    }


def _graph_tool_summary(tool: Mapping[str, Any]) -> Dict[str, Any]:
    raw = _as_dict(tool.get("raw"))
    return {
        "toolName": tool.get("name"),
        "displayName": tool.get("displayName"),
        "callerNode": tool.get("callerNode"),
        "ok": tool.get("ok", True),
        "elapsedMs": tool.get("durationMs"),
        "sourceCount": tool.get("sourceCount"),
        "warnings": tool.get("warningCodes") or [],
        "artifactKeys": _as_list(raw.get("artifact_keys")),
        "toolInput": raw.get("tool_input"),
        "resultPreview": raw.get("result_preview"),
        "artifactRefs": raw.get("artifact_refs"),
        "artifactSummary": raw.get("artifact_summary"),
        "traceSpan": tool.get("span"),
        "raw": raw,
    }


def build_debug_graph(
    *,
    resolved_spec: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the API-only graph view model from canonical stored telemetry."""

    graph_spec = _as_dict(_as_dict(resolved_spec.get("config")).get("graph"))
    spec_nodes = [node for node in _as_list(graph_spec.get("nodes")) if isinstance(node.get("id"), str)]
    spec_edges = [edge for edge in _as_list(graph_spec.get("edges")) if isinstance(edge.get("from"), str)]
    summary_nodes = {node["id"]: node for node in _as_list(summary.get("nodes")) if isinstance(node.get("id"), str)}
    tools_by_node: Dict[str, List[Dict[str, Any]]] = {}
    for tool in _as_list(summary.get("tools")):
        caller = tool.get("callerNode")
        if isinstance(caller, str):
            tools_by_node.setdefault(caller, []).append(tool)

    execution_plan = _execution_plan_from_summary(summary)
    selected_route = summary.get("route")
    nodes = []
    for spec in spec_nodes:
        node_id = spec["id"]
        summary_node = summary_nodes.get(node_id, {})
        tool_summaries = tools_by_node.get(node_id, [])
        status = _graph_node_status(node_id, summary_node, tool_summaries, execution_plan)
        raw_events = [summary_node.get("raw")] if summary_node.get("raw") else []
        raw = _as_dict(summary_node.get("raw"))
        nodes.append(
            {
                "id": node_id,
                "type": str(spec.get("type") or node_id),
                "label": str(spec.get("label") or _node_display_name(str(spec.get("type") or node_id))),
                "category": spec.get("category") or _node_metadata(str(spec.get("type") or node_id)).get("category"),
                "description": spec.get("description"),
                "capabilities": spec.get("capabilities") or _node_metadata(str(spec.get("type") or node_id)).get("capabilities"),
                "observability": spec.get("observability") or _observability_metadata(str(spec.get("type") or node_id)),
                "status": status,
                "elapsedMs": summary_node.get("durationMs"),
                "route": summary_node.get("route"),
                "routeReason": summary_node.get("routeReason"),
                "evaluatorRoute": summary_node.get("evaluatorRoute"),
                "evaluationConfidence": summary_node.get("evaluationConfidence"),
                "replanCount": summary_node.get("replanCount"),
                "skipped": bool(summary_node.get("skipped")),
                "skipReason": raw.get("skip_reason"),
                "executionPlan": execution_plan if node_id in {WorkflowNodeType.PLANNER.value, WorkflowNodeType.REPLANNER.value} else None,
                "warnings": summary_node.get("warningCodes") or [],
                "inputRefs": raw.get("input_refs"),
                "outputRefs": raw.get("output_refs"),
                "inputPreview": raw.get("input_preview"),
                "outputPreview": raw.get("output_preview"),
                "promptSummary": raw.get("prompt_summary"),
                "llmResultSummary": raw.get("llm_result_summary"),
                "llmSummary": _as_dict(raw.get("llm_result_summary")).get("llm"),
                "toolSummaries": [_graph_tool_summary(tool) for tool in tool_summaries],
                "warningCount": len(summary_node.get("warningCodes") or []) + sum(len(tool.get("warningCodes") or []) for tool in tool_summaries),
                "errorCount": (1 if summary_node.get("error") else 0) + sum(1 for tool in tool_summaries if not tool.get("ok", True)),
                "sourceCount": sum(int(tool.get("sourceCount") or 0) for tool in tool_summaries),
                "artifactCount": sum(len(_as_list(_as_dict(tool.get("raw")).get("artifact_keys"))) for tool in tool_summaries),
                "traceSpans": [summary_node.get("span")] if summary_node.get("span") else [],
                "rawEvents": raw_events,
            }
        )

    nodes_by_id = {node["id"]: node for node in nodes}
    edges = []
    for index, edge in enumerate(spec_edges):
        source = edge.get("from")
        if source == GraphSentinel.START.value or edge.get("to") == GraphSentinel.END.value:
            continue
        if edge.get("conditional") and isinstance(edge.get("routes"), dict):
            source_node = nodes_by_id.get(str(source), {})
            source_raw = _as_dict(_as_list(source_node.get("rawEvents"))[-1]) if _as_list(source_node.get("rawEvents")) else {}
            for route, target in edge["routes"].items():
                if not isinstance(target, str):
                    continue
                selected = selected_route == route or (
                    source == WorkflowNodeType.EVIDENCE_EVALUATOR.value
                    and (source_node.get("evaluatorRoute") == route or source_raw.get("evaluator_route") == route)
                )
                edges.append(
                    {
                        "id": f"{source}-{route}-{target}",
                        "source": source,
                        "target": target,
                        "label": route,
                        "route": route,
                        "selected": selected,
                        "active": selected or (_has_active_node(source, nodes_by_id) and _has_active_node(target, nodes_by_id)),
                        "conditional": True,
                        "raw": edge,
                    }
                )
            continue
        target = edge.get("to")
        if not isinstance(target, str):
            continue
        edges.append(
            {
                "id": f"{source}-{target}-{index}",
                "source": source,
                "target": target,
                "selected": False,
                "active": _has_active_node(source, nodes_by_id) and _has_active_node(target, nodes_by_id),
                "conditional": False,
                "raw": edge,
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "executionPlan": execution_plan,
        "selectedRoute": selected_route,
    }
