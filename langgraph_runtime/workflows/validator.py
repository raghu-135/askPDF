from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from langgraph_runtime.agent.tool_registry import known_tool_contract_ids, tool_contracts_by_id
from langgraph_runtime.workflows.graph_validation import GenericGraphValidator
from langgraph_runtime.workflows.node_catalog import get_node_catalog
from langgraph_runtime.workflows.route_registry import get_route_function_registry
from langgraph_runtime.workflows.workflow_requirements import (
    workflow_node_tool_requirements,
    workflow_required_tool_ids,
    workflow_required_tool_ids_from_nodes,
)
from langgraph_runtime.workflows.workflow_runtime import (
    ALLOWED_WORKFLOW_CONFIG_KEYS,
    RUNTIME_TEXT_FIELDS,
    SUPPORTED_RUNTIME_KINDS,
    normalize_runtime_for_validation,
    replan_loop_policy,
    workflow_allows_replans_override,
    workflow_supports_replans,
)


class WorkflowValidationError(ValueError):
    """Raised when an agent workflow spec is invalid."""


class WorkflowValidator:
    """Validator for schema v2 catalog-backed agent workflow specs."""

    def validate(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        errors = self.collect_errors(spec)
        result = {"valid": not errors, "errors": errors}
        if errors:
            raise WorkflowValidationError("; ".join(errors))
        return result

    def collect_errors(self, spec: Dict[str, Any]) -> list[str]:
        if not isinstance(spec, dict):
            return ["spec must be an object"]
        if spec.get("schema_version") != 1:
            return ["schema_version must be 1"]
        errors = self._collect_runtime_errors(spec)
        errors.extend(GenericGraphValidator().collect_errors(spec))
        return errors

    def _collect_runtime_errors(self, spec: Dict[str, Any]) -> list[str]:
        errors: list[str] = []
        runtime = normalize_runtime_for_validation(spec.get("runtime"))
        if not runtime:
            return ["runtime must be an object"]
        kind = runtime.get("kind")
        if kind not in SUPPORTED_RUNTIME_KINDS:
            errors.append(f"runtime.kind must be one of: {', '.join(sorted(SUPPORTED_RUNTIME_KINDS))}")
        for field in sorted(RUNTIME_TEXT_FIELDS):
            if not isinstance(runtime.get(field), str) or not runtime.get(field):
                errors.append(f"runtime.{field} must be a non-empty string")
        features = runtime.get("features", {})
        if not isinstance(features, dict):
            errors.append("runtime.features must be an object")
        else:
            for feature in (
                "supports_replans",
                "supports_parallel_dispatch",
                "supports_answer_quality",
                "supports_corrective_retrieval",
                "supports_long_running_tasks",
                "supports_artifacts",
                "allows_replans_override",
            ):
                if feature in features and not isinstance(features.get(feature), bool):
                    errors.append(f"runtime.features.{feature} must be a boolean")
        prompt_preview = runtime.get("prompt_preview")
        if prompt_preview is not None and not isinstance(prompt_preview, str):
            errors.append("runtime.prompt_preview must be a string")
        return errors

    def report(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structured validation report for admin/debug API consumers."""

        spec_obj = spec if isinstance(spec, dict) else {}
        errors = self.collect_errors(spec)
        config = spec_obj.get("config") if isinstance(spec_obj.get("config"), dict) else {}
        config = config if isinstance(config, dict) else {}
        allowed_tool_ids = config.get("allowed_tool_ids") if isinstance(config.get("allowed_tool_ids"), list) else []
        known_tool_ids = known_tool_contract_ids()
        required_tool_ids = workflow_required_tool_ids_from_nodes(spec_obj) or workflow_required_tool_ids(spec_obj)
        missing_required_tool_ids = set(required_tool_ids - set(allowed_tool_ids))
        for error in errors:
            prefix = "missing required allowed_tool_ids: "
            if isinstance(error, str) and error.startswith(prefix):
                missing_required_tool_ids.update(
                    item.strip()
                    for item in error[len(prefix):].split(",")
                    if item.strip()
                )
        issues = [self._structured_issue(message, spec_obj) for message in errors]
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": [],
            "issues": issues,
            "schema_version": spec_obj.get("schema_version"),
            "workflow_id": spec_obj.get("workflow_id"),
            "runtime": normalize_runtime_for_validation(spec_obj.get("runtime")),
            "supported_workflow_ids": ["declarative"],
            "allowed_tool_ids": allowed_tool_ids,
            "required_tool_ids": sorted(required_tool_ids),
            "missing_required_tool_ids": sorted(missing_required_tool_ids),
            "unknown_allowed_tool_ids": sorted(set(allowed_tool_ids) - known_tool_ids),
        }

    @staticmethod
    def _structured_issue(message: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        graph = ((spec.get("config") or {}).get("graph") or {}) if isinstance(spec, dict) else {}
        node_ids = {
            str(node.get("id"))
            for node in graph.get("nodes", [])
            if isinstance(node, dict) and node.get("id")
        }
        explicit_node_id = None
        if "graph conditional edge from " in message:
            explicit_node_id = message.split("graph conditional edge from ", 1)[1].split(" ", 1)[0]
        elif " from node " in message:
            explicit_node_id = message.split(" from node ", 1)[1].split(" ", 1)[0]
        elif message.startswith("node "):
            explicit_node_id = message.split(" ", 2)[1]
        elif message.startswith("evidence evaluator "):
            explicit_node_id = message.split(" ", 2)[2].split(" ", 1)[0]
        elif message.startswith("replanner "):
            explicit_node_id = message.split(" ", 1)[1].split(" ", 1)[0]
        node_id = (
            explicit_node_id
            if explicit_node_id in node_ids
            else next((candidate for candidate in node_ids if candidate in message), None)
        )
        lowered = message.lower()
        code = (
            "incomplete_conditional_routes" if "missing route labels" in lowered
            else "invalid_route_target" if lowered.startswith("route ") and "must target" in lowered
            else "ambiguous_outgoing_flow" if "outgoing edges" in lowered
            else "non_terminating_path" if "no path to end" in lowered
            else "missing_evidence_flow" if "upstream worker evidence" in lowered
            else "invalid_replan_cycle" if "replanner" in lowered and (
                "evidence evaluator" in lowered or "one more visit" in lowered
            )
            else "insufficient_loop_budget" if "cycle revisit" in lowered or "repeat visit budget" in lowered
            else "missing_finalizer" if "requires a finalizer node" in lowered
            else "missing_finalizer_flow" if "flow through a finalizer" in lowered
            else "missing_start" if "start" in lowered and ("missing" in lowered or "must" in lowered)
            else "missing_end" if "end" in lowered and ("missing" in lowered or "must" in lowered)
            else "incompatible_connection" if "cannot connect" in lowered
            else "unreachable_node" if "unreachable" in lowered
            else "missing_route" if "route" in lowered and "missing" in lowered
            else "invalid_workflow"
        )
        edge_index = None
        route = None
        if code in {"incomplete_conditional_routes", "invalid_route_target", "ambiguous_outgoing_flow"}:
            source = None
            if "graph conditional edge from " in message:
                source = message.split("graph conditional edge from ", 1)[1].split(" ", 1)[0]
            elif " from node " in message:
                source = message.split(" from node ", 1)[1].split(" ", 1)[0]
            elif message.startswith("node "):
                source = message.split(" ", 2)[1]
            for index, edge in enumerate(graph.get("edges", [])):
                if isinstance(edge, dict) and edge.get("from") == source:
                    edge_index = index
                    break
            if message.startswith("route "):
                route = message.split(" ", 2)[1]
        fix = None
        if code in {"missing_start", "missing_end", "unreachable_node", "incompatible_connection", "missing_route"}:
            fix = {
                "kind": code,
                **({"node_id": node_id} if node_id else {}),
                **({"requires_confirmation": True} if code in {"unreachable_node", "missing_route"} else {}),
            }
        return {
            "code": code,
            "severity": "error",
            "message": message,
            "node_id": node_id,
            "edge_index": edge_index,
            "route": route,
            "allowed_alternatives": [],
            "fix": fix,
        }



class WorkflowResolver:
    """Freeze the effective built-in agent workflow config for an agent run."""

    def __init__(self, validator: Optional[WorkflowValidator] = None):
        self.validator = validator or WorkflowValidator()

    def resolve(
        self,
        spec: Dict[str, Any],
        *,
        thread_settings: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved = deepcopy(spec)
        config = dict(resolved.get("config") or {})

        for source in (thread_settings or {}, request_overrides or {}):
            for key in ALLOWED_WORKFLOW_CONFIG_KEYS:
                if key == "replans" and not workflow_allows_replans_override(resolved):
                    continue
                if key in source and source[key] is not None:
                    config[key] = source[key]
        if not workflow_supports_replans(resolved):
            config.pop("replans", None)

        resolved["config"] = config
        if workflow_supports_replans(resolved):
            config["loop_policy"] = replan_loop_policy(resolved, config)
        self.validator.validate(resolved)
        return resolved
