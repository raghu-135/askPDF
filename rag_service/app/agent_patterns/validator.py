from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from app.agent.tool_registry import collect_tool_contract_metadata_errors, known_tool_contract_ids, tool_contracts_by_id
from app.agent_patterns.node_catalog import (
    collect_node_catalog_errors,
    get_node_catalog,
    node_type_allowed_tool_contract_ids,
    known_node_types,
)
from app.agent_patterns.route_registry import (
    collect_route_function_registry_errors,
    get_route_function_registry,
    known_route_function_ids,
    route_function_allowed_for_node_type,
    route_function_labels,
)
from app.agent_patterns.templates import (
    ALLOWED_ROUTER_RAG_CONFIG_KEYS,
    EVALUATOR_REPLANNER_RAG_AGENT_ID,
    EVALUATOR_REPLANNER_RAG_NODE_TOOL_REQUIREMENTS,
    EVALUATOR_REPLANNER_RAG_REQUIRED_TOOL_IDS,
    PLAN_EXECUTE_RAG_AGENT_ID,
    PLAN_EXECUTE_RAG_NODE_TOOL_REQUIREMENTS,
    PLAN_EXECUTE_RAG_REQUIRED_TOOL_IDS,
    ROUTER_RAG_NODE_TOOL_REQUIREMENTS,
    ROUTER_RAG_AGENT_ID,
    ROUTER_RAG_REQUIRED_TOOL_IDS,
    SUPPORTED_BUILTIN_TEMPLATE_IDS,
    WEB_APPROVAL_GATE_ID,
)
from app.models.llm_server_client import (
    MAX_CUSTOM_INSTRUCTIONS_CHARS,
    REPLANS_LIMIT,
    MAX_SYSTEM_ROLE_CHARS,
)


class TemplateValidationError(ValueError):
    """Raised when an agent pattern template spec is invalid."""


def _known_tool_ids() -> set[str]:
    return known_tool_contract_ids()


PATTERN_REQUIRED_TOOL_IDS = {
    ROUTER_RAG_AGENT_ID: ROUTER_RAG_REQUIRED_TOOL_IDS,
    PLAN_EXECUTE_RAG_AGENT_ID: PLAN_EXECUTE_RAG_REQUIRED_TOOL_IDS,
    EVALUATOR_REPLANNER_RAG_AGENT_ID: EVALUATOR_REPLANNER_RAG_REQUIRED_TOOL_IDS,
}

PATTERN_NODE_TOOL_REQUIREMENTS = {
    ROUTER_RAG_AGENT_ID: ROUTER_RAG_NODE_TOOL_REQUIREMENTS,
    PLAN_EXECUTE_RAG_AGENT_ID: PLAN_EXECUTE_RAG_NODE_TOOL_REQUIREMENTS,
    EVALUATOR_REPLANNER_RAG_AGENT_ID: EVALUATOR_REPLANNER_RAG_NODE_TOOL_REQUIREMENTS,
}

HITL_ACTIONS = {"approve", "approve_selected", "continue_without", "reject", "edit"}
HITL_PHASES = {"before", "after", "inside_tool"}
HITL_MODES = {"approval", "choice", "review"}
HITL_SELECTION_MODES = {"single", "multi", "single_or_multi"}
HITL_GATE_KEYS = {
    "enabled",
    "title",
    "prompt",
    "body",
    "allowed_actions",
    "default_action",
    "target",
    "phase",
    "mode",
    "interrupt_type",
    "type",
    "selection_mode",
    "options",
    "routes",
    "conditions",
    "payload_projection",
    "editable_fields",
    "requires_reason",
    "reject_behavior",
    "max_interrupts_per_run",
}


class TemplateValidator:
    """Validator for schema v2 catalog-backed agent pattern specs."""

    def validate(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        errors = self.collect_errors(spec)
        result = {"valid": not errors, "errors": errors}
        if errors:
            raise TemplateValidationError("; ".join(errors))
        return result

    def collect_errors(self, spec: Dict[str, Any]) -> list[str]:
        if not isinstance(spec, dict):
            return ["spec must be an object"]
        if spec.get("schema_version") != 2:
            return ["schema_version must be 2"]
        return GenericGraphValidator().collect_errors(spec)

    def report(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structured validation report for admin/debug API consumers."""

        spec_obj = spec if isinstance(spec, dict) else {}
        errors = self.collect_errors(spec)
        config = spec_obj.get("config") if isinstance(spec_obj.get("config"), dict) else {}
        config = config if isinstance(config, dict) else {}
        allowed_tool_ids = config.get("allowed_tool_ids") if isinstance(config.get("allowed_tool_ids"), list) else []
        known_tool_ids = _known_tool_ids()
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": [],
            "schema_version": spec_obj.get("schema_version"),
            "pattern_type": spec_obj.get("pattern_type"),
            "supported_pattern_types": sorted([*SUPPORTED_BUILTIN_TEMPLATE_IDS, "custom_rag_agent"]),
            "allowed_tool_ids": allowed_tool_ids,
            "required_tool_ids": sorted(PATTERN_REQUIRED_TOOL_IDS.get(spec_obj.get("pattern_type"), set())),
            "missing_required_tool_ids": sorted(
                PATTERN_REQUIRED_TOOL_IDS.get(spec_obj.get("pattern_type"), set()) - set(allowed_tool_ids)
            ),
            "unknown_allowed_tool_ids": sorted(set(allowed_tool_ids) - known_tool_ids),
        }

    def _collect_tool_permission_errors(self, pattern_type: Any, allowed_tool_ids: set[str]) -> list[str]:
        errors: list[str] = []
        contracts_by_id = tool_contracts_by_id()
        node_tool_requirements = PATTERN_NODE_TOOL_REQUIREMENTS.get(pattern_type, {})
        for caller_node, contract_id in sorted(node_tool_requirements.items()):
            if contract_id not in allowed_tool_ids:
                continue
            contracts = contracts_by_id.get(contract_id) or []
            if not contracts:
                errors.append(f"{pattern_type} required tool contract is not registered: {contract_id}")
                continue
            if not any(caller_node in (contract.get("allowed_caller_nodes") or []) for contract in contracts):
                errors.append(
                    f"{pattern_type} tool contract {contract_id} is not allowed from node {caller_node}"
                )
        return errors

    def _collect_hitl_policy_errors(self, hitl_policy: Any, pattern_type: Any, graph: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(hitl_policy, dict):
            return ["hitl_policy must be an object"]
        unknown_policy_keys = sorted(set(hitl_policy) - {"enabled", "gates", "max_interrupts_per_run"})
        if unknown_policy_keys:
            errors.append(f"hitl_policy only supports keys: enabled, gates, max_interrupts_per_run; unknown: {', '.join(unknown_policy_keys)}")
        if "enabled" in hitl_policy and not isinstance(hitl_policy["enabled"], bool):
            errors.append("hitl_policy.enabled must be a boolean")
        if "max_interrupts_per_run" in hitl_policy:
            try:
                max_interrupts = int(hitl_policy.get("max_interrupts_per_run"))
            except (TypeError, ValueError):
                max_interrupts = 0
            if max_interrupts < 1:
                errors.append("hitl_policy.max_interrupts_per_run must be a positive integer")
        gates = hitl_policy.get("gates", {})
        if not isinstance(gates, dict):
            errors.append("hitl_policy.gates must be an object")
            return errors

        graph_nodes = graph.get("nodes") if isinstance(graph, dict) else []
        node_types = {
            node.get("id"): node.get("type")
            for node in graph_nodes
            if isinstance(node, dict) and isinstance(node.get("id"), str) and isinstance(node.get("type"), str)
        }
        for gate_id, gate in gates.items():
            if not isinstance(gate_id, str) or not gate_id:
                errors.append("hitl_policy.gates keys must be non-empty strings")
                continue
            if not isinstance(gate, dict):
                errors.append(f"hitl_policy.gates.{gate_id} must be an object")
                continue

            unknown_gate_keys = sorted(set(gate) - HITL_GATE_KEYS)
            if unknown_gate_keys:
                errors.append(f"hitl_policy.gates.{gate_id} has unknown keys: {', '.join(unknown_gate_keys)}")
            if "enabled" in gate and not isinstance(gate["enabled"], bool):
                errors.append(f"hitl_policy.gates.{gate_id}.enabled must be a boolean")

            mode = str(gate.get("mode") or "approval")
            if mode not in HITL_MODES:
                errors.append(f"hitl_policy.gates.{gate_id}.mode must be one of: {', '.join(sorted(HITL_MODES))}")
            phase = str(gate.get("phase") or "before")
            if phase not in HITL_PHASES:
                errors.append(f"hitl_policy.gates.{gate_id}.phase must be one of: {', '.join(sorted(HITL_PHASES))}")
            if phase == "inside_tool":
                errors.append(f"hitl_policy.gates.{gate_id}.phase inside_tool is reserved for tool wrappers")

            target = gate.get("target")
            if gate_id == WEB_APPROVAL_GATE_ID and target is None:
                target = {"node_id": "web_worker", "node_type": "web_worker"}
            if not isinstance(target, dict):
                errors.append(f"hitl_policy.gates.{gate_id}.target must be an object")
            else:
                target_node_id = target.get("node_id")
                target_node_type = target.get("node_type")
                if target_node_id is not None and not isinstance(target_node_id, str):
                    errors.append(f"hitl_policy.gates.{gate_id}.target.node_id must be a string")
                if target_node_type is not None and not isinstance(target_node_type, str):
                    errors.append(f"hitl_policy.gates.{gate_id}.target.node_type must be a string")
                if not target_node_id and not target_node_type:
                    errors.append(f"hitl_policy.gates.{gate_id}.target must include node_id or node_type")
                if isinstance(target_node_id, str) and target_node_id not in node_types:
                    errors.append(f"hitl_policy.gates.{gate_id}.target.node_id is unknown: {target_node_id}")
                if isinstance(target_node_type, str) and target_node_type not in set(node_types.values()):
                    errors.append(f"hitl_policy.gates.{gate_id}.target.node_type is unknown: {target_node_type}")

            for key in ("title", "prompt", "body", "default_action", "interrupt_type", "type", "selection_mode", "reject_behavior"):
                if key in gate and not isinstance(gate[key], str):
                    errors.append(f"hitl_policy.gates.{gate_id}.{key} must be a string")
            if "max_interrupts_per_run" in gate:
                try:
                    max_interrupts = int(gate.get("max_interrupts_per_run"))
                except (TypeError, ValueError):
                    max_interrupts = 0
                if max_interrupts < 1:
                    errors.append(f"hitl_policy.gates.{gate_id}.max_interrupts_per_run must be a positive integer")
            if "selection_mode" in gate and isinstance(gate.get("selection_mode"), str) and gate["selection_mode"] not in HITL_SELECTION_MODES:
                errors.append(f"hitl_policy.gates.{gate_id}.selection_mode is unsupported")

            allowed_actions = gate.get("allowed_actions", [])
            if not isinstance(allowed_actions, list) or not all(isinstance(action, str) for action in allowed_actions):
                errors.append(f"hitl_policy.gates.{gate_id}.allowed_actions must be a list of strings")
                allowed_actions = []
            else:
                unsupported = sorted(set(allowed_actions) - HITL_ACTIONS)
                if unsupported:
                    errors.append(f"hitl_policy.gates.{gate_id}.allowed_actions unsupported: {', '.join(unsupported)}")
                if mode == "choice" and "approve_selected" not in allowed_actions:
                    errors.append(f"hitl_policy.gates.{gate_id}.allowed_actions must include approve_selected for choice gates")
                if mode == "approval" and "approve" not in allowed_actions:
                    errors.append(f"hitl_policy.gates.{gate_id}.allowed_actions must include approve for approval gates")
            default_action = gate.get("default_action")
            if isinstance(default_action, str) and allowed_actions and default_action not in allowed_actions:
                errors.append(f"hitl_policy.gates.{gate_id}.default_action must be in allowed_actions")

            routes = gate.get("routes", {})
            if routes is not None and not isinstance(routes, dict):
                errors.append(f"hitl_policy.gates.{gate_id}.routes must be an object")
            elif isinstance(routes, dict):
                for route_name, route_target in routes.items():
                    if not isinstance(route_name, str) or not isinstance(route_target, str):
                        errors.append(f"hitl_policy.gates.{gate_id}.routes keys and values must be strings")
                    elif route_target not in node_types and route_target != "END":
                        errors.append(f"hitl_policy.gates.{gate_id}.routes.{route_name} target is unknown: {route_target}")

            options = gate.get("options", [])
            if mode == "choice":
                if not isinstance(options, list) or not options:
                    errors.append(f"hitl_policy.gates.{gate_id}.options must be a non-empty list for choice gates")
                else:
                    seen_option_ids: set[str] = set()
                    for option in options:
                        if not isinstance(option, dict):
                            errors.append(f"hitl_policy.gates.{gate_id}.options entries must be objects")
                            continue
                        option_id = option.get("id")
                        target_node_id = option.get("target_node_id")
                        if not isinstance(option_id, str) or not option_id:
                            errors.append(f"hitl_policy.gates.{gate_id}.options entries require string id")
                        elif option_id in seen_option_ids:
                            errors.append(f"hitl_policy.gates.{gate_id}.options duplicate id: {option_id}")
                        else:
                            seen_option_ids.add(option_id)
                        if not isinstance(target_node_id, str) or target_node_id not in node_types:
                            errors.append(f"hitl_policy.gates.{gate_id}.options.{option_id or '?'} target_node_id is unknown")
                    if (
                        pattern_type == ROUTER_RAG_AGENT_ID
                        and gate.get("selection_mode") in {"multi", "single_or_multi"}
                        and len(options) > 1
                    ):
                        errors.append(
                            f"hitl_policy.gates.{gate_id} multi-option choice gates require a sequential topology such as {PLAN_EXECUTE_RAG_AGENT_ID}"
                        )
        return errors

    def _collect_router_graph_errors(self, graph: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(graph, dict):
            return ["graph must be an object for router_rag_agent"]

        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return ["graph.nodes and graph.edges must be lists"]

        expected_nodes = {
            "context_loader": "context_loader",
            "router": "router",
            "retrieval_worker": "retrieval_worker",
            "memory_worker": "memory_worker",
            "timeline_worker": "timeline_worker",
            "web_worker": "web_worker",
            "direct_answer": "direct_answer",
            "synthesizer": "synthesizer",
            "finalizer": "finalizer",
        }
        actual_nodes: dict[str, str] = {}
        for node in nodes:
            if not isinstance(node, dict):
                errors.append("graph node entries must be objects")
                continue
            node_id = node.get("id")
            node_type = node.get("type")
            if isinstance(node_id, str) and isinstance(node_type, str):
                actual_nodes[node_id] = node_type
        if graph.get("hitl_compiled"):
            return self._collect_hitl_compiled_graph_errors(
                graph,
                expected_nodes=expected_nodes,
                pattern_type=ROUTER_RAG_AGENT_ID,
            )

        if actual_nodes != expected_nodes:
            errors.append("router_rag_agent graph nodes must match the built-in Router RAG topology")

        has_start = any(edge.get("from") == "START" and edge.get("to") == "context_loader" for edge in edges if isinstance(edge, dict))
        has_end = any(edge.get("from") == "finalizer" and edge.get("to") == "END" for edge in edges if isinstance(edge, dict))
        router_edges = [
            edge for edge in edges
            if isinstance(edge, dict) and edge.get("from") == "router" and edge.get("conditional") is True
        ]
        if not has_start:
            errors.append("router_rag_agent graph must start at context_loader")
        if not has_end:
            errors.append("router_rag_agent graph must end at finalizer")
        if len(router_edges) != 1:
            errors.append("router_rag_agent graph must have one conditional router edge")
        else:
            expected_routes = {
                "document": "retrieval_worker",
                "memory": "memory_worker",
                "timeline": "timeline_worker",
                "web": "web_worker",
                "direct": "direct_answer",
                "clarify": "finalizer",
            }
            if router_edges[0].get("routes") != expected_routes:
                errors.append("router_rag_agent router routes must match the built-in route map")

        return errors

    def _collect_plan_execute_graph_errors(self, graph: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(graph, dict):
            return ["graph must be an object for plan_execute_rag_agent"]

        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return ["graph.nodes and graph.edges must be lists"]

        expected_nodes = {
            "context_loader": "context_loader",
            "planner": "planner",
            "direct_answer": "direct_answer",
            "retrieval_worker": "retrieval_worker",
            "memory_worker": "memory_worker",
            "timeline_worker": "timeline_worker",
            "web_worker": "web_worker",
            "synthesizer": "synthesizer",
            "finalizer": "finalizer",
        }
        actual_nodes: dict[str, str] = {}
        for node in nodes:
            if not isinstance(node, dict):
                errors.append("graph node entries must be objects")
                continue
            node_id = node.get("id")
            node_type = node.get("type")
            if isinstance(node_id, str) and isinstance(node_type, str):
                actual_nodes[node_id] = node_type
        if graph.get("hitl_compiled"):
            return self._collect_hitl_compiled_graph_errors(
                graph,
                expected_nodes=expected_nodes,
                pattern_type=PLAN_EXECUTE_RAG_AGENT_ID,
            )
        if actual_nodes != expected_nodes:
            errors.append("plan_execute_rag_agent graph nodes must match the built-in Plan-and-Execute RAG topology")

        expected_edges = [
            {"from": "START", "to": "context_loader"},
            {"from": "context_loader", "to": "planner"},
            {
                "from": "planner",
                "conditional": True,
                "routes": {
                    "execute": "retrieval_worker",
                    "direct": "direct_answer",
                    "clarify": "finalizer",
                },
            },
            {"from": "direct_answer", "to": "finalizer"},
            {"from": "retrieval_worker", "to": "memory_worker"},
            {"from": "memory_worker", "to": "timeline_worker"},
            {"from": "timeline_worker", "to": "web_worker"},
            {"from": "web_worker", "to": "synthesizer"},
            {"from": "synthesizer", "to": "finalizer"},
            {"from": "finalizer", "to": "END"},
        ]
        if edges != expected_edges:
            errors.append("plan_execute_rag_agent graph edges must match the built-in fixed execution topology")

        return errors

    def _collect_evaluator_replanner_graph_errors(self, graph: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(graph, dict):
            return ["graph must be an object for evaluator_replanner_rag_agent"]

        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return ["graph.nodes and graph.edges must be lists"]

        expected_nodes = {
            "context_loader": "context_loader",
            "planner": "planner",
            "direct_answer": "direct_answer",
            "retrieval_worker": "retrieval_worker",
            "memory_worker": "memory_worker",
            "timeline_worker": "timeline_worker",
            "web_worker": "web_worker",
            "evidence_evaluator": "evidence_evaluator",
            "replanner": "replanner",
            "synthesizer": "synthesizer",
            "finalizer": "finalizer",
        }
        actual_nodes: dict[str, str] = {}
        for node in nodes:
            if not isinstance(node, dict):
                errors.append("graph node entries must be objects")
                continue
            node_id = node.get("id")
            node_type = node.get("type")
            if isinstance(node_id, str) and isinstance(node_type, str):
                actual_nodes[node_id] = node_type
        if graph.get("hitl_compiled"):
            return self._collect_hitl_compiled_graph_errors(
                graph,
                expected_nodes=expected_nodes,
                pattern_type=EVALUATOR_REPLANNER_RAG_AGENT_ID,
            )
        if actual_nodes != expected_nodes:
            errors.append("evaluator_replanner_rag_agent graph nodes must match the built-in Evaluator/Replanner RAG topology")

        expected_edges = [
            {"from": "START", "to": "context_loader"},
            {"from": "context_loader", "to": "planner"},
            {
                "from": "planner",
                "conditional": True,
                "routes": {
                    "execute": "retrieval_worker",
                    "direct": "direct_answer",
                    "clarify": "finalizer",
                },
            },
            {"from": "direct_answer", "to": "finalizer"},
            {"from": "retrieval_worker", "to": "memory_worker"},
            {"from": "memory_worker", "to": "timeline_worker"},
            {"from": "timeline_worker", "to": "web_worker"},
            {"from": "web_worker", "to": "evidence_evaluator"},
            {
                "from": "evidence_evaluator",
                "conditional": True,
                "routes": {
                    "answer": "synthesizer",
                    "replan": "replanner",
                    "answer_budget_exhausted": "synthesizer",
                },
            },
            {"from": "replanner", "to": "retrieval_worker"},
            {"from": "synthesizer", "to": "finalizer"},
            {"from": "finalizer", "to": "END"},
        ]
        if edges != expected_edges:
            errors.append("evaluator_replanner_rag_agent graph edges must match the built-in fixed evaluation topology")

        return errors

    def _collect_hitl_compiled_graph_errors(
        self,
        graph: Dict[str, Any],
        *,
        expected_nodes: dict[str, str],
        pattern_type: str,
    ) -> list[str]:
        """Validate materialized HITL overlays without hard-coding one gate topology."""

        errors: list[str] = []
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return ["graph.nodes and graph.edges must be lists"]

        actual_nodes: dict[str, str] = {}
        for node in nodes:
            if not isinstance(node, dict):
                errors.append("graph node entries must be objects")
                continue
            node_id = node.get("id")
            node_type = node.get("type")
            if isinstance(node_id, str) and isinstance(node_type, str):
                actual_nodes[node_id] = node_type

        for node_id, node_type in expected_nodes.items():
            if actual_nodes.get(node_id) != node_type:
                errors.append(f"{pattern_type} HITL graph missing base node: {node_id}")

        hitl_gate_ids = {
            node_id
            for node_id, node_type in actual_nodes.items()
            if node_id not in expected_nodes and node_type == "hitl_gate"
        }
        unexpected_nodes = sorted(
            node_id
            for node_id, node_type in actual_nodes.items()
            if node_id not in expected_nodes and node_type != "hitl_gate"
        )
        if unexpected_nodes:
            errors.append(f"{pattern_type} HITL graph contains unsupported non-HITL nodes: {', '.join(unexpected_nodes)}")
        if not hitl_gate_ids:
            errors.append(f"{pattern_type} HITL graph must include at least one hitl_gate node")

        valid_sources = set(actual_nodes) | {"START"}
        valid_targets = set(actual_nodes) | {"END"}
        conditional_gate_sources: set[str] = set()
        gate_incoming: set[str] = set()
        for edge in edges:
            if not isinstance(edge, dict):
                errors.append("graph edge entries must be objects")
                continue
            source = edge.get("from")
            if source not in valid_sources:
                errors.append(f"{pattern_type} HITL graph edge source is unknown: {source}")

            if edge.get("conditional"):
                routes = edge.get("routes")
                if not isinstance(routes, dict) or not routes:
                    errors.append(f"{pattern_type} HITL graph conditional edge must define routes")
                    continue
                if source in hitl_gate_ids:
                    conditional_gate_sources.add(str(source))
                for route_name, route_target in routes.items():
                    if not isinstance(route_name, str) or not isinstance(route_target, str):
                        errors.append(f"{pattern_type} HITL graph route keys and values must be strings")
                    elif route_target not in valid_targets:
                        errors.append(f"{pattern_type} HITL graph route target is unknown: {route_target}")
                    elif route_target in hitl_gate_ids:
                        gate_incoming.add(route_target)
                continue

            target = edge.get("to")
            if target not in valid_targets:
                errors.append(f"{pattern_type} HITL graph edge target is unknown: {target}")
            elif target in hitl_gate_ids:
                gate_incoming.add(target)

        missing_gate_edges = sorted(hitl_gate_ids - conditional_gate_sources)
        if missing_gate_edges:
            errors.append(f"{pattern_type} HITL gates missing conditional route edges: {', '.join(missing_gate_edges)}")
        unreachable_gates = sorted(hitl_gate_ids - gate_incoming)
        if unreachable_gates:
            errors.append(f"{pattern_type} HITL gates are not reachable from the base graph: {', '.join(unreachable_gates)}")
        return errors


class GenericGraphValidator:
    """Catalog-backed validator for schema v2 graph specs."""

    def collect_errors(self, spec: Dict[str, Any]) -> list[str]:
        errors: list[str] = []
        if not isinstance(spec, dict):
            return ["spec must be an object"]
        if spec.get("schema_version") != 2:
            errors.append("schema_version must be 2")

        pattern_type = spec.get("pattern_type")
        if not isinstance(pattern_type, str) or not pattern_type:
            errors.append("pattern_type must be a non-empty string")

        config = spec.get("config")
        if not isinstance(config, dict):
            errors.append("config must be an object")
            return errors
        errors.extend(self._collect_config_errors(config, pattern_type))

        allowed_tool_ids = config.get("allowed_tool_ids", [])
        known_tool_ids = _known_tool_ids()
        if not isinstance(allowed_tool_ids, list) or not all(isinstance(item, str) for item in allowed_tool_ids):
            errors.append("allowed_tool_ids must be a list of strings")
            allowed_tool_ids = []
        else:
            unknown_tool_ids = sorted(set(allowed_tool_ids) - known_tool_ids)
            if unknown_tool_ids:
                errors.append(f"unknown allowed_tool_ids: {', '.join(unknown_tool_ids)}")
            required_tool_ids = PATTERN_REQUIRED_TOOL_IDS.get(pattern_type, set())
            missing_tool_ids = sorted(required_tool_ids - set(allowed_tool_ids))
            if missing_tool_ids:
                errors.append(f"{pattern_type} missing required allowed_tool_ids: {', '.join(missing_tool_ids)}")

        graph = config.get("graph")
        if not isinstance(graph, dict):
            errors.append("graph must be an object")
            return errors
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            errors.append("graph.nodes and graph.edges must be lists")
            return errors
        errors.extend(TemplateValidator()._collect_hitl_policy_errors(config.get("hitl_policy", {}), pattern_type, graph))

        node_catalog = get_node_catalog()
        catalog_errors = collect_node_catalog_errors(node_catalog)
        if catalog_errors:
            return [f"node catalog incompatible: {error}" for error in catalog_errors]
        route_registry = get_route_function_registry()
        route_registry_errors = collect_route_function_registry_errors(route_registry)
        if route_registry_errors:
            return [f"route function registry incompatible: {error}" for error in route_registry_errors]
        tool_contracts = tool_contracts_by_id()
        tool_contract_errors = collect_tool_contract_metadata_errors(
            [record for records in tool_contracts.values() for record in records]
        )
        if tool_contract_errors:
            return [f"tool contract registry incompatible: {error}" for error in tool_contract_errors]
        errors.extend(self._collect_catalog_route_function_errors(node_catalog, route_registry))
        errors.extend(self._collect_catalog_tool_contract_errors(node_catalog, tool_contracts))
        node_ids: set[str] = set()
        node_types_by_id: dict[str, str] = {}
        node_type_counts: dict[str, int] = {}
        graph_supported_tool_ids: set[str] = set()
        known_types = known_node_types()

        for node in nodes:
            if not isinstance(node, dict):
                errors.append("graph node entries must be objects")
                continue
            node_id = node.get("id")
            node_type = node.get("type")
            if not isinstance(node_id, str) or not node_id:
                errors.append("graph node entries require non-empty string id")
                continue
            if node_id in {"START", "END"}:
                errors.append(f"graph node id is reserved: {node_id}")
            if node_id in node_ids:
                errors.append(f"duplicate graph node id: {node_id}")
            node_ids.add(node_id)

            if not isinstance(node_type, str) or not node_type:
                errors.append(f"graph node {node_id} requires non-empty string type")
                continue
            if node_type not in known_types:
                errors.append(f"graph node {node_id} has unknown type: {node_type}")
                continue
            node_types_by_id[node_id] = node_type
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
            graph_supported_tool_ids.update(node_type_allowed_tool_contract_ids(node_type))
            errors.extend(self._collect_node_contract_errors(node_id, node_type, node, node_catalog))

            node_tool_ids = node.get("tool_contract_ids", [])
            if node_tool_ids in (None, []):
                continue
            if not isinstance(node_tool_ids, list) or not all(isinstance(item, str) for item in node_tool_ids):
                errors.append(f"graph node {node_id}.tool_contract_ids must be a list of strings")
                continue
            unsupported_for_node = sorted(set(node_tool_ids) - node_type_allowed_tool_contract_ids(node_type))
            if unsupported_for_node:
                errors.append(
                    f"graph node {node_id} type {node_type} does not allow tool contracts: {', '.join(unsupported_for_node)}"
                )
            disabled_for_graph = sorted(set(node_tool_ids) - set(allowed_tool_ids))
            if disabled_for_graph:
                errors.append(f"graph node {node_id} uses disabled tool contracts: {', '.join(disabled_for_graph)}")

        disallowed_enabled_tools = sorted(set(allowed_tool_ids) - graph_supported_tool_ids)
        if disallowed_enabled_tools:
            errors.append(
                "allowed_tool_ids are not supported by any node in this graph: "
                + ", ".join(disallowed_enabled_tools)
            )

        for node_type, count in sorted(node_type_counts.items()):
            max_instances = (node_catalog.get(node_type) or {}).get("max_instances")
            if isinstance(max_instances, int) and not isinstance(max_instances, bool) and count > max_instances:
                errors.append(f"graph has {count} nodes of type {node_type}; maximum allowed is {max_instances}")

        valid_sources = set(node_ids) | {"START"}
        valid_targets = set(node_ids) | {"END"}
        adjacency: dict[str, set[str]] = {}
        for edge in edges:
            if not isinstance(edge, dict):
                errors.append("graph edge entries must be objects")
                continue
            source = edge.get("from")
            if not isinstance(source, str) or source not in valid_sources:
                errors.append(f"graph edge source is unknown: {source}")
                continue

            if edge.get("conditional"):
                routes = edge.get("routes")
                if not isinstance(routes, dict) or not routes:
                    errors.append(f"graph conditional edge from {source} must define routes")
                    continue
                route_fn = edge.get("route_fn")
                source_type = node_types_by_id.get(source)
                if not isinstance(route_fn, str) or not route_fn:
                    errors.append(f"graph conditional edge from {source} must declare route_fn")
                elif route_fn not in known_route_function_ids():
                    errors.append(f"graph conditional edge from {source} has unknown route_fn: {route_fn}")
                elif source_type and not route_function_allowed_for_node_type(route_fn, source_type):
                    errors.append(f"route_fn {route_fn} is not allowed from node {source} type {source_type}")
                labels = route_function_labels(route_fn) if isinstance(route_fn, str) else None
                for route_name, route_target in routes.items():
                    if not isinstance(route_name, str) or not isinstance(route_target, str):
                        errors.append(f"graph conditional edge from {source} routes keys and values must be strings")
                        continue
                    if labels is not None and route_name not in labels:
                        errors.append(f"graph conditional edge from {source} has invalid route label: {route_name}")
                    if route_target not in valid_targets:
                        errors.append(f"graph conditional edge from {source} route {route_name} target is unknown: {route_target}")
                        continue
                    adjacency.setdefault(source, set()).add(route_target)
                    errors.extend(self._collect_edge_compatibility_errors(source, route_target, node_types_by_id, node_catalog))
                continue

            target = edge.get("to")
            if not isinstance(target, str) or target not in valid_targets:
                errors.append(f"graph edge target is unknown: {target}")
                continue
            adjacency.setdefault(source, set()).add(target)
            errors.extend(self._collect_edge_compatibility_errors(source, target, node_types_by_id, node_catalog))

        errors.extend(self._collect_loop_policy_errors(config.get("loop_policy"), adjacency, node_ids, node_types_by_id, node_catalog))
        errors.extend(self._collect_reachability_errors(adjacency, node_ids))
        return errors

    def _collect_config_errors(self, config: Dict[str, Any], pattern_type: Any) -> list[str]:
        errors: list[str] = []
        allowed_keys = ALLOWED_ROUTER_RAG_CONFIG_KEYS | {"context_policy", "loop_policy"}
        unknown_keys = sorted(set(config) - allowed_keys)
        if unknown_keys:
            errors.append(f"unknown config keys: {', '.join(unknown_keys)}")

        for key in ("use_web_search", "use_reranker"):
            if key in config and not isinstance(config[key], bool):
                errors.append(f"{key} must be a boolean")

        if "replans" in config:
            replans = config.get("replans")
            if not isinstance(replans, int):
                errors.append("replans must be an integer")
            elif replans < 1 or replans > REPLANS_LIMIT:
                errors.append(f"replans must be between 1 and {REPLANS_LIMIT}")

        system_role = config.get("system_role", "")
        if not isinstance(system_role, str) or len(system_role) > MAX_SYSTEM_ROLE_CHARS:
            errors.append(f"system_role must be a string up to {MAX_SYSTEM_ROLE_CHARS} characters")

        custom_instructions = config.get("custom_instructions", "")
        if not isinstance(custom_instructions, str) or len(custom_instructions) > MAX_CUSTOM_INSTRUCTIONS_CHARS:
            errors.append(f"custom_instructions must be a string up to {MAX_CUSTOM_INSTRUCTIONS_CHARS} characters")

        tool_instructions = config.get("tool_instructions", {})
        if not isinstance(tool_instructions, dict):
            errors.append("tool_instructions must be an object")
        elif not all(isinstance(k, str) and isinstance(v, str) for k, v in tool_instructions.items()):
            errors.append("tool_instructions keys and values must be strings")

        prefetch_policy = config.get("prefetch_policy", {})
        if not isinstance(prefetch_policy, dict):
            errors.append("prefetch_policy must be an object")
        elif set(prefetch_policy) - {"enabled"}:
            errors.append("prefetch_policy only supports the enabled key")
        elif "enabled" in prefetch_policy and not isinstance(prefetch_policy["enabled"], bool):
            errors.append("prefetch_policy.enabled must be a boolean")

        context_policy = config.get("context_policy", {})
        if not isinstance(context_policy, dict):
            errors.append("context_policy must be an object")
        else:
            unknown_context_keys = sorted(
                set(context_policy)
                - {
                    "evidence_packet_limit",
                    "evidence_packet_content_limit",
                    "final_prompt_assembly",
                    "evidence_dedupe",
                    "evidence_compression",
                    "final_context_char_limit",
                }
            )
            if unknown_context_keys:
                errors.append(f"context_policy has unknown keys: {', '.join(unknown_context_keys)}")
            for key in ("evidence_packet_limit", "evidence_packet_content_limit", "final_context_char_limit"):
                if key in context_policy:
                    try:
                        value = int(context_policy[key])
                    except (TypeError, ValueError):
                        value = 0
                    if value < 1:
                        errors.append(f"context_policy.{key} must be a positive integer")
            if "final_prompt_assembly" in context_policy and not isinstance(context_policy["final_prompt_assembly"], str):
                errors.append("context_policy.final_prompt_assembly must be a string")
            if "evidence_dedupe" in context_policy and not isinstance(context_policy["evidence_dedupe"], bool):
                errors.append("context_policy.evidence_dedupe must be a boolean")
            if "evidence_compression" in context_policy and not isinstance(context_policy["evidence_compression"], str):
                errors.append("context_policy.evidence_compression must be a string")
        return errors

    def _collect_catalog_route_function_errors(
        self,
        node_catalog: Dict[str, Dict[str, Any]],
        route_registry: Dict[str, Dict[str, Any]],
    ) -> list[str]:
        errors: list[str] = []
        for node_type, metadata in sorted(node_catalog.items()):
            allowed_route_functions = metadata.get("allowed_route_functions") or []
            for route_fn in allowed_route_functions:
                route_metadata = route_registry.get(route_fn)
                if not isinstance(route_metadata, dict):
                    errors.append(f"node catalog type {node_type} references unknown route_fn: {route_fn}")
                    continue
                allowed_source_types = set(route_metadata.get("allowed_source_types") or [])
                if node_type not in allowed_source_types:
                    errors.append(
                        f"node catalog type {node_type} allows route_fn {route_fn}, "
                        "but route registry does not allow that source type"
                    )
        return errors

    def _collect_catalog_tool_contract_errors(
        self,
        node_catalog: Dict[str, Dict[str, Any]],
        tool_contracts: Dict[str, list[Dict[str, Any]]],
    ) -> list[str]:
        errors: list[str] = []
        known_contract_ids = set(tool_contracts)
        for node_type, metadata in sorted(node_catalog.items()):
            node_capabilities = set(metadata.get("capabilities") or [])
            for contract_id in metadata.get("allowed_tool_contract_ids") or []:
                if contract_id not in known_contract_ids:
                    errors.append(f"node catalog type {node_type} references unknown tool contract: {contract_id}")
                    continue
                compatible = False
                for contract in tool_contracts.get(contract_id) or []:
                    allowed_node_types = set(contract.get("allowed_node_types") or [])
                    required_capabilities = set(contract.get("required_node_capabilities") or [])
                    if node_type in allowed_node_types or node_capabilities.intersection(required_capabilities):
                        compatible = True
                        break
                if not compatible:
                    errors.append(
                        f"node catalog type {node_type} allows tool contract {contract_id}, "
                        "but tool registry does not allow that node type or capability"
                    )
        return errors

    def _collect_node_contract_errors(
        self,
        node_id: str,
        node_type: str,
        node: Dict[str, Any],
        node_catalog: Dict[str, Dict[str, Any]],
    ) -> list[str]:
        errors: list[str] = []
        metadata = node_catalog.get(node_type) or {}
        for key in ("state_reads", "state_writes", "prompt_slots"):
            if key not in node:
                continue
            value = node.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                errors.append(f"graph node {node_id}.{key} must be a list of non-empty strings")
                continue
            allowed_values = set(metadata.get(key) or [])
            unsupported = sorted(set(value) - allowed_values)
            if unsupported:
                errors.append(
                    f"graph node {node_id}.{key} includes unsupported values for type {node_type}: "
                    + ", ".join(unsupported)
                )

        context_policy = node.get("context_policy")
        if context_policy is not None:
            if not isinstance(context_policy, dict):
                errors.append(f"graph node {node_id}.context_policy must be an object")
            else:
                unknown_keys = sorted(set(context_policy) - {"mode", "input_budget", "output_budget"})
                if unknown_keys:
                    errors.append(f"graph node {node_id}.context_policy has unknown keys: {', '.join(unknown_keys)}")
                catalog_policy = metadata.get("context_policy") if isinstance(metadata.get("context_policy"), dict) else {}
                for key in ("mode", "input_budget", "output_budget"):
                    value = context_policy.get(key)
                    if value is None:
                        continue
                    if not isinstance(value, str) or not value:
                        errors.append(f"graph node {node_id}.context_policy.{key} must be a non-empty string")
                    elif catalog_policy.get(key) and value != catalog_policy.get(key):
                        errors.append(
                            f"graph node {node_id}.context_policy.{key} must match catalog value "
                            f"{catalog_policy.get(key)} for type {node_type}"
                        )

        observability = node.get("observability")
        if observability is not None:
            if not isinstance(observability, dict):
                errors.append(f"graph node {node_id}.observability must be an object")
            else:
                unknown_keys = sorted(set(observability) - {"span_kind", "event_prefix", "summary_fields", "raw_payload"})
                if unknown_keys:
                    errors.append(f"graph node {node_id}.observability has unknown keys: {', '.join(unknown_keys)}")
                catalog_observability = (
                    metadata.get("observability") if isinstance(metadata.get("observability"), dict) else {}
                )
                for key in ("span_kind", "event_prefix", "raw_payload"):
                    value = observability.get(key)
                    if value is None:
                        continue
                    if not isinstance(value, str) or not value:
                        errors.append(f"graph node {node_id}.observability.{key} must be a non-empty string")
                    elif catalog_observability.get(key) and value != catalog_observability.get(key):
                        errors.append(
                            f"graph node {node_id}.observability.{key} must match catalog value "
                            f"{catalog_observability.get(key)} for type {node_type}"
                        )
                summary_fields = observability.get("summary_fields")
                if summary_fields is not None:
                    if not isinstance(summary_fields, list) or not all(
                        isinstance(item, str) and item for item in summary_fields
                    ):
                        errors.append(
                            f"graph node {node_id}.observability.summary_fields must be a list of non-empty strings"
                        )
                    else:
                        allowed_summary_fields = set(catalog_observability.get("summary_fields") or [])
                        unsupported = sorted(set(summary_fields) - allowed_summary_fields)
                        if unsupported:
                            errors.append(
                                f"graph node {node_id}.observability.summary_fields includes unsupported values "
                                f"for type {node_type}: {', '.join(unsupported)}"
                            )
        return errors

    def _collect_edge_compatibility_errors(
        self,
        source: str,
        target: str,
        node_types_by_id: dict[str, str],
        node_catalog: Dict[str, Dict[str, Any]],
    ) -> list[str]:
        if source == "START" or target == "END":
            return []
        source_type = node_types_by_id.get(source)
        target_type = node_types_by_id.get(target)
        if not source_type or not target_type:
            return []
        allowed_children = set((node_catalog.get(source_type) or {}).get("allowed_child_types") or [])
        if target_type not in allowed_children:
            return [f"node {source} type {source_type} cannot connect to {target} type {target_type}"]
        return []

    def _collect_loop_policy_errors(
        self,
        loop_policy: Any,
        adjacency: dict[str, set[str]],
        node_ids: set[str],
        node_types_by_id: dict[str, str],
        node_catalog: Dict[str, Dict[str, Any]],
    ) -> list[str]:
        errors: list[str] = []
        has_cycle = self._graph_has_cycle(adjacency, node_ids)
        if not has_cycle and loop_policy in (None, {}):
            return errors
        if not isinstance(loop_policy, dict):
            return ["graph contains cycles and requires loop_policy"]

        unknown_keys = sorted(set(loop_policy) - {"max_total_visits", "default_max_node_visits", "node_visit_limits"})
        if unknown_keys:
            errors.append(f"loop_policy has unknown keys: {', '.join(unknown_keys)}")

        max_total_visits = loop_policy.get("max_total_visits")
        try:
            max_total_visits_value = int(max_total_visits)
        except (TypeError, ValueError):
            max_total_visits_value = 0
        if max_total_visits_value < max(1, len(node_ids)):
            errors.append("loop_policy.max_total_visits must be an integer at least the number of graph nodes")

        default_max = loop_policy.get("default_max_node_visits", 1)
        try:
            default_max_value = int(default_max)
        except (TypeError, ValueError):
            default_max_value = 0
        if default_max_value < 1:
            errors.append("loop_policy.default_max_node_visits must be a positive integer")

        node_visit_limits = loop_policy.get("node_visit_limits", {})
        if not isinstance(node_visit_limits, dict):
            errors.append("loop_policy.node_visit_limits must be an object")
            return errors

        for node_id, raw_limit in node_visit_limits.items():
            if not isinstance(node_id, str) or node_id not in node_ids:
                errors.append(f"loop_policy.node_visit_limits has unknown node: {node_id}")
                continue
            try:
                limit = int(raw_limit)
            except (TypeError, ValueError):
                limit = 0
            if limit < 1:
                errors.append(f"loop_policy.node_visit_limits.{node_id} must be a positive integer")
                continue
            node_type = node_types_by_id.get(node_id)
            metadata = node_catalog.get(node_type or "") or {}
            limits = metadata.get("limits") if isinstance(metadata.get("limits"), dict) else {}
            try:
                catalog_default = int(limits.get("default_max_visits", default_max_value))
            except (TypeError, ValueError):
                catalog_default = default_max_value
            if limit > max(default_max_value, catalog_default):
                errors.append(
                    f"loop_policy.node_visit_limits.{node_id} exceeds allowed max for node type {node_type}"
                )

        if has_cycle and max_total_visits_value > 0 and default_max_value > 0:
            effective_total = 0
            for node_id in node_ids:
                raw_limit = node_visit_limits.get(node_id, default_max_value)
                try:
                    effective_total += max(1, int(raw_limit))
                except (TypeError, ValueError):
                    effective_total += default_max_value
            if max_total_visits_value > effective_total:
                errors.append("loop_policy.max_total_visits cannot exceed the sum of per-node visit limits")
        return errors

    def _graph_has_cycle(self, adjacency: dict[str, set[str]], node_ids: set[str]) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in {"START", "END"}:
                return False
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for target in adjacency.get(node) or set():
                if target in node_ids and visit(target):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node_id) for node_id in sorted(node_ids))

    def _collect_reachability_errors(self, adjacency: dict[str, set[str]], node_ids: set[str]) -> list[str]:
        errors: list[str] = []
        if "START" not in adjacency:
            errors.append("graph must have an edge from START")
            return errors
        visited: set[str] = set()
        stack = list(adjacency.get("START") or [])
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node == "END":
                continue
            stack.extend(adjacency.get(node) or [])
        unreachable = sorted(node_ids - visited)
        if unreachable:
            errors.append(f"graph contains unreachable nodes: {', '.join(unreachable)}")
        if "END" not in visited:
            errors.append("graph must be able to reach END from START")
        return errors


class TemplateResolver:
    """Freeze the effective built-in agent pattern config for an agent run."""

    def __init__(self, validator: Optional[TemplateValidator] = None):
        self.validator = validator or TemplateValidator()

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
            for key in ALLOWED_ROUTER_RAG_CONFIG_KEYS:
                if key == "replans" and resolved.get("pattern_type") != EVALUATOR_REPLANNER_RAG_AGENT_ID:
                    continue
                if key in source and source[key] is not None:
                    config[key] = source[key]

        resolved["config"] = config
        self.validator.validate(resolved)
        return resolved
