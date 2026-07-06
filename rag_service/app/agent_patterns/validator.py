from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from app.agent.tool_registry import known_tool_contract_ids, tool_contracts_by_id
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
}


class TemplateValidator:
    """Validator for supported built-in agent pattern schemas."""

    def validate(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        errors = self.collect_errors(spec)
        result = {"valid": not errors, "errors": errors}
        if errors:
            raise TemplateValidationError("; ".join(errors))
        return result

    def collect_errors(self, spec: Dict[str, Any]) -> list[str]:
        errors: list[str] = []
        if not isinstance(spec, dict):
            return ["spec must be an object"]

        if spec.get("schema_version") != 1:
            errors.append("schema_version must be 1")
        pattern_type = spec.get("pattern_type")
        if pattern_type not in SUPPORTED_BUILTIN_TEMPLATE_IDS:
            errors.append(f"pattern_type must be one of: {', '.join(sorted(SUPPORTED_BUILTIN_TEMPLATE_IDS))}")

        config = spec.get("config")
        if not isinstance(config, dict):
            errors.append("config must be an object")
            return errors

        unknown_keys = sorted(set(config) - ALLOWED_ROUTER_RAG_CONFIG_KEYS)
        if unknown_keys:
            errors.append(f"unknown config keys: {', '.join(unknown_keys)}")

        for key in ("use_web_search", "use_reranker"):
            if key in config and not isinstance(config[key], bool):
                errors.append(f"{key} must be a boolean")

        if "replans" in config and pattern_type != EVALUATOR_REPLANNER_RAG_AGENT_ID:
            errors.append("replans is only supported for evaluator_replanner_rag_agent")
        elif "replans" in config:
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

        allowed_tool_ids = config.get("allowed_tool_ids", [])
        known_tool_ids = _known_tool_ids()
        if not isinstance(allowed_tool_ids, list) or not all(isinstance(item, str) for item in allowed_tool_ids):
            errors.append("allowed_tool_ids must be a list of strings")
        else:
            unknown_tool_ids = sorted(set(allowed_tool_ids) - known_tool_ids)
            if unknown_tool_ids:
                errors.append(f"unknown allowed_tool_ids: {', '.join(unknown_tool_ids)}")
            required_tool_ids = PATTERN_REQUIRED_TOOL_IDS.get(pattern_type, set())
            if required_tool_ids:
                missing_tool_ids = sorted(required_tool_ids - set(allowed_tool_ids))
                if missing_tool_ids:
                    errors.append(f"{pattern_type} missing required allowed_tool_ids: {', '.join(missing_tool_ids)}")
                errors.extend(self._collect_tool_permission_errors(pattern_type, set(allowed_tool_ids)))

        prefetch_policy = config.get("prefetch_policy", {})
        if not isinstance(prefetch_policy, dict):
            errors.append("prefetch_policy must be an object")
        elif set(prefetch_policy) - {"enabled"}:
            errors.append("prefetch_policy only supports the enabled key in v1")
        elif "enabled" in prefetch_policy and not isinstance(prefetch_policy["enabled"], bool):
            errors.append("prefetch_policy.enabled must be a boolean")

        errors.extend(self._collect_hitl_policy_errors(config.get("hitl_policy", {}), pattern_type, config.get("graph")))

        if pattern_type == ROUTER_RAG_AGENT_ID:
            errors.extend(self._collect_router_graph_errors(config.get("graph")))
        elif pattern_type == PLAN_EXECUTE_RAG_AGENT_ID:
            errors.extend(self._collect_plan_execute_graph_errors(config.get("graph")))
        elif pattern_type == EVALUATOR_REPLANNER_RAG_AGENT_ID:
            errors.extend(self._collect_evaluator_replanner_graph_errors(config.get("graph")))

        return errors

    def report(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structured validation report for admin/debug API consumers."""

        errors = self.collect_errors(spec)
        config = spec.get("config") if isinstance(spec, dict) else {}
        config = config if isinstance(config, dict) else {}
        allowed_tool_ids = config.get("allowed_tool_ids") if isinstance(config.get("allowed_tool_ids"), list) else []
        known_tool_ids = _known_tool_ids()
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": [],
            "schema_version": spec.get("schema_version") if isinstance(spec, dict) else None,
            "pattern_type": spec.get("pattern_type") if isinstance(spec, dict) else None,
            "supported_pattern_types": sorted(SUPPORTED_BUILTIN_TEMPLATE_IDS),
            "allowed_tool_ids": allowed_tool_ids,
            "required_tool_ids": sorted(PATTERN_REQUIRED_TOOL_IDS.get(spec.get("pattern_type"), ROUTER_RAG_REQUIRED_TOOL_IDS)),
            "missing_required_tool_ids": sorted(
                PATTERN_REQUIRED_TOOL_IDS.get(spec.get("pattern_type"), ROUTER_RAG_REQUIRED_TOOL_IDS) - set(allowed_tool_ids)
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
        unknown_policy_keys = sorted(set(hitl_policy) - {"enabled", "gates"})
        if unknown_policy_keys:
            errors.append(f"hitl_policy only supports keys: enabled, gates; unknown: {', '.join(unknown_policy_keys)}")
        if "enabled" in hitl_policy and not isinstance(hitl_policy["enabled"], bool):
            errors.append("hitl_policy.enabled must be a boolean")
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
