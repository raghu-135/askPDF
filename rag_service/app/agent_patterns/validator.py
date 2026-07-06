from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from app.agent.tool_registry import known_tool_contract_ids, tool_contracts_by_id
from app.agent_patterns.templates import (
    ALLOWED_ROUTER_RAG_CONFIG_KEYS,
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
    MAX_MAX_ITERATIONS,
    MAX_SYSTEM_ROLE_CHARS,
    MIN_MAX_ITERATIONS,
)


class TemplateValidationError(ValueError):
    """Raised when an agent pattern template spec is invalid."""


def _known_tool_ids() -> set[str]:
    return known_tool_contract_ids()


PATTERN_REQUIRED_TOOL_IDS = {
    ROUTER_RAG_AGENT_ID: ROUTER_RAG_REQUIRED_TOOL_IDS,
    PLAN_EXECUTE_RAG_AGENT_ID: PLAN_EXECUTE_RAG_REQUIRED_TOOL_IDS,
}

PATTERN_NODE_TOOL_REQUIREMENTS = {
    ROUTER_RAG_AGENT_ID: ROUTER_RAG_NODE_TOOL_REQUIREMENTS,
    PLAN_EXECUTE_RAG_AGENT_ID: PLAN_EXECUTE_RAG_NODE_TOOL_REQUIREMENTS,
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

        max_iterations = config.get("max_iterations")
        if not isinstance(max_iterations, int):
            errors.append("max_iterations must be an integer")
        elif max_iterations < MIN_MAX_ITERATIONS or max_iterations > MAX_MAX_ITERATIONS:
            errors.append(f"max_iterations must be between {MIN_MAX_ITERATIONS} and {MAX_MAX_ITERATIONS}")

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

        errors.extend(self._collect_hitl_policy_errors(config.get("hitl_policy", {})))

        if pattern_type == ROUTER_RAG_AGENT_ID:
            errors.extend(self._collect_router_graph_errors(config.get("graph")))
        elif pattern_type == PLAN_EXECUTE_RAG_AGENT_ID:
            errors.extend(self._collect_plan_execute_graph_errors(config.get("graph")))

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

    def _collect_hitl_policy_errors(self, hitl_policy: Any) -> list[str]:
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
        unknown_gates = sorted(set(gates) - {WEB_APPROVAL_GATE_ID})
        if unknown_gates:
            errors.append(f"hitl_policy.gates only supports {WEB_APPROVAL_GATE_ID}; unknown: {', '.join(unknown_gates)}")
        gate = gates.get(WEB_APPROVAL_GATE_ID)
        if gate is None:
            return errors
        if not isinstance(gate, dict):
            errors.append(f"hitl_policy.gates.{WEB_APPROVAL_GATE_ID} must be an object")
            return errors
        unknown_gate_keys = sorted(set(gate) - {"enabled", "title", "prompt", "body", "allowed_actions", "default_action"})
        if unknown_gate_keys:
            errors.append(
                f"hitl_policy.gates.{WEB_APPROVAL_GATE_ID} has unknown keys: {', '.join(unknown_gate_keys)}"
            )
        if "enabled" in gate and not isinstance(gate["enabled"], bool):
            errors.append(f"hitl_policy.gates.{WEB_APPROVAL_GATE_ID}.enabled must be a boolean")
        for key in ("title", "prompt", "body", "default_action"):
            if key in gate and not isinstance(gate[key], str):
                errors.append(f"hitl_policy.gates.{WEB_APPROVAL_GATE_ID}.{key} must be a string")
        allowed_actions = gate.get("allowed_actions", [])
        if not isinstance(allowed_actions, list) or not all(isinstance(action, str) for action in allowed_actions):
            errors.append(f"hitl_policy.gates.{WEB_APPROVAL_GATE_ID}.allowed_actions must be a list of strings")
        else:
            unsupported = sorted(set(allowed_actions) - {"approve", "continue_without"})
            if unsupported:
                errors.append(
                    f"hitl_policy.gates.{WEB_APPROVAL_GATE_ID}.allowed_actions unsupported: {', '.join(unsupported)}"
                )
            if "approve" not in allowed_actions or "continue_without" not in allowed_actions:
                errors.append(
                    f"hitl_policy.gates.{WEB_APPROVAL_GATE_ID}.allowed_actions must include approve and continue_without"
                )
        default_action = gate.get("default_action", "continue_without")
        if isinstance(default_action, str) and default_action not in {"approve", "continue_without"}:
            errors.append(f"hitl_policy.gates.{WEB_APPROVAL_GATE_ID}.default_action is unsupported")
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
        expected_hitl_nodes = {
            "context_loader": "context_loader",
            "router": "router",
            "retrieval_worker": "retrieval_worker",
            "memory_worker": "memory_worker",
            "timeline_worker": "timeline_worker",
            WEB_APPROVAL_GATE_ID: "hitl_gate",
            "web_worker": "web_worker",
            "direct_answer": "direct_answer",
            "synthesizer": "synthesizer",
            "finalizer": "finalizer",
        }
        if actual_nodes == expected_hitl_nodes:
            expected_hitl_edges = [
                {"from": "START", "to": "context_loader"},
                {"from": "context_loader", "to": "router"},
                {
                    "from": "router",
                    "conditional": True,
                    "routes": {
                        "document": "retrieval_worker",
                        "memory": "memory_worker",
                        "timeline": "timeline_worker",
                        "web": WEB_APPROVAL_GATE_ID,
                        "direct": "direct_answer",
                        "clarify": "finalizer",
                    },
                },
                {"from": "retrieval_worker", "to": "synthesizer"},
                {"from": "memory_worker", "to": "synthesizer"},
                {"from": "timeline_worker", "to": "synthesizer"},
                {
                    "from": WEB_APPROVAL_GATE_ID,
                    "conditional": True,
                    "routes": {
                        "approve": "web_worker",
                        "continue_without": "synthesizer",
                    },
                },
                {"from": "web_worker", "to": "synthesizer"},
                {"from": "direct_answer", "to": "finalizer"},
                {"from": "synthesizer", "to": "finalizer"},
                {"from": "finalizer", "to": "END"},
            ]
            if edges != expected_hitl_edges:
                errors.append("router_rag_agent HITL web graph edges must match the built-in fixed topology")
            return errors

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
                if key in source and source[key] is not None:
                    config[key] = source[key]

        resolved["config"] = config
        self.validator.validate(resolved)
        return resolved
