from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
from app.agent_patterns.templates import (
    ALLOWED_ROUTER_RAG_CONFIG_KEYS,
    ROUTER_RAG_AGENT_ID,
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
    return {
        config["id"]
        for config in TOOL_FRIENDLY_CONFIG.values()
        if isinstance(config, dict) and config.get("id")
    }


class TemplateValidator:
    """Validator for the supported built-in Router RAG Agent schema."""

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
        if pattern_type != ROUTER_RAG_AGENT_ID:
            errors.append(f"pattern_type must be {ROUTER_RAG_AGENT_ID}")

        config = spec.get("config")
        if not isinstance(config, dict):
            errors.append("config must be an object")
            return errors

        unknown_keys = sorted(set(config) - ALLOWED_ROUTER_RAG_CONFIG_KEYS)
        if unknown_keys:
            errors.append(f"unknown config keys: {', '.join(unknown_keys)}")

        for key in ("use_intent_agent", "use_web_search", "use_reranker"):
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

        prefetch_policy = config.get("prefetch_policy", {})
        if not isinstance(prefetch_policy, dict):
            errors.append("prefetch_policy must be an object")
        elif set(prefetch_policy) - {"enabled"}:
            errors.append("prefetch_policy only supports the enabled key in v1")
        elif "enabled" in prefetch_policy and not isinstance(prefetch_policy["enabled"], bool):
            errors.append("prefetch_policy.enabled must be a boolean")

        if pattern_type == ROUTER_RAG_AGENT_ID:
            errors.extend(self._collect_router_graph_errors(config.get("graph")))

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


class TemplateResolver:
    """Freeze the effective Router RAG config for an agent run."""

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
