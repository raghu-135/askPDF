from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from app.agent_workflows.hitl_materializer import materialize_hitl_gates
from app.agent_workflows.node_catalog import get_node_type_metadata


CANONICAL_NODE_TYPE_ORDER = {
    "context_loader": 0,
    "router": 1,
    "planner": 1,
    "retrieval_worker": 2,
    "memory_worker": 3,
    "timeline_worker": 4,
    "web_worker": 5,
    "evidence_evaluator": 6,
    "replanner": 7,
    "direct_answer": 8,
    "synthesizer": 9,
    "finalizer": 10,
    "hitl_gate": 11,
}


class WorkflowMaterializer:
    """Materialize validated workflow specs before LangGraph compilation."""

    def materialize_spec(
        self,
        spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        materialized = deepcopy(spec)
        config = materialized.get("config") if isinstance(materialized.get("config"), dict) else {}
        graph_spec = config.get("graph") if isinstance(config.get("graph"), dict) else {}
        hitl_policy = config.get("hitl_policy") if isinstance(config.get("hitl_policy"), dict) else {}
        explicit_graph = self._with_explicit_route_functions(graph_spec)
        compiled_graph = materialize_hitl_gates(
            explicit_graph,
            hitl_policy=hitl_policy,
        )
        config["graph"] = self._with_catalog_node_metadata(compiled_graph)
        config["loop_policy"] = self._with_materialized_loop_policy(
            config.get("loop_policy"),
            graph_spec=config["graph"],
        )
        materialized["config"] = config
        return materialized

    def _with_materialized_loop_policy(self, loop_policy: Any, *, graph_spec: Dict[str, Any]) -> Dict[str, Any]:
        policy = dict(loop_policy) if isinstance(loop_policy, dict) else {}
        nodes = [node for node in graph_spec.get("nodes", []) if isinstance(node, dict)]
        node_count = len(nodes)
        try:
            max_total_visits = int(policy.get("max_total_visits", 0))
        except (TypeError, ValueError):
            max_total_visits = 0
        if node_count and max_total_visits < node_count:
            policy["max_total_visits"] = node_count
        node_visit_limits = policy.get("node_visit_limits")
        if isinstance(node_visit_limits, dict):
            policy["node_visit_limits"] = dict(node_visit_limits)
        elif node_visit_limits is not None:
            policy["node_visit_limits"] = {}
        return policy

    def _with_explicit_route_functions(self, graph_spec: Dict[str, Any]) -> Dict[str, Any]:
        nodes = [dict(node) for node in graph_spec.get("nodes", []) if isinstance(node, dict)]
        node_types = {
            str(node.get("id")): str(node.get("type"))
            for node in nodes
            if isinstance(node.get("id"), str) and isinstance(node.get("type"), str)
        }
        route_by_type = {
            "router": "router_route",
            "planner": "planner_route",
            "evidence_evaluator": "evaluator_route",
            "hitl_gate": "hitl_gate_route",
        }
        edges = []
        for raw_edge in graph_spec.get("edges", []):
            if not isinstance(raw_edge, dict):
                continue
            edge = dict(raw_edge)
            if edge.get("conditional") and not edge.get("route_fn"):
                route_fn = route_by_type.get(node_types.get(str(edge.get("from")), ""))
                if route_fn:
                    edge["route_fn"] = route_fn
            edges.append(edge)
        edges.sort(
            key=lambda edge: (
                self._edge_order(edge, node_types),
                1 if edge.get("conditional") else 0,
                str(edge.get("from") or ""),
                str(edge.get("to") or ""),
            )
        )
        return {**graph_spec, "nodes": nodes, "edges": edges}

    def _edge_order(self, edge: Dict[str, Any], node_types: Dict[str, str]) -> int:
        source = edge.get("from")
        if source == "START":
            return -1
        source_type = node_types.get(str(source))
        return CANONICAL_NODE_TYPE_ORDER.get(str(source_type), 100)

    def _with_catalog_node_metadata(self, graph_spec: Dict[str, Any]) -> Dict[str, Any]:
        nodes = []
        for index, raw_node in enumerate(graph_spec.get("nodes", [])):
            if not isinstance(raw_node, dict):
                continue
            node = dict(raw_node)
            node["_materialized_order"] = index
            node_type = node.get("type")
            metadata = get_node_type_metadata(str(node_type)) if isinstance(node_type, str) else {}
            display_name = metadata.get("display_name")
            category = metadata.get("category")
            if isinstance(display_name, str) and display_name:
                node["label"] = display_name
            if isinstance(category, str) and category:
                node["category"] = category
            for key in (
                "capabilities",
                "allowed_route_functions",
                "allowed_tool_contract_ids",
                "state_reads",
                "state_writes",
                "prompt_slots",
                "context_policy",
                "observability",
                "max_instances",
            ):
                if key in metadata:
                    node[key] = deepcopy(metadata[key])
            nodes.append(node)
        nodes.sort(
            key=lambda node: (
                CANONICAL_NODE_TYPE_ORDER.get(str(node.get("type")), 100),
                str(node.get("id") or ""),
                int(node.get("_materialized_order") or 0),
            )
        )
        for node in nodes:
            node.pop("_materialized_order", None)
        return {**graph_spec, "nodes": nodes}
