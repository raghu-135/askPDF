from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from app.agent_workflows.enums import GraphSentinel, RouteFunctionId, WorkflowNodeType
from app.agent_workflows.hitl_materializer import materialize_hitl_gates
from app.agent_workflows.node_catalog import get_node_type_metadata
from app.agent_workflows.routes import route_function_for_edge
from app.agent_workflows.state import RouterRagState

if TYPE_CHECKING:
    from app.agent_workflows.graph import NodeRegistry


CANONICAL_NODE_TYPE_ORDER = {
    WorkflowNodeType.CONTEXT_LOADER.value: 0,
    WorkflowNodeType.ROUTER.value: 1,
    WorkflowNodeType.PLANNER.value: 1,
    WorkflowNodeType.RETRIEVAL_WORKER.value: 2,
    WorkflowNodeType.MEMORY_WORKER.value: 3,
    WorkflowNodeType.TIMELINE_WORKER.value: 4,
    WorkflowNodeType.WEB_WORKER.value: 5,
    WorkflowNodeType.EVIDENCE_EVALUATOR.value: 6,
    WorkflowNodeType.REPLANNER.value: 7,
    WorkflowNodeType.DIRECT_ANSWER.value: 8,
    WorkflowNodeType.SYNTHESIZER.value: 9,
    WorkflowNodeType.FINALIZER.value: 10,
    WorkflowNodeType.HITL_GATE.value: 11,
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
            WorkflowNodeType.ROUTER.value: RouteFunctionId.ROUTER.value,
            WorkflowNodeType.PLANNER.value: RouteFunctionId.PLANNER.value,
            WorkflowNodeType.EVIDENCE_EVALUATOR.value: RouteFunctionId.EVALUATOR.value,
            WorkflowNodeType.HITL_GATE.value: RouteFunctionId.HITL_GATE.value,
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
        if source == GraphSentinel.START.value:
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
            label = node.get("label")
            if (
                (not isinstance(label, str) or not label.strip())
                and isinstance(display_name, str)
                and display_name
            ):
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


class WorkflowCompiler(WorkflowMaterializer):
    """Compile validated v2 workflow specs into LangGraph StateGraph instances."""

    def __init__(self, registry: Optional["NodeRegistry"] = None):
        if registry is None:
            from app.agent_workflows.graph import NodeRegistry

            registry = NodeRegistry()
        self.registry = registry

    def compile(
        self,
        spec: Dict[str, Any],
        *,
        checkpointer: Any = None,
    ):
        from app.agent_workflows.validator import WorkflowValidator

        graph_spec = ((spec.get("config") or {}).get("graph") or {}) if isinstance(spec, dict) else {}
        if not graph_spec.get("hitl_compiled"):
            WorkflowValidator().validate(spec)
            spec = self.materialize_spec(spec)
            graph_spec = (spec.get("config") or {}).get("graph") or {}
        workflow = StateGraph(RouterRagState)
        node_types: Dict[str, str] = {}
        outgoing_route_labels: Dict[str, list[str]] = {
            str(edge.get("from")): [str(label) for label in dict(edge.get("routes") or {})]
            for edge in graph_spec.get("edges", [])
            if edge.get("conditional") and isinstance(edge.get("routes"), dict)
        }
        for node in graph_spec.get("nodes", []):
            node_types[node["id"]] = node["type"]
            workflow.add_node(
                node["id"],
                self.registry.get_for_spec(node, route_labels=outgoing_route_labels.get(str(node["id"]))),
            )

        for edge in graph_spec.get("edges", []):
            source = edge.get("from")
            target = edge.get("to")
            if edge.get("conditional"):
                route_fn = route_function_for_edge(
                    edge,
                    source=str(source),
                    node_types=node_types,
                )
                routes = {
                    key: END if value == GraphSentinel.END.value else value
                    for key, value in dict(edge["routes"]).items()
                }
                workflow.add_conditional_edges(source, route_fn, routes)
                continue
            source_ref = START if source == GraphSentinel.START.value else source
            target_ref = END if target == GraphSentinel.END.value else target
            workflow.add_edge(source_ref, target_ref)

        return workflow.compile(checkpointer=checkpointer)
