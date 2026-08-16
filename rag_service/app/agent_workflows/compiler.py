from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.agent_workflows.enums import GraphSentinel, RouteFunctionId, WorkflowNodeType
from app.agent_workflows.hitl_materializer import materialize_hitl_gates
from app.agent_workflows.node_catalog import (
    get_node_type_metadata,
    node_type_default_max_visits,
    node_type_max_visits,
)
from app.agent_workflows.routes import route_function_for_edge
from app.agent_workflows.state import RouterRagState
from app.agent_workflows.parallel_runtime import normalized_parallel_policy, parallel_retryable_error
from app.agent_workflows.corrective_contracts import normalized_corrective_policy
from app.agent_workflows.parallel_contracts import (
    PARALLEL_RETRIEVAL_WORKER_TYPES,
    PARALLEL_RETRY_BACKOFF_FACTOR,
    PARALLEL_RETRY_INITIAL_INTERVAL_SECONDS,
    PARALLEL_RETRY_JITTER,
    PARALLEL_RETRY_MAX_INTERVAL_SECONDS,
)

if TYPE_CHECKING:
    from app.agent_workflows.graph import NodeRegistry


CANONICAL_NODE_TYPE_ORDER = {
    WorkflowNodeType.CONTEXT_LOADER.value: 0,
    WorkflowNodeType.ROUTER.value: 1,
    WorkflowNodeType.PLANNER.value: 1,
    WorkflowNodeType.PARALLEL_DISPATCH.value: 2,
    WorkflowNodeType.SERIAL_DISPATCH.value: 2,
    WorkflowNodeType.RETRIEVAL_WORKER.value: 2,
    WorkflowNodeType.THREAD_CONVERSATION_HISTORY_WORKER.value: 3,
    WorkflowNodeType.DURABLE_MEMORY_WORKER.value: 4,
    WorkflowNodeType.THREAD_EVENTS_WORKER.value: 5,
    WorkflowNodeType.WEB_WORKER.value: 6,
    WorkflowNodeType.AGGREGATOR.value: 7,
    WorkflowNodeType.EVIDENCE_EVALUATOR.value: 8,
    WorkflowNodeType.RETRIEVAL_QUALITY_GRADER.value: 8,
    WorkflowNodeType.REPLANNER.value: 8,
    WorkflowNodeType.DIRECT_ANSWER.value: 9,
    WorkflowNodeType.SYNTHESIZER.value: 10,
    WorkflowNodeType.ANSWER_EVALUATOR.value: 11,
    WorkflowNodeType.ANSWER_REVISER.value: 12,
    WorkflowNodeType.GROUNDED_ANSWER_VERIFIER.value: 12,
    WorkflowNodeType.FINALIZER.value: 13,
    WorkflowNodeType.HITL_GATE.value: 14,
    WorkflowNodeType.DEEP_TASK_PLANNER.value: 1,
    WorkflowNodeType.DEEP_TASK_SCHEDULER.value: 2,
    WorkflowNodeType.DEEP_RESEARCH_SUBAGENT.value: 3,
    WorkflowNodeType.DEEP_COORDINATOR.value: 4,
    WorkflowNodeType.DEEP_TASK_SYNTHESIZER.value: 10,
    WorkflowNodeType.EVIDENCE_CRITIC.value: 11,
}


class WorkflowMaterializer:
    """Materialize validated workflow specs before LangGraph compilation."""

    def materialize_spec(
        self,
        spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        materialized = deepcopy(spec)
        # Persisted workflow snapshots are compiled by the v2 validator.  The
        # resolver can return a v1-shaped envelope for older built-in/task
        # rows, even though the graph payload is otherwise compatible.  Make
        # the materialized snapshot self-describing so continuations do not
        # fail solely because the marker was omitted.
        materialized["schema_version"] = 2
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
        if "parallel_policy" in config:
            config["parallel_policy"] = normalized_parallel_policy(config.get("parallel_policy"))
        if "corrective_policy" in config:
            config["corrective_policy"] = normalized_corrective_policy(config.get("corrective_policy"))
        materialized["config"] = config
        return materialized

    def _with_materialized_loop_policy(self, loop_policy: Any, *, graph_spec: Dict[str, Any]) -> Dict[str, Any]:
        policy = dict(loop_policy) if isinstance(loop_policy, dict) else {}
        nodes = [node for node in graph_spec.get("nodes", []) if isinstance(node, dict)]
        node_types = {
            str(node.get("id")): str(node.get("type"))
            for node in nodes
            if node.get("id") and node.get("type")
        }
        node_count = len(nodes)
        try:
            max_total_visits = int(policy.get("max_total_visits", 0))
        except (TypeError, ValueError):
            max_total_visits = 0
        if node_count and max_total_visits < node_count:
            max_total_visits = node_count
            policy["max_total_visits"] = max_total_visits
        node_visit_limits = policy.get("node_visit_limits")
        if isinstance(node_visit_limits, dict):
            policy["node_visit_limits"] = dict(node_visit_limits)
        elif node_visit_limits is not None:
            policy["node_visit_limits"] = {}
        else:
            policy["node_visit_limits"] = {}

        # HITL nodes are inserted after the source workflow's loop policy has
        # been resolved. A gate that guards a loop must be visitable once for
        # each distinct entry into that loop (for example, the initial plan
        # plus every corrective replan), otherwise a valid final wave fails at
        # the gate before dispatch. Derive that budget from the materialized
        # gate's incoming sources while preserving explicitly authored limits.
        limits = policy["node_visit_limits"]
        default_limit = max(1, int(policy.get("default_max_node_visits", 1)))
        added_gate_budget = 0
        edges = [edge for edge in graph_spec.get("edges", []) if isinstance(edge, dict)]
        for gate_id, node_type in node_types.items():
            if node_type != WorkflowNodeType.HITL_GATE.value or gate_id in limits:
                continue
            incoming_sources = {
                str(edge.get("from"))
                for edge in edges
                if edge.get("to") == gate_id
                or gate_id in (
                    edge.get("routes", {}).values()
                    if isinstance(edge.get("routes"), dict)
                    else ()
                )
            }
            derived_limit = sum(
                max(
                    1,
                    int(
                        limits.get(
                            source_id,
                            min(
                                default_limit,
                                node_type_default_max_visits(node_types.get(source_id, "")),
                            ),
                        )
                    ),
                )
                for source_id in incoming_sources
            )
            gate_limit = max(
                node_type_default_max_visits(node_type),
                min(derived_limit, node_type_max_visits(node_type)),
            )
            limits[gate_id] = gate_limit
            added_gate_budget += gate_limit
        if added_gate_budget:
            policy["max_total_visits"] = max_total_visits + added_gate_budget
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
            WorkflowNodeType.PARALLEL_DISPATCH.value: RouteFunctionId.PARALLEL_DISPATCH.value,
            WorkflowNodeType.SERIAL_DISPATCH.value: RouteFunctionId.SERIAL_DISPATCH.value,
            WorkflowNodeType.ANSWER_EVALUATOR.value: RouteFunctionId.ANSWER_QUALITY.value,
            WorkflowNodeType.RETRIEVAL_QUALITY_GRADER.value: RouteFunctionId.CORRECTIVE_RETRIEVAL.value,
            WorkflowNodeType.GROUNDED_ANSWER_VERIFIER.value: RouteFunctionId.GROUNDED_ANSWER.value,
            WorkflowNodeType.DEEP_TASK_SCHEDULER.value: RouteFunctionId.DEEP_TASK_DISPATCH.value,
            WorkflowNodeType.DEEP_COORDINATOR.value: RouteFunctionId.DEEP_TASK.value,
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
                "parallel_state_writes",
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

        # Continuations persist the resolved snapshot and may predate the v2
        # envelope marker. Normalize that compatibility detail before the
        # validator runs; otherwise the validator rejects a graph that the
        # materializer can safely compile. The graph shape is still validated
        # below, so this does not bypass workflow validation.
        if isinstance(spec, dict) and spec.get("schema_version") != 2:
            spec = deepcopy(spec)
            spec["schema_version"] = 2
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
        dynamic_targets: Dict[str, tuple[str, ...]] = {}
        for edge in graph_spec.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("from") or "")
            targets: list[str] = []
            if edge.get("dynamic") is True and edge.get("to"):
                targets.append(str(edge["to"]))
            if edge.get("route_fn") in {RouteFunctionId.PARALLEL_DISPATCH.value, RouteFunctionId.SERIAL_DISPATCH.value}:
                targets.extend(str(value) for value in (edge.get("routes") or {}).values() if value)
            if source and targets:
                dynamic_targets[source] = tuple(dict.fromkeys((*dynamic_targets.get(source, ()), *targets)))
        config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
        parallel_policy = normalized_parallel_policy(config.get("parallel_policy"))
        parallel_worker_ids = {
            str(edge.get("to"))
            for edge in graph_spec.get("edges", [])
            if isinstance(edge, dict) and edge.get("dynamic") is True
        }
        for node in graph_spec.get("nodes", []):
            node_types[node["id"]] = node["type"]
            add_options: Dict[str, Any] = {}
            if node["id"] in dynamic_targets:
                add_options["destinations"] = dynamic_targets[node["id"]]
            if node["id"] in parallel_worker_ids and node["type"] in PARALLEL_RETRIEVAL_WORKER_TYPES:
                add_options["retry_policy"] = RetryPolicy(
                    max_attempts=parallel_policy["max_attempts"],
                    initial_interval=PARALLEL_RETRY_INITIAL_INTERVAL_SECONDS,
                    backoff_factor=PARALLEL_RETRY_BACKOFF_FACTOR,
                    max_interval=PARALLEL_RETRY_MAX_INTERVAL_SECONDS,
                    jitter=PARALLEL_RETRY_JITTER,
                    retry_on=parallel_retryable_error,
                )
                add_options["error_handler"] = self.registry.get_parallel_error_handler_for_spec(node)
            workflow.add_node(
                node["id"],
                self.registry.get_for_spec(node, route_labels=outgoing_route_labels.get(str(node["id"]))),
                **add_options,
            )

        for edge in graph_spec.get("edges", []):
            if edge.get("dynamic"):
                continue
            source = edge.get("from")
            target = edge.get("to")
            if edge.get("conditional"):
                route_fn = route_function_for_edge(
                    edge,
                    source=str(source),
                    node_types=node_types,
                )
                if node_types.get(str(source)) in {
                    WorkflowNodeType.PARALLEL_DISPATCH.value,
                    WorkflowNodeType.SERIAL_DISPATCH.value,
                    WorkflowNodeType.DEEP_TASK_SCHEDULER.value,
                }:
                    workflow.add_conditional_edges(source, route_fn)
                    continue
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
