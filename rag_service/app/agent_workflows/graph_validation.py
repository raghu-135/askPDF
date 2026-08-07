from __future__ import annotations

import sys
from typing import Any, Dict

from app.agent.tool_registry import (
    collect_tool_contract_metadata_errors,
)
from app.agent_workflows.builtin_workflows import builtin_workflow_keys
from app.agent_workflows.parallel_contracts import (
    PARALLEL_FORBIDDEN_CUMULATIVE_CHANNELS,
    PARALLEL_RETRIEVAL_WORKER_TYPES,
)
from app.agent_workflows.enums import GraphSentinel, RouteFunctionId, WorkflowNodeType
from app.agent_workflows.hitl_materializer import materialize_hitl_gates
from app.agent_workflows.hitl_policy_validation import collect_hitl_policy_errors
from app.agent_workflows.node_catalog import (
    collect_node_catalog_errors,
    get_node_catalog as _default_get_node_catalog,
    node_type_allowed_tool_contract_ids,
    node_type_max_visits,
    known_node_types,
)
from app.agent_workflows.route_registry import (
    collect_route_function_registry_errors,
    get_route_function_registry as _default_get_route_function_registry,
    known_route_function_ids,
    route_function_allowed_for_node_type,
    route_function_labels,
)
from app.agent_workflows.tool_permission_validation import (
    collect_tool_permission_errors,
    known_workflow_tool_ids,
    tool_contracts_by_id,
)
from app.agent_workflows.workflow_config_validation import collect_config_errors


def get_node_catalog() -> Dict[str, Dict[str, Any]]:
    validator_module = sys.modules.get("app.agent_workflows.validator")
    accessor = getattr(validator_module, "get_node_catalog", _default_get_node_catalog)
    return accessor()


def get_route_function_registry() -> Dict[str, Dict[str, Any]]:
    validator_module = sys.modules.get("app.agent_workflows.validator")
    accessor = getattr(validator_module, "get_route_function_registry", _default_get_route_function_registry)
    return accessor()


class GenericGraphValidator:
    """Catalog-backed validator for schema v2 graph specs."""

    def collect_errors(self, spec: Dict[str, Any]) -> list[str]:
        errors: list[str] = []
        if not isinstance(spec, dict):
            return ["spec must be an object"]
        if spec.get("schema_version") != 2:
            errors.append("schema_version must be 2")

        workflow_id = spec.get("workflow_id")
        if not isinstance(workflow_id, str) or not workflow_id:
            errors.append("workflow_id must be a non-empty string")
        strict_route_completeness = isinstance(workflow_id, str) and workflow_id in builtin_workflow_keys()

        config = spec.get("config")
        if not isinstance(config, dict):
            errors.append("config must be an object")
            return errors
        errors.extend(collect_config_errors(config, workflow_id))

        allowed_tool_ids = config.get("allowed_tool_ids", [])
        known_tool_ids = known_workflow_tool_ids()
        if not isinstance(allowed_tool_ids, list) or not all(isinstance(item, str) for item in allowed_tool_ids):
            errors.append("allowed_tool_ids must be a list of strings")
            allowed_tool_ids = []
        else:
            unknown_tool_ids = sorted(set(allowed_tool_ids) - known_tool_ids)
            if unknown_tool_ids:
                errors.append(f"unknown allowed_tool_ids: {', '.join(unknown_tool_ids)}")
            errors.extend(collect_tool_permission_errors(spec, set(allowed_tool_ids)))

        graph = config.get("graph")
        if not isinstance(graph, dict):
            errors.append("graph must be an object")
            return errors
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            errors.append("graph.nodes and graph.edges must be lists")
            return errors
        errors.extend(collect_hitl_policy_errors(config.get("hitl_policy", {}), workflow_id, graph))

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
        errors.extend(self._collect_catalog_state_flow_errors(node_catalog))
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
            if node_id in {GraphSentinel.START.value, GraphSentinel.END.value}:
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

        valid_sources = set(node_ids) | {GraphSentinel.START.value}
        valid_targets = set(node_ids) | {GraphSentinel.END.value}
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
                if labels is not None and strict_route_completeness:
                    missing_labels = sorted(labels - set(routes))
                    if missing_labels:
                        errors.append(
                            f"graph conditional edge from {source} is missing route labels: "
                            + ", ".join(missing_labels)
                        )
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
                    errors.extend(self._collect_edge_contract_errors(source, route_target, node_types_by_id, node_catalog))
                continue

            target = edge.get("to")
            if not isinstance(target, str) or target not in valid_targets:
                errors.append(f"graph edge target is unknown: {target}")
                continue
            adjacency.setdefault(source, set()).add(target)
            errors.extend(self._collect_edge_contract_errors(source, target, node_types_by_id, node_catalog))

        errors.extend(self._collect_loop_policy_errors(config.get("loop_policy"), adjacency, node_ids, node_types_by_id, node_catalog))
        errors.extend(self._collect_reachability_errors(adjacency, node_ids))
        errors.extend(
            self._collect_state_flow_errors(
                graph=graph,
                hitl_policy=config.get("hitl_policy"),
                loop_policy=config.get("loop_policy"),
                adjacency=adjacency,
                node_ids=node_ids,
                node_types_by_id=node_types_by_id,
                node_catalog=node_catalog,
                route_registry=route_registry,
            )
        )
        errors.extend(self._collect_parallel_graph_errors(spec, nodes, edges, node_types_by_id))
        errors.extend(self._collect_corrective_graph_errors(spec, nodes, edges, node_types_by_id, adjacency))
        return errors

    def _collect_corrective_graph_errors(
        self,
        spec: Dict[str, Any],
        nodes: list[Any],
        edges: list[Any],
        node_types_by_id: dict[str, str],
        adjacency: dict[str, set[str]],
    ) -> list[str]:
        if spec.get("workflow_id") != "corrective_self_rag_agent":
            return []
        errors: list[str] = []
        required = {
            WorkflowNodeType.PLANNER.value,
            WorkflowNodeType.REPLANNER.value,
            WorkflowNodeType.PARALLEL_DISPATCH.value,
            WorkflowNodeType.AGGREGATOR.value,
            WorkflowNodeType.RETRIEVAL_QUALITY_GRADER.value,
            WorkflowNodeType.GROUNDED_ANSWER_VERIFIER.value,
            WorkflowNodeType.FINALIZER.value,
        }
        counts = {node_type: list(node_types_by_id.values()).count(node_type) for node_type in required}
        for node_type, count in sorted(counts.items()):
            if count != 1:
                errors.append(f"corrective workflow requires exactly one {node_type} node")
        features = ((spec.get("runtime") or {}).get("features") or {})
        for feature in ("supports_corrective_retrieval", "supports_replans", "supports_parallel_dispatch", "supports_answer_quality"):
            if features.get(feature) is not True:
                errors.append(f"corrective workflow requires runtime.features.{feature}=true")
        policy = ((spec.get("config") or {}).get("corrective_policy") or {})
        if policy.get("max_corrective_waves") != 2 or policy.get("max_answer_revisions") != 1:
            errors.append("corrective workflow requires two corrective waves and one answer revision")
        return errors

    def _collect_parallel_graph_errors(
        self,
        spec: Dict[str, Any],
        nodes: list[Any],
        edges: list[Any],
        node_types_by_id: dict[str, str],
    ) -> list[str]:
        errors: list[str] = []
        dispatch_ids = [node_id for node_id, node_type in node_types_by_id.items() if node_type == WorkflowNodeType.PARALLEL_DISPATCH.value]
        aggregator_ids = [node_id for node_id, node_type in node_types_by_id.items() if node_type == WorkflowNodeType.AGGREGATOR.value]
        # Aggregators are shared by serial and parallel execution. Parallel-only
        # topology and feature-gate rules apply only when a parallel dispatcher exists.
        if not dispatch_ids:
            return errors
        runtime = spec.get("runtime") if isinstance(spec.get("runtime"), dict) else {}
        features = runtime.get("features") if isinstance(runtime.get("features"), dict) else {}
        config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
        policy = config.get("parallel_policy") if isinstance(config.get("parallel_policy"), dict) else {}
        if not features.get("supports_parallel_dispatch"):
            errors.append("parallel nodes require runtime.features.supports_parallel_dispatch=true")
        if policy.get("enabled") is not True:
            errors.append("parallel nodes require parallel_policy.enabled=true")
        if len(dispatch_ids) != 1 or len(aggregator_ids) != 1:
            errors.append("parallel workflows require exactly one parallel_dispatch and one aggregator node")
            return errors
        dispatch_id = dispatch_ids[0]
        aggregator_id = aggregator_ids[0]
        incoming_dispatch = [
            edge for edge in edges
            if isinstance(edge, dict)
            and (
                edge.get("to") == dispatch_id
                or dispatch_id in set((edge.get("routes") or {}).values())
            )
        ]
        is_corrective = spec.get("workflow_id") == "corrective_self_rag_agent"
        dispatch_parent = str(incoming_dispatch[0].get("from")) if len(incoming_dispatch) == 1 else ""
        parent_is_planner = node_types_by_id.get(dispatch_parent) == WorkflowNodeType.PLANNER.value
        parent_is_pre_dispatch_gate = (
            node_types_by_id.get(dispatch_parent) == WorkflowNodeType.HITL_GATE.value
            and any(
                isinstance(edge, dict)
                and (
                    edge.get("to") == dispatch_parent
                    or dispatch_parent in set((edge.get("routes") or {}).values())
                )
                and node_types_by_id.get(str(edge.get("from"))) == WorkflowNodeType.PLANNER.value
                for edge in edges
            )
        )
        if is_corrective:
            allowed_parent_types = {WorkflowNodeType.PLANNER.value, WorkflowNodeType.REPLANNER.value, WorkflowNodeType.HITL_GATE.value}
            if not incoming_dispatch or any(node_types_by_id.get(str(edge.get("from"))) not in allowed_parent_types for edge in incoming_dispatch):
                errors.append("corrective parallel_dispatch parents must be Planner, Replanner, or the web approval gate")
        elif len(incoming_dispatch) != 1 or not (parent_is_planner or parent_is_pre_dispatch_gate):
            errors.append("parallel_dispatch must have exactly one Planner parent")
        dispatch_edge = next(
            (
                edge for edge in edges
                if isinstance(edge, dict)
                and edge.get("from") == dispatch_id
                and edge.get("conditional")
                and edge.get("route_fn") == RouteFunctionId.PARALLEL_DISPATCH.value
            ),
            None,
        )
        if dispatch_edge is None:
            errors.append("parallel_dispatch must declare one parallel_dispatch_route conditional edge")
        dynamic_workers = {
            str(edge.get("to"))
            for edge in edges
            if isinstance(edge, dict) and edge.get("from") == dispatch_id and edge.get("dynamic") is True
        }
        if not dynamic_workers:
            errors.append("parallel_dispatch must declare at least one dynamic worker target")
        for worker_id in sorted(dynamic_workers):
            worker_type = node_types_by_id.get(worker_id)
            if worker_type not in PARALLEL_RETRIEVAL_WORKER_TYPES:
                errors.append(f"parallel dispatch target {worker_id} must be a read-only retrieval worker")
            metadata = _default_get_node_catalog().get(str(worker_type), {})
            parallel_writes = metadata.get("parallel_state_writes") if isinstance(metadata, dict) else None
            if not isinstance(parallel_writes, list) or "worker_result_packets" not in parallel_writes:
                errors.append(f"parallel worker {worker_id} must declare reducer-safe parallel_state_writes")
            unsafe = sorted(set(parallel_writes or []) & PARALLEL_FORBIDDEN_CUMULATIVE_CHANNELS)
            if unsafe:
                errors.append(f"parallel worker {worker_id} declares unsafe cumulative writes: {', '.join(unsafe)}")
            outgoing = [
                edge for edge in edges
                if isinstance(edge, dict) and edge.get("from") == worker_id
            ]
            incoming = [
                edge for edge in edges
                if isinstance(edge, dict) and edge.get("to") == worker_id
            ]
            if any(edge.get("from") != dispatch_id or edge.get("dynamic") is not True for edge in incoming):
                errors.append(f"parallel worker {worker_id} cannot have parents outside dispatcher {dispatch_id}")
            joins = [edge for edge in outgoing if edge.get("to") == aggregator_id and not edge.get("conditional")]
            if len(joins) != 1:
                errors.append(f"parallel worker {worker_id} must join aggregator {aggregator_id} exactly once")
            if len(outgoing) != 1:
                errors.append(f"parallel worker {worker_id} cannot have exits outside aggregator {aggregator_id}")
        aggregator_parents = {
            str(edge.get("from"))
            for edge in edges
            if isinstance(edge, dict) and edge.get("to") == aggregator_id
        }
        allowed_parents = dynamic_workers | {dispatch_id}
        bypasses = sorted(aggregator_parents - allowed_parents)
        if bypasses:
            errors.append(f"aggregator cannot have bypass parents: {', '.join(bypasses)}")
        if any(node_types_by_id.get(node_id) == WorkflowNodeType.HITL_GATE.value for node_id in dynamic_workers):
            errors.append("HITL gates are not allowed inside parallel branches")
        if any(
            isinstance(edge, dict)
            and edge.get("from") in dynamic_workers
            and (
                edge.get("conditional")
                or node_types_by_id.get(str(edge.get("to"))) in {
                    WorkflowNodeType.HITL_GATE.value,
                    WorkflowNodeType.PARALLEL_DISPATCH.value,
                }
            )
            for edge in edges
        ):
            errors.append("parallel regions cannot contain HITL, nested dispatch, conditional exits, or cycles")
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

    def _collect_catalog_state_flow_errors(
        self,
        node_catalog: Dict[str, Dict[str, Any]],
    ) -> list[str]:
        errors: list[str] = []
        worker_types = {
            "retrieval_worker",
            "thread_conversation_history_worker",
            "durable_memory_worker",
            "thread_events_worker",
            "web_worker",
        }
        for node_type in sorted(worker_types):
            writes = set((node_catalog.get(node_type) or {}).get("state_writes") or [])
            missing = sorted({"evidence", "evidence_packets"} - writes)
            if missing:
                errors.append(
                    f"node catalog type {node_type} must write worker evidence state: {', '.join(missing)}"
                )
        for node_type in ("direct_answer", "synthesizer"):
            writes = set((node_catalog.get(node_type) or {}).get("state_writes") or [])
            if "final_answer" not in writes:
                errors.append(f"node catalog type {node_type} must write final_answer")
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

    def _collect_edge_contract_errors(
        self,
        source: str,
        target: str,
        node_types_by_id: dict[str, str],
        node_catalog: Dict[str, Dict[str, Any]],
    ) -> list[str]:
        if source == GraphSentinel.START.value and target == GraphSentinel.END.value:
            return ["START cannot connect directly to END"]
        source_type = (
            GraphSentinel.START.value
            if source == GraphSentinel.START.value
            else node_types_by_id.get(source)
        )
        target_type = (
            GraphSentinel.END.value
            if target == GraphSentinel.END.value
            else node_types_by_id.get(target)
        )
        if not source_type or not target_type:
            return []
        errors: list[str] = []
        if source_type != GraphSentinel.START.value:
            allowed_children = set((node_catalog.get(source_type) or {}).get("allowed_child_types") or [])
            if target_type not in allowed_children:
                errors.append(f"node {source} type {source_type} cannot connect to child {target} type {target_type}")
        if target_type != GraphSentinel.END.value:
            allowed_parents = set((node_catalog.get(target_type) or {}).get("allowed_parent_types") or [])
            if source_type not in allowed_parents:
                errors.append(f"node {target} type {target_type} cannot accept parent {source} type {source_type}")
        return errors

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
            max_visits = node_type_max_visits(node_type or "")
            if limit > max(default_max_value, max_visits):
                errors.append(
                    f"loop_policy.node_visit_limits.{node_id} exceeds allowed max for node type {node_type}"
                )

        has_dynamic_parallel_dispatch = WorkflowNodeType.PARALLEL_DISPATCH.value in set(node_types_by_id.values())
        if has_cycle and not has_dynamic_parallel_dispatch and max_total_visits_value > 0 and default_max_value > 0:
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
            if node in {GraphSentinel.START.value, GraphSentinel.END.value}:
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

    def _collect_state_flow_errors(
        self,
        *,
        graph: Dict[str, Any],
        hitl_policy: Any,
        loop_policy: Any,
        adjacency: dict[str, set[str]],
        node_ids: set[str],
        node_types_by_id: dict[str, str],
        node_catalog: Dict[str, Dict[str, Any]],
        route_registry: Dict[str, Dict[str, Any]],
    ) -> list[str]:
        errors: list[str] = []
        edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
        outgoing_edges: dict[str, list[Dict[str, Any]]] = {}
        for edge in edges:
            source = edge.get("from")
            if isinstance(source, str) and not edge.get("dynamic"):
                outgoing_edges.setdefault(source, []).append(edge)

        for source, source_edges in sorted(outgoing_edges.items()):
            conditional_count = sum(bool(edge.get("conditional")) for edge in source_edges)
            if conditional_count > 1:
                errors.append(f"node {source} has multiple conditional outgoing edges")
            if (
                conditional_count
                and conditional_count != len(source_edges)
                and node_types_by_id.get(source) != "hitl_gate"
            ):
                errors.append(f"node {source} mixes conditional and unconditional outgoing edges")

        errors.extend(
            self._collect_route_target_errors(
                edges,
                adjacency,
                node_types_by_id,
                route_registry,
            )
        )

        finalizer_ids = {
            node_id for node_id, node_type in node_types_by_id.items() if node_type == "finalizer"
        }
        if not finalizer_ids:
            errors.append("graph requires a finalizer node")

        for node_id, node_type in sorted(node_types_by_id.items()):
            if node_type in {"direct_answer", "synthesizer"} and not self._can_reach_type(
                node_id,
                {"finalizer"},
                adjacency,
                node_types_by_id,
            ):
                errors.append(f"node {node_id} type {node_type} must flow through a finalizer")
            if node_type == "evidence_evaluator" and self._can_reach_while_avoiding_types(
                GraphSentinel.START.value,
                node_id,
                {
                    "retrieval_worker",
                    "thread_conversation_history_worker",
                    "durable_memory_worker",
                    "thread_events_worker",
                    "web_worker",
                    "aggregator",
                },
                adjacency,
                node_types_by_id,
            ):
                errors.append(f"evidence evaluator {node_id} requires upstream worker evidence on every path")
            if node_type == "replanner":
                reaches_worker = self._can_reach_type(
                    node_id,
                    {
                        "retrieval_worker",
                        "thread_conversation_history_worker",
                        "durable_memory_worker",
                        "thread_events_worker",
                        "web_worker",
                    },
                    adjacency,
                    node_types_by_id,
                )
                reaches_evaluator = self._can_reach_type(
                    node_id,
                    {"evidence_evaluator", "retrieval_quality_grader"},
                    adjacency,
                    node_types_by_id,
                )
                if not reaches_worker or not reaches_evaluator:
                    errors.append(
                        f"replanner {node_id} must return through an evidence worker to an evidence evaluator or retrieval quality grader"
                    )

        errors.extend(
            self._collect_cycle_state_flow_errors(
                adjacency,
                node_ids,
                node_types_by_id,
                loop_policy,
            )
        )

        effective_graph = materialize_hitl_gates(
            graph,
            hitl_policy=hitl_policy if isinstance(hitl_policy, dict) else {},
        )
        effective_nodes = {
            str(node.get("id")): str(node.get("type"))
            for node in effective_graph.get("nodes", [])
            if isinstance(node, dict)
            and isinstance(node.get("id"), str)
            and isinstance(node.get("type"), str)
        }
        effective_adjacency = self._adjacency_from_edges(effective_graph.get("edges", []))
        reachable = self._reachable_from(GraphSentinel.START.value, effective_adjacency)
        can_end = self._nodes_that_can_reach_end(effective_adjacency)
        non_terminating = sorted(
            node_id
            for node_id in reachable
            if node_id in effective_nodes and node_id not in can_end
        )
        if non_terminating:
            errors.append(
                "graph contains reachable nodes with no path to END after HITL materialization: "
                + ", ".join(non_terminating)
            )

        for edge in effective_graph.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source = edge.get("from")
            targets = (
                list((edge.get("routes") or {}).values())
                if edge.get("conditional") and isinstance(edge.get("routes"), dict)
                else [edge.get("to")]
            )
            if (
                GraphSentinel.END.value in targets
                and source in effective_nodes
                and effective_nodes[source] not in {"finalizer", "hitl_gate"}
            ):
                errors.append(f"node {source} must flow through a finalizer before END")
        return errors

    def _collect_route_target_errors(
        self,
        edges: list[Dict[str, Any]],
        adjacency: dict[str, set[str]],
        node_types_by_id: dict[str, str],
        route_registry: Dict[str, Dict[str, Any]],
    ) -> list[str]:
        errors: list[str] = []
        for edge in edges:
            if not edge.get("conditional") or not isinstance(edge.get("routes"), dict):
                continue
            source = edge.get("from")
            route_fn = edge.get("route_fn")
            metadata = route_registry.get(route_fn) if isinstance(route_fn, str) else None
            target_types_by_label = (
                metadata.get("target_types_by_label")
                if isinstance(metadata, dict)
                and isinstance(metadata.get("target_types_by_label"), dict)
                else {}
            )
            for label, target in edge["routes"].items():
                allowed_types = set(target_types_by_label.get(label) or [])
                if not allowed_types or not isinstance(target, str):
                    continue
                actual_types = self._first_non_hitl_target_types(
                    target,
                    adjacency,
                    node_types_by_id,
                )
                if actual_types and actual_types.isdisjoint(allowed_types):
                    errors.append(
                        f"route {label} from node {source} must target node types "
                        f"{', '.join(sorted(allowed_types))}; found {', '.join(sorted(actual_types))}"
                    )
        return errors

    def _first_non_hitl_target_types(
        self,
        target: str,
        adjacency: dict[str, set[str]],
        node_types_by_id: dict[str, str],
    ) -> set[str]:
        pending = [target]
        visited: set[str] = set()
        result: set[str] = set()
        while pending:
            node_id = pending.pop()
            if node_id in visited or node_id == GraphSentinel.END.value:
                continue
            visited.add(node_id)
            node_type = node_types_by_id.get(node_id)
            if node_type == "hitl_gate":
                pending.extend(adjacency.get(node_id) or [])
            elif node_type:
                result.add(node_type)
        return result

    def _collect_cycle_state_flow_errors(
        self,
        adjacency: dict[str, set[str]],
        node_ids: set[str],
        node_types_by_id: dict[str, str],
        loop_policy: Any,
    ) -> list[str]:
        errors: list[str] = []
        components = self._strongly_connected_components(adjacency, node_ids)
        cyclic_components = [
            component
            for component in components
            if len(component) > 1
            or any(node_id in adjacency.get(node_id, set()) for node_id in component)
        ]
        if not cyclic_components or not isinstance(loop_policy, dict):
            return errors
        default_limit = self._positive_int(loop_policy.get("default_max_node_visits"), 1)
        node_limits = (
            loop_policy.get("node_visit_limits")
            if isinstance(loop_policy.get("node_visit_limits"), dict)
            else {}
        )
        max_total = self._positive_int(loop_policy.get("max_total_visits"), 0)
        if max_total <= len(node_ids):
            errors.append("loop_policy.max_total_visits must allow at least one bounded cycle revisit")
        for component in cyclic_components:
            limits = {
                node_id: self._positive_int(node_limits.get(node_id), default_limit)
                for node_id in component
            }
            if max(limits.values(), default=1) <= 1:
                errors.append(
                    "cycle has no node with a repeat visit budget: "
                    + ", ".join(sorted(component))
                )
            replanners = [
                node_id for node_id in component if node_types_by_id.get(node_id) == "replanner"
            ]
            evaluators = [
                node_id for node_id in component if node_types_by_id.get(node_id) in {"evidence_evaluator", "retrieval_quality_grader"}
            ]
            for replanner_id in replanners:
                for evaluator_id in evaluators:
                    if limits[evaluator_id] < limits[replanner_id] + 1:
                        errors.append(
                            f"loop_policy must allow evaluator {evaluator_id} one more visit than "
                            f"replanner {replanner_id}"
                        )
                for worker_id in component:
                    if (
                        node_types_by_id.get(worker_id)
                        in {"retrieval_worker", "thread_conversation_history_worker", "durable_memory_worker", "thread_events_worker", "web_worker"}
                        and limits[worker_id] < limits[replanner_id] + 1
                    ):
                        errors.append(
                            f"loop_policy must allow worker {worker_id} one more visit than "
                            f"replanner {replanner_id}"
                        )
        return errors

    def _strongly_connected_components(
        self,
        adjacency: dict[str, set[str]],
        node_ids: set[str],
    ) -> list[set[str]]:
        index = 0
        indices: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        stack: list[str] = []
        on_stack: set[str] = set()
        components: list[set[str]] = []

        def visit(node_id: str) -> None:
            nonlocal index
            indices[node_id] = index
            lowlinks[node_id] = index
            index += 1
            stack.append(node_id)
            on_stack.add(node_id)
            for target in adjacency.get(node_id, set()):
                if target not in node_ids:
                    continue
                if target not in indices:
                    visit(target)
                    lowlinks[node_id] = min(lowlinks[node_id], lowlinks[target])
                elif target in on_stack:
                    lowlinks[node_id] = min(lowlinks[node_id], indices[target])
            if lowlinks[node_id] == indices[node_id]:
                component: set[str] = set()
                while stack:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.add(member)
                    if member == node_id:
                        break
                components.append(component)

        for node_id in sorted(node_ids):
            if node_id not in indices:
                visit(node_id)
        return components

    def _can_reach_type(
        self,
        start: str,
        target_types: set[str],
        adjacency: dict[str, set[str]],
        node_types_by_id: dict[str, str],
    ) -> bool:
        return any(
            node_types_by_id.get(node_id) in target_types
            for node_id in self._reachable_from(start, adjacency)
            if node_id != start
        )

    def _can_reach_while_avoiding_types(
        self,
        start: str,
        target: str,
        avoided_types: set[str],
        adjacency: dict[str, set[str]],
        node_types_by_id: dict[str, str],
    ) -> bool:
        pending = [start]
        visited: set[str] = set()
        while pending:
            node_id = pending.pop()
            if node_id == target:
                return True
            if node_id in visited:
                continue
            visited.add(node_id)
            for child in adjacency.get(node_id, set()):
                if child == target or node_types_by_id.get(child) not in avoided_types:
                    pending.append(child)
        return False

    def _adjacency_from_edges(self, edges: Any) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = {}
        if not isinstance(edges, list):
            return adjacency
        for edge in edges:
            if not isinstance(edge, dict) or not isinstance(edge.get("from"), str):
                continue
            targets = (
                list(edge.get("routes", {}).values())
                if edge.get("conditional") and isinstance(edge.get("routes"), dict)
                else [edge.get("to")]
            )
            for target in targets:
                if isinstance(target, str):
                    adjacency.setdefault(edge["from"], set()).add(target)
        return adjacency

    def _reachable_from(
        self,
        start: str,
        adjacency: dict[str, set[str]],
    ) -> set[str]:
        visited: set[str] = set()
        pending = list(adjacency.get(start) or [])
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            pending.extend(adjacency.get(node_id) or [])
        return visited

    def _nodes_that_can_reach_end(
        self,
        adjacency: dict[str, set[str]],
    ) -> set[str]:
        reverse: dict[str, set[str]] = {}
        for source, targets in adjacency.items():
            for target in targets:
                reverse.setdefault(target, set()).add(source)
        return self._reachable_from(GraphSentinel.END.value, reverse)

    @staticmethod
    def _positive_int(value: Any, fallback: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return fallback
        return parsed if parsed > 0 else fallback

    def _collect_reachability_errors(self, adjacency: dict[str, set[str]], node_ids: set[str]) -> list[str]:
        errors: list[str] = []
        if GraphSentinel.START.value not in adjacency:
            errors.append("graph must have an edge from START")
            return errors
        visited: set[str] = set()
        stack = list(adjacency.get(GraphSentinel.START.value) or [])
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node == GraphSentinel.END.value:
                continue
            stack.extend(adjacency.get(node) or [])
        unreachable = sorted(node_ids - visited)
        if unreachable:
            errors.append(f"graph contains unreachable nodes: {', '.join(unreachable)}")
        if GraphSentinel.END.value not in visited:
            errors.append("graph must be able to reach END from START")
        return errors
