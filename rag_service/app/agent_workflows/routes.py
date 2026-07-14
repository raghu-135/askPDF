from __future__ import annotations

from typing import Any, Callable, Dict

from app.agent_workflows.route_registry import route_function_allowed_for_node_type


def router_route(state: Dict[str, Any]) -> str:
    route = state.get("route")
    return route if route in {"document", "memory", "timeline", "web", "direct", "clarify"} else "document"


def planner_route(state: Dict[str, Any]) -> str:
    route = state.get("route")
    return route if route in {"execute", "direct", "clarify"} else "execute"


def evaluator_route(state: Dict[str, Any]) -> str:
    route = state.get("evaluator_route")
    return route if route in {"answer", "replan", "answer_budget_exhausted"} else "answer"


def hitl_gate_route(state: Dict[str, Any]) -> str:
    route = state.get("hitl_gate_route")
    return route if route in {"approve", "continue_without"} else "continue_without"


def hitl_gate_route_for(gate_id: str) -> Callable[[Dict[str, Any]], Any]:
    def _route(state: Dict[str, Any]) -> Any:
        routes = state.get("hitl_gate_routes") if isinstance(state.get("hitl_gate_routes"), dict) else {}
        route = routes.get(gate_id, state.get("hitl_gate_route"))
        return route if isinstance(route, str) and route else "continue_without"

    return _route


def route_function_for_edge(
    edge: Dict[str, Any],
    *,
    source: str,
    node_types: Dict[str, str],
) -> Callable[[Dict[str, Any]], Any]:
    route_fn_id = edge.get("route_fn")
    source_type = node_types.get(source)
    if isinstance(route_fn_id, str) and route_fn_id:
        if source_type and not route_function_allowed_for_node_type(route_fn_id, source_type):
            raise ValueError(f"Route function {route_fn_id} is not allowed from node type {source_type}")
        if route_fn_id == "hitl_gate_route":
            return hitl_gate_route_for(str(source))
        if route_fn_id == "planner_route":
            return planner_route
        if route_fn_id == "evaluator_route":
            return evaluator_route
        if route_fn_id == "router_route":
            return router_route
        raise ValueError(f"Unknown route function: {route_fn_id}")

    raise ValueError(f"Conditional edge from {source} must declare route_fn")
