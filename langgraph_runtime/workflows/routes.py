from __future__ import annotations

from typing import Any, Callable, Dict

from langgraph_runtime.workflows.enums import (
    AgentRunResumeAction,
    AnswerQualityRoute,
    CorrectiveRetrievalRoute,
    EvaluatorRoute,
    GroundedAnswerRoute,
    PlannerRoute,
    RouteFunctionId,
    RouterRoute,
    EVALUATOR_ROUTES,
    PLANNER_ROUTES,
    ROUTER_ROUTES,
    ANSWER_QUALITY_ROUTES,
    CORRECTIVE_RETRIEVAL_ROUTES,
    GROUNDED_ANSWER_ROUTES,
)
from langgraph_runtime.workflows.route_registry import route_function_allowed_for_node_type
from langgraph_runtime.workflows.parallel_runtime import dispatch_sends, serial_dispatch_next
from langgraph_runtime.workflows.deep_research_nodes import budget_review_route, deep_task_dispatch_sends, deep_task_route


def router_route(state: Dict[str, Any]) -> str:
    route = state.get("route")
    return route if route in ROUTER_ROUTES else RouterRoute.DOCUMENT.value


def planner_route(state: Dict[str, Any]) -> str:
    route = state.get("route")
    return route if route in PLANNER_ROUTES else PlannerRoute.EXECUTE.value


def evaluator_route(state: Dict[str, Any]) -> str:
    route = state.get("evaluator_route")
    return route if route in EVALUATOR_ROUTES else EvaluatorRoute.ANSWER.value


def answer_quality_route(state: Dict[str, Any]) -> str:
    route = state.get("answer_quality_route")
    return route if route in ANSWER_QUALITY_ROUTES else AnswerQualityRoute.PASS.value


def corrective_retrieval_route(state: Dict[str, Any]) -> str:
    route = state.get("corrective_retrieval_route")
    return route if route in CORRECTIVE_RETRIEVAL_ROUTES else CorrectiveRetrievalRoute.INSUFFICIENT.value


def grounded_answer_route(state: Dict[str, Any]) -> str:
    route = state.get("grounded_answer_route")
    return route if route in GROUNDED_ANSWER_ROUTES else GroundedAnswerRoute.FINALIZE_CAUTIOUS.value


def hitl_gate_route(state: Dict[str, Any]) -> str:
    route = state.get("hitl_gate_route")
    return route if route in {AgentRunResumeAction.APPROVE.value, AgentRunResumeAction.CONTINUE_WITHOUT.value} else AgentRunResumeAction.CONTINUE_WITHOUT.value


def hitl_gate_route_for(gate_id: str) -> Callable[[Dict[str, Any]], Any]:
    def _route(state: Dict[str, Any]) -> Any:
        routes = state.get("hitl_gate_routes") if isinstance(state.get("hitl_gate_routes"), dict) else {}
        route = routes.get(gate_id, state.get("hitl_gate_route"))
        return route if isinstance(route, str) and route else AgentRunResumeAction.CONTINUE_WITHOUT.value

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
        if route_fn_id == RouteFunctionId.HITL_GATE.value:
            return hitl_gate_route_for(str(source))
        if route_fn_id == RouteFunctionId.PLANNER.value:
            return planner_route
        if route_fn_id == RouteFunctionId.EVALUATOR.value:
            return evaluator_route
        if route_fn_id == RouteFunctionId.ROUTER.value:
            return router_route
        if route_fn_id == RouteFunctionId.PARALLEL_DISPATCH.value:
            return dispatch_sends
        if route_fn_id == RouteFunctionId.SERIAL_DISPATCH.value:
            return serial_dispatch_next
        if route_fn_id == RouteFunctionId.ANSWER_QUALITY.value:
            return answer_quality_route
        if route_fn_id == RouteFunctionId.CORRECTIVE_RETRIEVAL.value:
            return corrective_retrieval_route
        if route_fn_id == RouteFunctionId.GROUNDED_ANSWER.value:
            return grounded_answer_route
        if route_fn_id == RouteFunctionId.DEEP_TASK_DISPATCH.value:
            return deep_task_dispatch_sends
        if route_fn_id == RouteFunctionId.DEEP_TASK.value:
            return deep_task_route
        if route_fn_id == RouteFunctionId.BUDGET_REVIEW.value:
            return budget_review_route
        raise ValueError(f"Unknown route function: {route_fn_id}")

    raise ValueError(f"Conditional edge from {source} must declare route_fn")
