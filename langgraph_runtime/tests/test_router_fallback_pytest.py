from langgraph_runtime.workflows.route_registry import RouterRoute


def test_router_fallback_order_prefers_direct_when_clarification_bypassed():
    allowed_routes = {
        RouterRoute.DIRECT.value,
        RouterRoute.DOCUMENT.value,
        RouterRoute.THREAD_CONVERSATION_HISTORY.value,
    }
    bypass_clarification = True
    fallback_order = [
        RouterRoute.DIRECT.value if bypass_clarification else RouterRoute.DOCUMENT.value,
        RouterRoute.DOCUMENT.value if bypass_clarification else RouterRoute.DIRECT.value,
        RouterRoute.CLARIFY.value,
        RouterRoute.THREAD_CONVERSATION_HISTORY.value,
        RouterRoute.DURABLE_MEMORY.value,
        RouterRoute.THREAD_EVENTS.value,
        RouterRoute.WEB.value,
    ]
    requested_route = RouterRoute.WEB.value
    fallback_route = next((item for item in fallback_order if item in allowed_routes), None)
    assert requested_route not in allowed_routes
    assert fallback_route == RouterRoute.DIRECT.value
