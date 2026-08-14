"""Runtime context passed to framework-neutral tools."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class ToolInvocationContext:
    thread_id: str | None = None
    run_id: str | None = None
    tool_call_id: str | None = None
    mcp_request_id: str | None = None
    deadline_at: datetime | None = None
    cancellation_token: Any = None
    cancellation_scope_id: str | None = None
    prefetched_durable_memories: Any = None
    prefetched_durable_memory_scopes: Any = None
    prefetched_durable_memory_scope_policy: Any = None
    prefetched_durable_memory_debug: Any = None
    prefetched_durable_memory_query_vector: Any = None
    caller_node: str | None = None
    caller_node_type: str | None = None
    caller_capabilities: tuple[str, ...] = ()
    route: str | None = None
    embedding_model: str | None = None
    context_window: int | None = None
    use_web_search: bool = False
    use_reranker: bool = True
    web_search_index: bool = True
    client_timezone: str | None = None
    client_locale: str | None = None
    client_now_iso: str | None = None
    traceparent: str | None = None
    tracestate: str | None = None
    # Future authorization seams. They are intentionally not enforced yet.
    principal: str | None = None
    tenant_id: str | None = None
    scopes: tuple[str, ...] = ()
    extensions: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ToolInvocationContext":
        data = dict(value or {})
        scopes = data.get("scopes", ())
        if isinstance(scopes, str):
            scopes = (scopes,)
        elif not isinstance(scopes, (list, tuple)):
            scopes = ()
        return cls(
            thread_id=data.get("thread_id"),
            run_id=data.get("run_id"),
            tool_call_id=data.get("tool_call_id"),
            mcp_request_id=data.get("mcp_request_id"),
            deadline_at=data.get("deadline_at"),
            cancellation_token=data.get("cancellation_token"),
            cancellation_scope_id=data.get("cancellation_scope_id"),
            prefetched_durable_memories=data.get("prefetched_durable_memories"),
            prefetched_durable_memory_scopes=data.get("prefetched_durable_memory_scopes"),
            prefetched_durable_memory_scope_policy=data.get("prefetched_durable_memory_scope_policy"),
            prefetched_durable_memory_debug=data.get("prefetched_durable_memory_debug"),
            prefetched_durable_memory_query_vector=data.get("prefetched_durable_memory_query_vector"),
            caller_node=data.get("caller_node"),
            caller_node_type=data.get("caller_node_type"),
            caller_capabilities=tuple(str(item) for item in (data.get("caller_capabilities") or ()) if item),
            route=data.get("route"),
            embedding_model=data.get("embedding_model"),
            context_window=data.get("context_window"),
            use_web_search=bool(data.get("use_web_search", False)),
            use_reranker=bool(data.get("use_reranker", True)),
            web_search_index=bool(data.get("web_search_index", True)),
            client_timezone=data.get("client_timezone"),
            client_locale=data.get("client_locale"),
            client_now_iso=data.get("client_now_iso"),
            traceparent=data.get("traceparent"),
            tracestate=data.get("tracestate"),
            principal=data.get("principal"),
            tenant_id=data.get("tenant_id"),
            scopes=tuple(str(item) for item in scopes),
            extensions=dict(data.get("extensions") or {}) if isinstance(data.get("extensions"), Mapping) else None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "tool_call_id": self.tool_call_id,
            "mcp_request_id": self.mcp_request_id,
            "deadline_at": self.deadline_at.isoformat() if isinstance(self.deadline_at, datetime) else self.deadline_at,
            "cancellation_scope_id": self.cancellation_scope_id,
            "prefetched_durable_memories": self.prefetched_durable_memories,
            "prefetched_durable_memory_scopes": self.prefetched_durable_memory_scopes,
            "prefetched_durable_memory_scope_policy": self.prefetched_durable_memory_scope_policy,
            "prefetched_durable_memory_debug": self.prefetched_durable_memory_debug,
            "prefetched_durable_memory_query_vector": self.prefetched_durable_memory_query_vector,
            "caller_node": self.caller_node,
            "caller_node_type": self.caller_node_type,
            "caller_capabilities": list(self.caller_capabilities),
            "route": self.route,
            "embedding_model": self.embedding_model,
            "context_window": self.context_window,
            "use_web_search": self.use_web_search,
            "use_reranker": self.use_reranker,
            "web_search_index": self.web_search_index,
            "client_timezone": self.client_timezone,
            "client_locale": self.client_locale,
            "client_now_iso": self.client_now_iso,
            "traceparent": self.traceparent,
            "tracestate": self.tracestate,
            "principal": self.principal,
            "tenant_id": self.tenant_id,
            "scopes": list(self.scopes),
            "extensions": self.extensions,
        }
