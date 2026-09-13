# Runtime capability matrix

The control plane remains authoritative for product records while framework
execution is owned by the selected external runtime.

| Behavior | Product/runtime owner | Persisted compatibility |
| --- | --- | --- |
| Start and terminal success | AgentRunService and selected runtime | AgentRun, ChatTurn, workflow/version IDs |
| Stream ordering and terminal event | Event sink and runtime API | Event IDs and trace payload |
| Clarification | Product projection and runtime | Clarification turn and continuation cleanup |
| Human interrupt | Product projection and runtime | pending_interrupt_json and opaque binding |
| Resume | Product service through runtime adapter | Same run ID and opaque binding |
| Cancellation | Product service and runtime | Cancelled status and cleanup behavior |
| Deep-research tasks | Task repository and selected runtime | Task/run/todo/artifact linkage |
| Runtime failures | Runtime adapter and API | Typed error payload and terminal state |
| Trace persistence/redaction | Trace recorder and product projection | debug_trace_json and trace references |
| Checkpoint pruning | Runtime administration | Paused checkpoints retained |
| Workflow compatibility | Workflow store and thread services | Stable IDs, versions, settings, fork metadata |
| MCP behavior | MCP adapters and tool contracts | Tool names and context propagation |

Capability responses are deployment-specific, definition-specific, and
run-state-specific. Never infer support from a shared endpoint name alone.
