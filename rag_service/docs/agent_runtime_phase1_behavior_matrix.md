# Agent runtime Phase 1 behavior matrix

This matrix records the product behavior protected by the external runtime
boundary. The control plane remains authoritative for product records while
framework execution is owned by `langgraph-runtime`.

| Behavior | Current owner | Persisted compatibility | Regression coverage |
| --- | --- | --- | --- |
| Start and terminal success | `AgentRunService.run_thread_chat` | `AgentRun`, `ChatTurn`, workflow/version IDs | `test_agent_workflows_pytest.py` run-service cases |
| Stream ordering and terminal event | `AgentExecutionEventSink`, workflow API | event IDs and trace payload | `test_agent_workflows_pytest.py`, `test_parallel_agent_runtime_pytest.py` |
| Clarification | control-plane projection and remote runtime | clarification turn plus remote continuation cleanup | run-service clarification tests |
| Human interrupt | neutral product interrupt projection and remote runtime | `pending_interrupt_json`, opaque runtime binding | pending-interrupt and HITL tests |
| Resume | `AgentRunService.resume_agent_run` through HTTP | same run ID and opaque runtime binding | resume guard, duplicate, stale, and invalid request tests |
| Cancellation | `chat_cancellation`, run service | cancelled status and cleanup behavior | cancellation and cleanup tests |
| Deep-research task execution | `agent_task_runtime`, task repository | task/run/todo/artifact linkage | `test_deep_research_tasks_pytest.py` |
| Runtime failures | run service and router runtime | typed error payload, terminal run state | failed-run and validation tests |
| Trace persistence/redaction | trace recorder and debug trace modules | `debug_trace_json`, `agent_trace_refs_json` | trace schema, replay, and redaction tests |
| Checkpoint pruning | remote runtime administration | paused checkpoints retained; terminal checkpoints removed | runtime checkpoint tests |
| Thread/workflow compatibility | workflow store and thread services | stable workflow IDs, versions, settings, fork metadata | fork, workflow store, and API tests |
| MCP behavior | MCP adapters and tool contracts | existing tool names and context propagation | MCP contract and context tests |

Framework/builder identity and opaque runtime binding metadata are product
records, while graph compilation, execution, interrupts, and checkpoints stay
inside the external runtime service.
