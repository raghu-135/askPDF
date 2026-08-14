# Agent runtime Phase 1 behavior matrix

This matrix records the compatibility surface protected before runtime
extraction. The existing workflow implementation remains authoritative during
Phase 1.

| Behavior | Current owner | Persisted compatibility | Regression coverage |
| --- | --- | --- | --- |
| Start and terminal success | `AgentRunService.run_thread_chat` | `AgentRun`, `ChatTurn`, workflow/version IDs | `test_agent_workflows_pytest.py` run-service cases |
| Stream ordering and terminal event | `AgentExecutionEventSink`, workflow API | event IDs and trace payload | `test_agent_workflows_pytest.py`, `test_parallel_agent_runtime_pytest.py` |
| Clarification | router runtime and run service | clarification turn plus checkpoint cleanup | run-service clarification tests |
| Human interrupt | repository interrupt resolver | `pending_interrupt_json`, checkpoint thread ID | pending-interrupt and HITL tests |
| Resume | `AgentRunService.resume_agent_run` | same run ID and checkpoint binding | resume guard, duplicate, stale, and invalid request tests |
| Cancellation | `chat_cancellation`, run service | cancelled status and cleanup behavior | cancellation and cleanup tests |
| Deep-research task execution | `agent_task_runtime`, task repository | task/run/todo/artifact linkage | `test_deep_research_tasks_pytest.py` |
| Runtime failures | run service and router runtime | typed error payload, terminal run state | failed-run and validation tests |
| Trace persistence/redaction | trace recorder and debug trace modules | `debug_trace_json`, `agent_trace_refs_json` | trace schema, replay, and redaction tests |
| Checkpoint pruning | `run_cleanup`, checkpointing | paused checkpoints retained; terminal checkpoints removed | checkpoint pruning tests |
| Thread/workflow compatibility | workflow store and thread services | stable workflow IDs, versions, settings, fork metadata | fork, workflow store, and API tests |
| MCP behavior | MCP adapters and tool contracts | existing tool names and context propagation | MCP contract and context tests |

Phase 1 additions are deliberately additive: framework/builder identity and
opaque runtime binding metadata are stored alongside the existing fields, while
execution remains on the current LangGraph-backed service.
