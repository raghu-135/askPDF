# External LangGraph runtime

askPDF uses a strict external execution boundary:

```text
frontend -> askPDF product APIs -> runtime_protocol HTTP/SSE -> langgraph-runtime
                              \-> Hermes HTTP adapter -> hermes-runtime
```

The control plane owns declarative workflow definitions and versions, product tool policy, task leases and commands, plans/todos/subagent records, budgets, artifacts, user authorization, and product trace projections. It serializes an immutable `RuntimeTaskContext` and transactionally applies idempotent `TaskOrchestrationDelta` responses.

`langgraph-runtime` owns LangGraph validation and compilation, graph and Deep Agent execution, framework interrupts, checkpoints, execution leases/journals, dependency discovery, and opaque continuation bindings. It has its own database credentials and never imports `app` or accesses product persistence.

`runtime_protocol` is the only shared Python package. Its values are JSON-only and versioned. Checkpoint references are signed inside `langgraph-runtime`; the control plane stores only an opaque `binding_id` and exposes only `checkpoint_boundary_available`.

## Local operation

Run the control plane and runtime through Compose. A separately launched control plane must set `LANGGRAPH_RUNTIME_URL` and the same `LANGGRAPH_RUNTIME_TOKEN` as the runtime. There is no in-process mode. Unit tests use a fake HTTP runtime; framework tests run with runtime dependencies.

Runtime-only settings include `ASKPDF_AGENT_CHECKPOINTER`, `AGENT_CHECKPOINT_DATABASE_URL`, `AGENT_RUNTIME_EXECUTION_DATABASE_URL`, and `LANGGRAPH_RUNTIME_BINDING_SECRET`. Product database and object-store credentials must not be supplied to the runtime container.

## Recovery and administration

The runtime journal reclaims expired execution leases and replays events by cursor. The control plane deduplicates canonical runtime events and task deltas. An uncertain product outcome is resolved with the remote inspect endpoint; the control plane never queries runtime tables.

Checkpoint deletion is available only through `python -m langgraph_runtime.admin --delete-thread ID --dry-run` followed by the same command with `--confirm`. Legacy checkpoint references can be converted through authenticated `POST /v1/admin/bindings/migrate`; failed conversions must be surfaced for operator intervention rather than restarted.
