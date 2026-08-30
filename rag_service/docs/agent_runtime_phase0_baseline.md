# Agent runtime Phase 0 baseline and contract freeze

Phase 0 freezes the externally observable behavior that subsequent runtime
extraction must preserve. It does not move implementation ownership, alter the
database schema, or change production runtime selection.

## Frozen compatibility boundary

Contract version 1 consists of:

- the JSON shapes emitted by `app.runtime.contracts`;
- the request/result/event conversion and SSE framing in
  `app.runtime.transport`;
- the internal HTTP routes exposed by the LangGraph and Hermes runtime apps;
- stable event identity (`run_id`, `attempt`, `sequence`, `event_id`) and
  continuation bindings;
- the product behavior listed in `agent_runtime_phase1_behavior_matrix.md`.

The executable fixture is
`tests/fixtures/runtime_contract_v1.json`. The fixture is intentionally exact,
including null/default fields and SSE field order. A change to that fixture
requires one of the following:

1. proof that the old representation remains accepted and emitted where v1 was
   negotiated; or
2. a new contract version, migration notes, and dual-version contract tests.

Adding an endpoint or field may be operationally safe, but it still requires an
explicit fixture review so service owners do not acquire an accidental API.

## Phase 0 ownership record

| Boundary | Baseline owner | Authoritative persistence |
| --- | --- | --- |
| Product APIs, thread/chat/task lifecycle, run projection | `rag_service/app/api`, `app/services` | product PostgreSQL |
| Neutral runtime objects and HTTP connector | `rag_service/app/runtime` | none |
| Current LangGraph execution HTTP service | `rag_service/runtime_service` plus legacy `app/agent_workflows` | runtime execution DB and LangGraph checkpoint DB |
| Hermes HTTP gateway | repository-root `hermes_runtime` | file-backed execution journal |
| Tools, document retrieval, memory, MCP and authorization context | `rag_service/app/tools`, `app/mcp`, product services | product stores |
| Runtime events | runtime journal first, projected to `agent_run_events` | runtime DB and product PostgreSQL |

## Known inconsistencies frozen for migration, not endorsed as target design

- `runtime_service` imports `app.agent_workflows`, reconstructs legacy
  run-shaped objects, and its image copies the whole control-plane `app` tree.
- runtime dependency manifests inherit control-plane dependencies.
- task graph nodes still contain product repository/service integrations, with
  execution-mode branches suppressing writes in the external process.
- runtime migrations reuse the product Alembic chain even though the databases
  have different ownership.
- `RuntimeExecutionContext` and serialized context carry untyped legacy state;
  request authentication can be persisted with execution payloads.
- Hermes is currently a gateway to an external API, not a repository-owned
  native executor, and its file journal is single-replica proof storage.
- Hermes exposes unsupported resume/continue/cleanup routes as typed failures;
  the route remains part of v1 while capabilities advertise support as false.
- production and test paths still retain `AGENT_RUNTIME_MODE=in_process`.

These are migration inputs. Later phases must not preserve their internal
coupling merely because their current wire behavior is frozen.

## Acceptance gate

Run the focused gate with:

```sh
./run_tests.sh --runtime-contract
```

Phase 0 is accepted when:

- the v1 fixture and HTTP surface tests pass;
- existing neutral-contract, HTTP-adapter, runtime-registry, MCP-contract, and
  control-plane import-boundary tests pass;
- the existing LangGraph external-runtime smoke gate and Hermes proof remain
  available for environment-backed verification;
- no schema migration or production configuration change is present.

The rollback checkpoint is the commit immediately before Phase 1 extraction.
Rollback consists only of reverting Phase 0 documentation, fixture, test, and
test-runner changes; no data rollback is required.

## Working-tree preservation

The worktree was clean at Phase 0 inspection. Future phases must continue to
check `git status --short` before editing and preserve any unrelated user-owned
changes rather than normalizing or reverting them.
