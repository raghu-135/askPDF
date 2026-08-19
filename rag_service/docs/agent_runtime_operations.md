# Agent runtime operational guardrails

## LangGraph runtime

The runtime execution store uses PostgreSQL and assigns one owner, lease, and
fencing token to every active execution. A worker that loses its lease must
stop and may not append events or finalize a run.

Runtime schema auto-creation is disabled by default. Production deployments
must provision `runtime_executions` and `runtime_events` through a migration or
bootstrap job before starting the runtime. The Compose proof explicitly sets
`AGENT_RUNTIME_SCHEMA_AUTO_CREATE=true`.

The runtime lease migration is `0d7e4a9b2c1f`. Apply the normal application
migrations with `DATABASE_URL`, then apply the runtime migrations with the
dedicated bootstrap command:

```bash
cd rag_service
AGENT_RUNTIME_EXECUTION_DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgresql:5432/runtime_checkpoints \
python -m app.db.migrate_runtime
```

The bootstrap stamps legacy runtime databases at the pre-runtime baseline and
then runs the runtime revision using the standard `DATABASE_URL` convention.
The migration is guarded so it is safe for the
application database, where runtime-owned tables are not present. Existing
runtime tables receive the lease/fencing columns without losing checkpoints or
execution records.

The control-plane HTTP adapter reconnects to the durable events endpoint after
an SSE transport failure. `AGENT_RUNTIME_RECONNECT_MAX_ATTEMPTS`,
`AGENT_RUNTIME_RECONNECT_BACKOFF_SECONDS`, and
`AGENT_RUNTIME_RECONNECT_DEADLINE_SECONDS` bound this recovery.

## Hermes runtime integration

The upstream contract is pinned to NousResearch/hermes-agent commit
`bdd0a79c6a0ebc2344d5d6913c70bd89fa59c894`. The verified surface is
`POST /v1/runs`, `GET /v1/runs/{id}`, `GET /v1/runs/{id}/events`, and the
run-scoped `/approval`, `/steer`, and `/stop` operations. askPDF sends the user
input at top level, maps `system_prompt` to `instructions`, and captures the
Hermes `session_id` from run status into its opaque continuation binding.

Supported upstream events are `message.delta`, `tool.started`,
`tool.completed`, `reasoning.available`, `approval.request`,
`approval.responded`, `run.steered`, `run.completed`, `run.failed`,
`run.cancelled`, `subagent.start`, and `subagent.complete`. Unknown events are
retained as bounded `runtime.event` records for forward compatibility.

Hermes definitions use `definition_version: 1`. The Hermes builder resolves
them into a deterministic `profile_version: 1` managed profile containing MCP
and tool policy, model/provider policy, skills, memory, delegation, and limits.
The profile hash is reproducible. API keys, tokens, passwords, credentials, and
other provider secrets are rejected from definitions and must be supplied via
the runtime environment.

The bundled gateway currently uses an atomic whole-file journal and is intentionally limited
to one worker and one replica. `HERMES_RUNTIME_STORAGE_BACKEND=file` and
`HERMES_RUNTIME_WORKERS=1` are enforced at startup. A production Hermes
deployment must replace this store with the PostgreSQL execution-store
contract before scaling horizontally.

It rewrites the complete journal for every state or event update and retains
events indefinitely. Do not run multiple workers or replicas; replace the file
journal with PostgreSQL or another shared transactional store with a retention
policy before horizontal production rollout.

Hermes and the askPDF adapter are part of the default Compose application.
Configure `API_SERVER_KEY` in the shared `.env`, then start normally. Each run
inherits the thread's selected `llm_model` and uses askPDF's existing
OpenAI-compatible `LLM_API_URL` through Hermes's custom provider:

```bash
docker compose up --build
```

`/healthz` is liveness-only. Compose and deployment readiness use `/readyz`,
which requires both the Hermes upstream and the configured MCP dependency.

`./run_tests.sh --phase7` builds the same pinned Hermes revision and runs it
against an isolated deterministic OpenAI-compatible provider. Rare event and
protocol failures use checked-in data fixtures and mocked transports; there is
no executable fake Hermes runtime or fallback path.

```bash
./run_tests.sh --phase7
```
