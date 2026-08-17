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

## Hermes Phase 7 proof

Hermes currently uses an atomic whole-file journal and is intentionally limited
to one worker and one replica. `HERMES_RUNTIME_STORAGE_BACKEND=file` and
`HERMES_RUNTIME_WORKERS=1` are enforced at startup. A production Hermes
deployment must replace this store with the PostgreSQL execution-store
contract before scaling horizontally.

This gateway is a development/proof runtime, not a production deployment. It
rewrites the complete journal for every state or event update and retains all
events indefinitely. Do not run multiple workers or replicas, and do not use it
in production until the file journal is replaced by PostgreSQL or another
shared transactional store with a defined retention policy.

Hermes is excluded from the default Compose application. Start it explicitly
with a reachable, separately managed Hermes API:

```bash
HERMES_API_URL=http://host.docker.internal:<port> \
docker compose --profile second-runtime-proof up
```

`/healthz` is liveness-only. Compose and deployment readiness use `/readyz`,
which requires both the Hermes upstream and the configured MCP dependency.
