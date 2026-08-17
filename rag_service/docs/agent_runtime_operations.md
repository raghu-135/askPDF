# Agent runtime operational guardrails

## LangGraph runtime

The runtime execution store uses PostgreSQL and assigns one owner, lease, and
fencing token to every active execution. A worker that loses its lease must
stop and may not append events or finalize a run.

Runtime schema auto-creation is disabled by default. Production deployments
must provision `runtime_executions` and `runtime_events` through a migration or
bootstrap job before starting the runtime. The Compose proof explicitly sets
`AGENT_RUNTIME_SCHEMA_AUTO_CREATE=true`.

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
