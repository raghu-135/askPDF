# Runtime operations

## Startup and readiness

The control plane, LangGraph runtime, Hermes runtime, and browser capture
service each have separate health/readiness behavior. A service may be alive
while not ready to accept work because a database, MCP endpoint, model
provider, checkpoint store, or upstream runtime is unavailable.

Use the service-specific health checks defined in docker-compose.yml. Those
probes run inside each container. The host-published ports in the example
stack are frontend `:3000`, control plane `127.0.0.1:8000`, browser capture
`127.0.0.1:8090`, Weaviate `:8080`, and PostgreSQL `:5432`. LangGraph and
Hermes remain on the Compose network. The default stack does not publish
LangGraph `:8100` or Hermes `:8200` on the host.

## Leases and fencing

Every active runtime execution has an owner, lease, and fencing token. A worker
that loses its lease must stop appending events or finalizing the run.

## Reconnect and replay

The control-plane adapter reconnects to durable event streams using bounded
attempts, backoff, and deadlines. Runtime event streams are replayable by
cursor. Product event projection deduplicates canonical events.

## Operations and idempotency

Runtime operations record request fingerprints, attempts, status, and results.
Identical retries replay the original result. A reused operation ID with
different input is rejected.

Product task commands use the task.start, task.pause, task.resume, task.cancel,
and task.retry namespace. Runtime continuation operations use the run namespace.

## Cleanup

Runtime checkpoint cleanup must be requested through the runtime boundary.
Product cleanup must not issue raw SQL against runtime-owned tables. Paused
human-review checkpoints are retained until the continuation or expiration
policy resolves them.

## Hermes-specific operations

Hermes cancellation uses bounded terminal confirmation. Hermes currently
capability-disables active-run course correction and some continuation controls.
Its bundled execution store is single-worker/single-replica.

Implementation details and tunables are in:

- rag_service/app/runtime/
- langgraph_runtime/
- hermes_runtime/
- .env.example
