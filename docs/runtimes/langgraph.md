# LangGraph runtime

askPDF uses a strict external LangGraph execution boundary. The control plane
does not import LangGraph execution code or access runtime persistence.

## Responsibilities

langgraph-runtime owns:

- workflow validation and compilation
- graph and Deep Agent execution
- framework interrupts
- Postgres checkpoints
- execution leases and event journals
- dependency discovery
- opaque continuation bindings
- runtime recovery and checkpoint administration

The product control plane owns user authorization, product workflow records,
task lifecycle, budgets, artifacts, tool policy, and trace projection.

When a workflow admits `research_canvas_publish`, answer and synthesis nodes
may call `publish_canvas` once the evidence pass is done. Retrieval workers
do not. Flattened tool arguments are normalized to `canvas_spec_v1` before
MCP. See [Research canvases](../research-canvases.md).

## Configuration

Production runtime deployments require:

- ASKPDF_AGENT_CHECKPOINTER=postgres
- AGENT_CHECKPOINT_DATABASE_URL
- AGENT_RUNTIME_EXECUTION_DATABASE_URL
- LANGGRAPH_RUNTIME_BINDING_SECRET

Runtime-only credentials must not be supplied to the control-plane container.
The runtime fails closed when durable checkpoint storage or required migration
tables are unavailable.

## HTTP/SSE behavior

The runtime provides authenticated /v1 endpoints for:

- run start
- event streaming and replay
- resume and continuation
- cancellation and pause
- state inspection
- dependency status
- cleanup and administration

The control-plane HTTP adapter reconnects to the durable events endpoint after
bounded SSE failures. Runtime operations use idempotency identifiers and
request fingerprints. Identical retries replay stored results; reused
identifiers with different input are conflicts.

## Recovery

The runtime journal reclaims expired leases and replays events by cursor. The
control plane deduplicates canonical runtime events and task deltas. Uncertain
product outcomes are resolved through the remote inspect endpoint rather than
by querying runtime tables.

Checkpoint deletion is performed from the LangGraph runtime image:

    python -m langgraph_runtime.admin --delete-thread ID --dry-run
    python -m langgraph_runtime.admin --delete-thread ID --confirm

The protocol deliberately does not expose raw checkpoint identifiers.
