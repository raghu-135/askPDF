# Hermes runtime

Hermes is a separate runtime gateway around a pinned upstream
NousResearch/hermes-agent revision. `HERMES_UPSTREAM_REVISION` in `.env`
(and `.env.example`) is the source of truth; `docker-compose.yml` builds
from that SHA and must stay synchronized with the adapter contract.

## Responsibilities

hermes-runtime owns upstream Hermes execution, managed profiles, native event
delivery, native approval integration, and its execution store. The askPDF
adapter translates product workflow definitions and runtime events into the
shared runtime_protocol contract.

Hermes definitions resolve into deterministic managed profiles containing MCP
and tool policy, model/provider policy, skills, memory, delegation, and limits.
Credentials are environment-owned and must not appear in definitions.

When `publish_canvas` is in `allowed_tool_ids`, resolve appends layout-skill
ids and the canvas layout prompt to instructions. Those skills are omitted
when the tool is not admitted. See [Research canvases](../research-canvases.md).

## Capabilities and limitations

Hermes supports execution, event streaming, approvals, and cooperative
cancellation. In the current v1 adapter it does not support every LangGraph
operation, including active-run course correction, general state updates,
follow-ups, or the full pause/resume surface.

The product capability gate must be consulted before exposing or invoking an
operation. See [Capability matrix](capability-matrix.md).

## Storage and scaling

The bundled gateway uses a file-backed atomic journal and is intentionally
single-worker/single-replica. Do not horizontally scale it without replacing
the journal with a shared transactional store and adding retention policy.

## Configuration

When enabled, configure distinct values for:

- HERMES_RUNTIME_TOKEN
- HERMES_API_TOKEN
- MCP_EXECUTION_CONTEXT_SECRET
- OPENAI_API_KEY when the shared LLM server requires a Bearer token

Hermes uses the selected askPDF thread model and the existing
OpenAI-compatible LLM_API_URL. The gateway talks to that URL as Hermes'
`custom` provider (`OPENAI_BASE_URL`). Pinned Hermes requires
`HERMES_MODEL_CONTEXT_LENGTH` of at least 64000; that value is written into
`model.context_length` so a catalog listing of 32k cannot reject a larger
hosted model. Readiness verifies both the upstream service and
the configured MCP dependency.

## Security boundary

Runtime and upstream Hermes ports should remain on the private Compose network.
The model receives tool capability through the managed profile and does not
receive the MCP execution credential.
