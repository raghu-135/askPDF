# API overview

The control-plane routers are mounted under /api in rag_service/main.py.
FastAPI’s generated OpenAPI schema is the detailed request/response reference;
this page documents the stable endpoint groups and important behavior.

## Endpoint groups

| Group | Examples |
| --- | --- |
| Projects | GET/POST /api/projects, clone, lifecycle, project threads |
| Threads | GET/POST /api/threads, fork, settings, indexing status |
| Files | upload, attach, download, parse status, delete, annotations |
| Chat | POST /api/threads/{thread_id}/chat |
| Canvases | POST/GET /api/threads/{thread_id}/canvases |
| Memories | scoped search, review, delete, retry indexing |
| Memory manager | plans, apply, reviews, continuation, status |
| Workflows | catalog, validation, source, save, delete |
| Agent runs | events, state, follow-ups, interrupts, steering, cancel, resume |
| Agent tasks | create, commands, todos, runs, artifacts, evidence, reviews |
| Models | model list and chat/embedding health |
| Tools | tool contract discovery |
| Operations | /health, /ready |

## Chat

POST /api/threads/{thread_id}/chat accepts a ThreadChatRequest containing the
question, LLM model, web-search/reranker settings, context window, replanning,
prompt overrides, and client locale/time metadata.

Without an Accept: text/event-stream header, the endpoint returns a completed
product projection. With that header, it returns text/event-stream events with
heartbeats, canonical run events, and a terminal result.

The implementation is in rag_service/app/api/messages.py.

## Research canvases

POST /api/threads/{thread_id}/canvases stores a versioned canvas_spec_v1 document
as a `research_canvas` artifact (`application/vnd.askpdf.canvas+json`) in the
shared content store. Chat-only requests persist one without an AgentTask,
the same way ChatTurn is a product projection. GET lists current canvases;
GET by id returns one document. Revisions use supersedes_id. The renderer only
accepts typed blocks (stat, table, callout, markdown, sources, dag).

Agents emit canvases with the first-party `publish_canvas` MCP tool
(`research_canvas_publish`). Admission rejects unknown blocks and requires a
sources block whose document `file_hash` values are attached to the thread.
Thread setting `hitl_canvas_publish` asks a human before that durable write.

When `publish_canvas` / `research_canvas_publish` is admitted, the agent
also receives prompt-only layout skills (`canvas_compare_papers`,
`canvas_evidence_matrix`, `canvas_timeline`). Those are recipes over the
existing blocks, not new runtimes or block types.

The implementation is in rag_service/app/models/canvas.py,
rag_service/app/tools/publish_canvas.py, and rag_service/app/api/canvases.py.

## Long-running tasks

Task creation and mutating commands require Idempotency-Key. Task reads are
scoped by task, thread, and current principal. Commands include start, pause,
resume, cancel, and retry. Task events, artifacts, evidence, timelines, and
reviews are separate read/write surfaces.

The implementation is in rag_service/app/api/agent_tasks.py.

## Runtime APIs

The LangGraph runtime exposes authenticated /v1 endpoints for starting runs,
streaming events, resuming, cancellation, pause, inspection, dependencies,
and cleanup. These APIs are service-to-service contracts, not browser APIs.

See [Runtime protocol](contracts/runtime-protocol.md) and
[LangGraph runtime](runtimes/langgraph.md).

## Internal APIs

The control plane mounts internal MCP and Hermes-MCP surfaces. They are not
public application endpoints and require runtime execution credentials at their
own boundary.

When adding an endpoint:

1. Define explicit Pydantic request/response models.
2. Enforce resource ownership where applicable.
3. Keep business logic in a service/repository rather than the router.
4. Add idempotency for retriable mutations.
5. Add API and failure-path coverage to the test inventory.
