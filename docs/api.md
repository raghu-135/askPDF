# API overview

The control-plane routers are mounted under /api in rag_service/main.py.
FastAPI’s generated OpenAPI schema is the detailed request/response reference;
this page documents the stable endpoint groups and important behavior.

## Endpoint groups

| Group | Examples |
| --- | --- |
| Projects | GET/POST /api/projects, clone, lifecycle, project threads |
| Threads | GET/POST /api/threads, fork, settings, indexing status, embeddings projection |
| Files | upload, attach, download, parse status, chunks inspection, delete, annotations |
| Chat | POST /api/threads/{thread_id}/chat |
| Canvases | POST/GET /api/threads/{thread_id}/canvases |
| Memories | scoped search, review, delete, retry indexing |
| Memory manager | plans, apply, reviews, continuation, status |
| Workflows | catalog, validation, source, save, delete |
| Agent runtimes | GET /api/agent-runtimes, runtime and workflow capabilities |
| Agent runs | events, state, follow-ups, interrupts, steering, cancel, resume |
| Prompt preview | GET /api/threads/prompt-tools, POST /api/threads/prompt-preview |
| Agent tasks | create, commands, todos, runs, artifacts, evidence, reviews |
| Models | model list and chat/embedding health |
| Tools | tool contract discovery |
| Operations | /health, /ready |

## Embeddings projection

GET /api/threads/{thread_id}/embeddings-projection returns 3D-projected
document chunk embeddings along with sequential and semantic similarity edges
for the thread workspace viewer. Optional query parameters: `file_hash`,
`source_kind`, and `limit` (default 300, max 1000).

The implementation is in rag_service/app/api/threads.py.

## Chat

POST /api/threads/{thread_id}/chat accepts a ThreadChatRequest containing the
question, LLM model, web-search/reranker settings, context window, replanning,
prompt overrides, and client locale/time metadata.

Without an Accept: text/event-stream header, the endpoint returns a completed
product projection. With that header, it returns text/event-stream events with
heartbeats, canonical run events, and a terminal result.

The implementation is in rag_service/app/api/messages.py.

## Research canvases

POST /api/threads/{thread_id}/canvases stores a `canvas_spec_v1` document as a
`research_canvas` artifact. GET lists current canvases (`current_only` defaults
true); GET by id returns one document. Chat responses may include `canvas_ref`
on the assistant turn. Revisions use `supersedes_id`.

Agent emit, admission, layout skills, and the workbench tab are documented in
[Research canvases](research-canvases.md).

The routers are rag_service/app/api/canvases.py and the `canvas_ref` projection
in rag_service/app/api/messages.py.

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
own boundary. Workflow-builder test-run and catalog routes under
`/internal/agent-workflows` are similarly not browser APIs.

When adding an endpoint:

1. Define explicit Pydantic request/response models.
2. Enforce resource ownership where applicable.
3. Keep business logic in a service/repository rather than the router.
4. Add idempotency for retriable mutations.
5. Add API and failure-path coverage to the test inventory.
