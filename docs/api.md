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
