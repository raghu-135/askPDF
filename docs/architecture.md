# Architecture

## Service topology

    Browser
      ↓
    Next.js frontend
      ↓ /api/backend proxy and SSE
    FastAPI control plane (rag_service)
      ├── PostgreSQL
      ├── Weaviate
      ├── local/OpenAI-compatible model provider
      ├── browser-capture service
      ├── langgraph-runtime
      └── hermes-runtime

The Compose topology and service ports are defined in docker-compose.yml.
LangGraph and Hermes listen only on the Compose network. The control plane
is bound to localhost. Weaviate and PostgreSQL are published on the host in
the example file.

## Responsibilities

### Frontend

The frontend is a client-heavy Next.js application. The main workspace is
assembled in frontend/src/pages/index.tsx. Chat, thread/project navigation,
PDF viewing, memory management, workflow building, traces, research canvases
(first-party `canvas_spec_v1` renderer, not a TSX compiler), and deep-research
panels are implemented under frontend/src/components.

The frontend normally calls the Next server proxy at /api/backend. The proxy
forwards requests to the control plane and adds the server-side admin bearer
token.

### Control plane

rag_service owns product APIs, authentication, projects, threads, files,
parsing and indexing jobs, workflow definitions, task state, artifacts,
memories, MCP tool authorization, and product-facing run projections.

Document conversion, embedding materialization, and AgentTask workers run
inside the rag-service process. Runtimes call back into rag-service over
loopback HTTP MCP for tools.

The main application setup and router mounting are in rag_service/main.py.

### Agent runtimes

The external runtimes are accessed through runtime_protocol and HTTP/SSE
adapters. The control plane does not import LangGraph execution code or access
runtime checkpoint tables.

LangGraph owns graph validation, compilation, execution, checkpoints, runtime
leases, event journaling, and recovery. Hermes provides a separate gateway
with a smaller capability surface.

### Persistence

PostgreSQL is authoritative for product identity, relationships, settings,
messages, workflows, runs, tasks, approvals, artifacts, and memories.
Weaviate stores derived, model-specific retrieval vectors. Runtime databases
store execution/checkpoint state owned by their respective runtime.

## End-to-end data flow

### Document flow

    upload or browser capture
      → content store
      → parse and extract sentences/bounding boxes
      → persist metadata and processing status
      → embedding job
      → model-specific Weaviate collection
      → retrieval during chat or task execution

### Chat flow

    frontend chat request
      → control-plane embedding readiness check
      → thread settings and workflow resolution
      → normalized frozen AgentRun
      → prompt assembly (layout skills only if publish_canvas is admitted)
      → LangGraph/Hermes adapter
      → retrieval MCP tools, then optional publish_canvas from answer/synthesis
      → canonical runtime events
      → product ChatTurn projection (optional canvas_ref)
      → optional research_canvas artifact in the content store
      → JSON response or SSE stream

### Long-running task flow

    task creation with Idempotency-Key
      → durable AgentTask
      → task worker and AgentRun attempt
      → plans, todos, subagents, artifacts, events, and reviews
      → runtime execution and recovery
      → final artifact
      → optional publication as a ChatTurn

## Important boundaries

- Product APIs never expose framework checkpoint identifiers.
- Runtime bindings are opaque to the product.
- Tool approval is enforced at the MCP boundary, not inside individual tools.
- Built-in workflow definitions are seeded and refreshed by the control plane.
- Embedding model identity is part of the project/thread data contract.
- Remote embedding packing uses a locally cached tokenizer; unknown models
  are rejected rather than downloaded at query time.

Related implementation:

- rag_service/main.py
- rag_service/app/product_orchestration/
- rag_service/app/runtime/
- rag_service/app/services/embedding_tokenizer_registry.py
- rag_service/scripts/download_embedding_tokenizers.py
- [Research canvases](research-canvases.md)
- runtime_protocol/
- langgraph_runtime/api.py
- hermes_runtime/api.py
