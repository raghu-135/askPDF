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

## Responsibilities

### Frontend

The frontend is a client-heavy Next.js application. The main workspace is
assembled in frontend/src/pages/index.tsx. Chat, thread/project navigation,
PDF viewing, memory management, workflow building, traces, research canvases,
and deep-research
panels are implemented under frontend/src/components.

The frontend normally calls the Next server proxy at /api/backend. The proxy
forwards requests to the control plane and adds the server-side admin bearer
token.

### Control plane

rag_service owns product APIs, authentication, projects, threads, files,
parsing and indexing jobs, workflow definitions, task state, artifacts,
memories, MCP tool authorization, and product-facing run projections.

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
      → LangGraph/Hermes adapter
      → runtime MCP tool calls
      → canonical runtime events
      → product ChatTurn projection
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

Related implementation:

- rag_service/main.py
- rag_service/app/product_orchestration/
- rag_service/app/runtime/
- runtime_protocol/
- langgraph_runtime/api.py
- hermes_runtime/api.py
