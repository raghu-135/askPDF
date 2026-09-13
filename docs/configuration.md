# Configuration

The file .env.example is the authoritative inventory of supported environment
variables. This page explains the main groups and deployment boundaries; add a
variable to the example file before documenting it here.

## Required boundaries

| Variable | Owner | Purpose |
| --- | --- | --- |
| DATABASE_URL | Control plane | Product PostgreSQL |
| TEST_DATABASE_URL | Test runner | Isolated test PostgreSQL |
| WEAVIATE_URL | Control plane | Vector database |
| LLM_API_URL | Control plane/runtime | OpenAI-compatible model provider |
| ASKPDF_ADMIN_TOKEN | Control plane/frontend proxy | Product API authentication |
| NEXT_PUBLIC_API_URL | Frontend build | Browser-visible API base |
| ASKPDF_BACKEND_URL | Frontend server | Server-side control-plane URL |
| LANGGRAPH_RUNTIME_URL | Control plane | External LangGraph runtime |
| LANGGRAPH_RUNTIME_TOKEN | Control plane/runtime | LangGraph service authentication |
| AGENT_CHECKPOINT_DATABASE_URL | LangGraph runtime | Runtime checkpoint database |
| HERMES_RUNTIME_URL | Control plane | Hermes runtime when enabled |
| HERMES_RUNTIME_TOKEN | Control plane/Hermes | Hermes service authentication |
| HERMES_API_TOKEN | Hermes/upstream | Pinned Hermes gateway authentication |
| MCP_EXECUTION_CONTEXT_SECRET | Control plane/runtimes | Signed tool execution context |

## Model configuration

- LOCAL_EMBEDDING_MODEL selects the local embedding model.
- LOCAL_RERANKER_MODEL selects the optional local cross-encoder.
- EMBEDDING_DEVICE and RERANKER_DEVICE select CPU, CUDA, or MPS where supported.
- LLM_API_URL points to Docker Model Runner, Ollama, LM Studio, or another
  compatible provider.

The chat model must support the tool-calling behavior required by the selected
workflow. Model readiness is checked by rag_service before dependent operations.

## Product behavior and limits

Configuration also covers token budgets, replanning limits, custom/system
instruction lengths, web-search behavior, memory limits, task budgets, SSE
heartbeats, runtime leases, reconnect attempts, and cleanup retention.

Do not duplicate the complete default table here. Use .env.example for exact
defaults and comments.

## Document processing

Docling, OCR, table extraction, formula enrichment, Poppler, Tesseract,
pdfplumber, PyMuPDF, and spaCy are configured through the DOCLING_* and related
processing variables.

Uploads are content-addressed and stored under ASKPDF_CONTENT_ROOT. The
content-store implementation rejects path traversal, absolute paths, and
symlink escapes.

## Authentication and CORS

- ASKPDF_ADMIN_TOKEN protects the product API.
- ASKPDF_TRUST_PROXY_AUTH enables identity supplied by a trusted reverse proxy.
- ASKPDF_CORS_ORIGINS should contain only intended browser origins.
- Runtime tokens and MCP execution secrets must be distinct.

Trusted-proxy mode is not, by itself, a complete multi-tenant authorization
system. Keep the control-plane port private unless all resource APIs are
explicitly principal-scoped.

## Compose deployment

docker-compose.yml wires the services, volumes, health checks, and internal
URLs. The default stack binds the control plane to localhost and keeps runtime
ports on the internal Compose network.

Database migrations run through the product migration service. LangGraph has a
separate runtime migration and checkpoint database. See [Data model](data-model.md)
and [Runtime operations](runtimes/operations.md).
