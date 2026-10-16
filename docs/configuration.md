# Configuration

The file .env.example is the authoritative inventory of supported environment
variables that operators set. This page explains the main groups and
deployment boundaries; add a variable to the example file before documenting
it here. Docker Compose also injects service URLs that do not belong in
`.env` (see below).

## Required boundaries

These names are set in `.env.example` unless noted.

| Variable | Owner | Purpose |
| --- | --- | --- |
| LLM_API_URL | Control plane/runtime | OpenAI-compatible model provider |
| OPENAI_API_KEY | Control plane/runtime | Optional Bearer token; empty for local servers |
| ASKPDF_ADMIN_TOKEN | Control plane/frontend proxy | Product API authentication |
| NEXT_PUBLIC_API_URL | Frontend build | Browser-visible API base |
| ASKPDF_BACKEND_URL | Frontend server | Server-side control-plane URL |
| LANGGRAPH_RUNTIME_URL | Control plane | External LangGraph runtime |
| LANGGRAPH_RUNTIME_TOKEN | Control plane/runtime | LangGraph service authentication |
| AGENT_CHECKPOINT_DATABASE_URL | LangGraph runtime | Runtime checkpoint database |
| HERMES_RUNTIME_URL | Control plane | Hermes runtime when enabled |
| HERMES_RUNTIME_TOKEN | Control plane/Hermes | Hermes service authentication |
| HERMES_API_TOKEN | Hermes/upstream | Pinned Hermes gateway authentication |
| HERMES_UPSTREAM_REVISION | Compose Hermes build | Pinned NousResearch/hermes-agent git SHA |
| MCP_EXECUTION_CONTEXT_SECRET | Control plane/runtimes | Signed tool execution context |
| COMPOSE_PROFILES | Compose | `.env.example` sets `hermes`; clear it to omit Hermes |

Compose injects these into the control-plane and migration containers; they
are not operator `.env` entries:

| Variable | Owner | Purpose |
| --- | --- | --- |
| DATABASE_URL | Control plane | Product PostgreSQL |
| TEST_DATABASE_URL | Test runner | Isolated test PostgreSQL |
| WEAVIATE_URL | Control plane | Vector database |
| LOG_LEVEL | Control plane/runtime | Required in-process; Compose sets `INFO` |
| CAPTURE_SERVICE_URL | Control plane | Browser-capture HTTP API |

## Model configuration

- LOCAL_EMBEDDING_MODEL selects the local embedding model baked into the
  rag-service image.
- Popular OpenRouter, LM Studio, and Ollama embedding ids are registered in
  `rag_service/app/services/embedding_tokenizer_registry.py` with matching
  Hugging Face tokenizers. Create Project lists those ids when the provider
  catalog includes them. EMBEDDING_TOKENIZER_CONFIG_JSON overrides or adds
  models.
- The control plane loads remote-model tokenizers with `local_files_only=True`.
  `rag_service/scripts/download_embedding_tokenizers.py` prefetches them during
  image build (`DOWNLOAD_LOCAL_MODELS=true`, the default). HF_TOKEN is optional
  for authenticated Hugging Face downloads at build time.
- LOCAL_RERANKER_MODEL selects the optional local cross-encoder.
- EMBEDDING_DEVICE and RERANKER_DEVICE select CPU, CUDA, or MPS where supported.
- LLM_API_URL points to Docker Model Runner, Ollama, LM Studio, OpenRouter, or
  another OpenAI-compatible provider. The app appends `/v1` when needed.
  Hermes receives the same URL as `OPENAI_BASE_URL` so it does not fall back
  to a local LM Studio default inside the container. Hermes also requires
  `HERMES_MODEL_CONTEXT_LENGTH` of at least 64000.
- OPENAI_API_KEY, when set, is sent as `Authorization: Bearer` on catalog,
  chat, and embedding calls from every service. Leave it empty for local
  servers. Model-list endpoints may be public; chat probes always use this auth.

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

Uploads are content-addressed. The default content root is `/static` inside
the control-plane container (`ASKPDF_CONTENT_ROOT` overrides it; Compose mounts
the `pdf_data` volume there). The content-store implementation rejects path
traversal, absolute paths, and symlink escapes.

Conversion, embedding materialization, and AgentTask execution run as
in-process workers in rag-service.

## MCP transport

`.env.example` sets `MCP_TRANSPORT=in_process` for the control plane. Compose
overrides LangGraph and Hermes to `loopback_http` against
`http://rag-service:8000/internal/mcp/`. Runtimes must not use `in_process`.

## Authentication and CORS

- ASKPDF_ADMIN_TOKEN protects the product API.
- ASKPDF_TRUST_PROXY_AUTH enables identity supplied by a trusted reverse proxy.
- ASKPDF_CORS_ORIGINS should contain only intended browser origins.
- Runtime tokens and MCP execution secrets must be distinct.

Trusted-proxy mode is not, by itself, a complete multi-tenant authorization
system. Keep the control-plane port private unless all resource APIs are
explicitly principal-scoped.

Local service secrets (`ASKPDF_ADMIN_TOKEN`, runtime tokens, MCP context
secret) must be unique random strings of at least 32 characters. The Compose
`env-secrets` job generates them from `replace-with-` placeholders on startup
and loads them into app containers. Leave the placeholders in `.env` if you
want Compose to fill them; already-set values are kept. Do not use documented
placeholder prefixes in production.

## Compose deployment

docker-compose.yml wires the services, volumes, health checks, and internal
URLs. The default stack binds the control plane to `127.0.0.1:8000` and keeps
LangGraph and Hermes ports on the internal Compose network. The example file
does publish Weaviate (`8080`, `50051`) and PostgreSQL (`5432`) on all host
interfaces, and browser capture on `127.0.0.1:8090`.

Database migrations run through the product migration service. LangGraph has a
separate runtime migration and checkpoint database. Service-secret generation
runs through `env-secrets` before those app containers start. See [Data model](data-model.md)
and [Runtime operations](runtimes/operations.md).
