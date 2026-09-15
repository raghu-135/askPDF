# Getting started

## Prerequisites

- Docker and Docker Compose
- An OpenAI-compatible chat provider: LM Studio, Ollama, Docker Model Runner,
  or a hosted API such as OpenRouter

## Configure the environment

Copy the example environment file:

    cp .env.example .env

`docker compose up` runs a one-shot `env-secrets` job (same pattern as
`db-migrate`) that replaces the `replace-with-...` service secrets (admin
token, runtime tokens, MCP context secret) and writes them into the stack.
It does not invent `OPENAI_API_KEY`. You do not need Python on the host.

At minimum, set the LLM provider block in `.env`:

- `LLM_API_URL`
- `OPENAI_API_KEY` when the server requires a Bearer token; leave empty for local servers

The complete variable inventory and service ownership are documented in
[Configuration](configuration.md).
See the README for copy-paste local vs OpenRouter examples.

### Docker Model Runner

    docker model pull ai/qwen3:latest
    docker model pull ai/nomic-embed-text-v1.5:latest

    LLM_API_URL=http://host.docker.internal:12434
    OPENAI_API_KEY=

### Ollama

    ollama pull llama3.2
    ollama pull nomic-embed-text

    LLM_API_URL=http://host.docker.internal:11434
    OPENAI_API_KEY=

### LM Studio

Start the local server, download a chat model and embedding model, then use:

    LLM_API_URL=http://host.docker.internal:1234/v1
    OPENAI_API_KEY=

The selected chat model must support tool calling. The control plane also
requires a compatible embedding model and may use a local reranker. Remote
embedding models need a registered Hugging Face tokenizer; the rag-service
image prefetches those tokenizers at build time.

### OpenRouter (or other hosted OpenAI-compatible APIs)

    LLM_API_URL=https://openrouter.ai/api/v1
    OPENAI_API_KEY=sk-or-replace-with-your-key

The control plane lists OpenRouter embedding models that have a registered
tokenizer (Qwen3, GTE, E5, BGE, MiniLM, LFM, and Nomic aliases). Models without
a Hugging Face tokenizer, such as OpenAI text-embedding-3-*, stay hidden until
you add them via EMBEDDING_TOKENIZER_CONFIG_JSON and cache that tokenizer.

## Start askPDF

    docker compose up --build

Open http://localhost:3000.

The default Compose stack includes the frontend, control plane, PostgreSQL,
Weaviate, browser capture, and LangGraph runtime. `.env.example` sets
`COMPOSE_PROFILES=hermes`, so Hermes starts unless you clear that variable.

## Stop and restart

    docker compose down
    docker compose up --build

Persistent database and content volumes are retained unless explicitly removed.

## Development frontend

For frontend hot reload, use the development Compose override:

    docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

## Operational checks

- Frontend: http://localhost:3000
- Control plane: http://127.0.0.1:8000/health (bound to localhost)
- Control-plane readiness, including the LangGraph probe: http://127.0.0.1:8000/ready

LangGraph `/startupz` and `/readyz` and Hermes `/readyz` are container-local.
The default `docker-compose.yml` does not publish `:8100` or `:8200` on the
host. Browser capture is on http://127.0.0.1:8090. Weaviate (`:8080`) and
PostgreSQL (`:5432`) are published on all interfaces in the example Compose
file; keep them off shared networks.

See [Security](security/tool-approval.md) and
[Runtime operations](runtimes/operations.md).
