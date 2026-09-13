# Getting started

## Prerequisites

- Docker and Docker Compose
- An OpenAI-compatible local model provider:
  Docker Model Runner, Ollama, or LM Studio

## Configure the environment

Copy the example environment file:

    cp .env.example .env

At minimum, configure LLM_API_URL and replace placeholder authentication
secrets. The complete variable inventory and service ownership are documented
in [Configuration](configuration.md).

### Docker Model Runner

    docker model pull ai/qwen3:latest
    docker model pull ai/nomic-embed-text-v1.5:latest

    LLM_API_URL=http://host.docker.internal:12434

### Ollama

    ollama pull llama3.2
    ollama pull nomic-embed-text

    LLM_API_URL=http://host.docker.internal:11434

### LM Studio

Start the local server, download a chat model and embedding model, then use:

    LLM_API_URL=http://host.docker.internal:1234/v1

The selected chat model must support tool calling. The control plane also
requires a compatible embedding model and may use a local reranker.

## Start askPDF

    docker compose up --build

Open http://localhost:3000.

The default Compose stack includes the frontend, control plane, PostgreSQL,
Weaviate, browser capture, and LangGraph runtime. Hermes services are enabled
when the Hermes Compose profile is enabled.

## Stop and restart

    docker compose down
    docker compose up --build

Persistent database and content volumes are retained unless explicitly removed.

## Development frontend

For frontend hot reload, use the development Compose override:

    docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

## Operational checks

- Control plane: http://localhost:8000/health
- LangGraph startup/readiness: http://localhost:8100/startupz and /readyz
- Hermes readiness: http://localhost:8200/readyz when enabled

Keep runtime and control-plane ports private in shared or production
deployments. See [Security](security/tool-approval.md) and
[Runtime operations](runtimes/operations.md).
