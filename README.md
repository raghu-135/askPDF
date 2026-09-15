# askPDF

askPDF is a private, local PDF research workspace. Upload PDFs or capture
webpages, read documents aloud, search your sources semantically, and chat with
configurable AI workflows.

## Features

- PDF and webpage ingestion with sentence-level extraction
- PDF viewing, annotations, highlighting, and local text-to-speech
- Thread and project organization
- Document, conversation, web, and memory retrieval
- Research canvases: structured comparison, evidence, and timeline documents in the thread workspace
- LangGraph and Hermes-backed agent workflows
- Deep-research tasks with plans, artifacts, approvals, and recovery
- Local or OpenAI-compatible model providers

## Quick start

Prerequisites:

- Docker and Docker Compose
- An OpenAI-compatible chat provider: LM Studio, Ollama, Docker Model Runner,
  or a hosted API such as OpenRouter

Set up the environment and start the stack:

    cp .env.example .env
    python3 scripts/fill_env_secrets.py
    # Edit .env: LLM_API_URL, and OPENAI_API_KEY when the server requires it.
    docker compose up --build

Open http://localhost:3000.

`.env.example` sets `COMPOSE_PROFILES=hermes`, so Hermes starts with the
default stack. Clear that variable to run without Hermes.

### LLM provider authentication

Chat and remote embeddings use one OpenAI-compatible base URL (`LLM_API_URL`)
and one optional API key (`OPENAI_API_KEY`). The control plane, LangGraph, and
Hermes all use those two values. A nonempty key sends `Authorization: Bearer`
on catalog, chat, and embedding calls. Leave the key empty for local servers.

**Local** (LM Studio, Ollama, Docker Model Runner):

    LLM_API_URL=http://host.docker.internal:1234/v1
    OPENAI_API_KEY=

**Hosted** (OpenRouter, OpenAI, and similar):

    LLM_API_URL=https://openrouter.ai/api/v1
    OPENAI_API_KEY=sk-or-replace-with-your-key

When switching from a hosted API to a local server, clear `OPENAI_API_KEY`.
A leftover key is sent as a Bearer token. A missing key on a hosted API shows
up in the UI as “Selected LLM model is unavailable.”

Replace the other placeholder secrets in `.env` as well. Full setup, model
requirements, and service-specific configuration are in
[Getting started](docs/getting-started.md).

## Architecture

The application is composed of a Next.js frontend, a FastAPI control plane,
PostgreSQL, Weaviate, a browser-capture service, and external agent runtimes.
The control plane owns product state and policy; runtimes own framework-specific
execution and checkpoints.

See [Architecture](docs/architecture.md) and [Runtime overview](docs/runtimes/overview.md).

## Documentation

- [Documentation index](docs/index.md)
- [Getting started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [API overview](docs/api.md)
- [Research canvases](docs/research-canvases.md)
- [Data model](docs/data-model.md)
- [Runtime documentation](docs/runtimes/overview.md)
- [Security and tool approval](docs/security/tool-approval.md)
- [Contracts](docs/contracts/runtime-protocol.md)
- [Testing](docs/testing.md)

## Development

Run the Docker-native test runner with:

    ./run_tests.sh

See [Testing](docs/testing.md) for test groups and CI behavior.

## License and acknowledgments

This project uses third-party technologies including Kokoro, spaCy, LangChain,
LangGraph, Weaviate, FastAPI, and Next.js. See their respective licenses and
the dependency manifests for details.
