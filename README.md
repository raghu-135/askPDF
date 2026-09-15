# askPDF

askPDF is a private, local PDF research workspace. Upload PDFs or capture
webpages, read documents aloud, search your sources semantically, and chat with
configurable AI workflows.

## Features

- PDF and webpage ingestion with sentence-level extraction
- PDF viewing, annotations, highlighting, and local text-to-speech
- Thread and project organization
- Document, conversation, web, and memory retrieval
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
    # Edit .env: LLM_API_URL plus LLM_AUTH_MODE (and OPENAI_API_KEY when required).
    docker compose up --build

Open http://localhost:3000.

### LLM provider authentication

Chat and remote embeddings use one OpenAI-compatible base URL (`LLM_API_URL`).
The model dropdown can list public catalogs without a key; **using** a model
always goes through `/chat/completions` (and `/embeddings`) with the auth mode
below. LangGraph uses the same variables.

**Local / keyless** (LM Studio, Ollama, Docker Model Runner):

    LLM_API_URL=http://host.docker.internal:1234/v1
    LLM_AUTH_MODE=none
    LLM_KEYLESS_PROVIDER=lmstudio
    OPENAI_API_KEY=

`LLM_KEYLESS_PROVIDER` must be `lmstudio`, `ollama`, or `local`. It is required
when `LLM_AUTH_MODE=none` and ignored when auth is `required`.

**Hosted / API key** (OpenRouter, OpenAI, and similar):

    LLM_API_URL=https://openrouter.ai/api/v1
    LLM_AUTH_MODE=required
    LLM_KEYLESS_PROVIDER=
    OPENAI_API_KEY=sk-or-replace-with-your-key

Do not leave `LLM_KEYLESS_PROVIDER=lmstudio` as your only mental model of
auth: with `LLM_AUTH_MODE=required` that value is unused; the Bearer token is
what OpenRouter checks. A missing or unused key shows up in the UI as
“Selected LLM model is unavailable.”

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
