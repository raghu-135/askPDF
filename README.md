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
- An OpenAI-compatible local model server such as Docker Model Runner,
  Ollama, or LM Studio

Set up the environment and start the stack:

    cp .env.example .env
    # Edit .env and set LLM_API_URL and required secrets.
    docker compose up --build

Open http://localhost:3000.

The full setup, model requirements, and service-specific configuration are in
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
