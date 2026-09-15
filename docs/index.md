# askPDF documentation

This directory contains the canonical human-facing documentation for askPDF.

## Start here

- [Getting started](getting-started.md): prerequisites, model setup, and first run
- [Architecture](architecture.md): services, boundaries, and end-to-end data flow
- [Configuration](configuration.md): environment variables and deployment settings

## Product and API

- [API overview](api.md): endpoint groups and request-flow behavior
- [Research canvases](research-canvases.md): chat-native `canvas_spec_v1` documents, persistence, and agent emit
- [Data model](data-model.md): PostgreSQL, Weaviate, files, runs, tasks, and memory
- [Security and tool approval](security/tool-approval.md): authentication boundaries and human-gated tools

## Agent runtimes

- [Runtime overview](runtimes/overview.md)
- [LangGraph runtime](runtimes/langgraph.md)
- [Hermes runtime](runtimes/hermes.md)
- [Runtime operations](runtimes/operations.md)
- [Capability matrix](runtimes/capability-matrix.md)

## Contracts and engineering

- [Runtime protocol](contracts/runtime-protocol.md)
- [Agent debug trace v1](contracts/debug-trace-v1.md)
- [Testing](testing.md)

## Documentation ownership rules

- Source code and Pydantic models define executable behavior and API contracts.
- .env.example defines variable names and example values.
- SQLModel definitions and Alembic migrations define the database schema.
- Runtime capability declarations define supported operations.
- Markdown files under prompt directories are executable prompt assets and are
  intentionally not duplicated here.

When behavior changes, update the relevant canonical document and link to the
owning code or configuration.
