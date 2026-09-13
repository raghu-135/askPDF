# Testing

The project uses a Docker-native test strategy so local and CI environments use
the same service topology.

## Main entry point

    ./run_tests.sh

The runner starts an isolated Compose project with its own PostgreSQL,
Weaviate, network, and volumes. The normal application stack can remain
running because test services do not publish the same host ports.

## Test groups

- unit and mock-based tests
- PostgreSQL/database and repository tests
- API tests
- MCP and runtime-contract tests
- model-aware integration tests
- schema and migration guardrails
- LangGraph runtime integration and recovery
- Hermes runtime proof and restart recovery
- frontend Node tests

The backend runner maintains explicit ownership in
rag_service/tests/test_inventory.json and validates that repository tests are
assigned or explicitly excluded.

## Useful commands

    ./run_tests.sh --unit
    ./run_tests.sh --db
    ./run_tests.sh --api
    ./run_tests.sh --schema
    ./run_tests.sh --langgraph-runtime
    ./run_tests.sh --hermes-runtime
    ./run_tests.sh --all

Use the options documented in run_tests.sh for individual files, tests,
coverage, verbosity, and retained test containers.

## CI

GitHub Actions is configured in .github/workflows/ci.yml. It builds the service
images, verifies import boundaries, runs the default test suite, and exercises
runtime-specific lanes.

## Documentation-related gaps

The application test suite is strong around backend contracts and runtime
recovery. Continue improving browser-level coverage for upload → indexing →
chat → citation/highlight → TTS, multi-user authorization isolation, and
production-scale PDF/model performance.
