# Testing

The project uses a Docker-native test strategy so local and CI environments use
the same service topology.

## Main entry point

    ./run_tests.sh

The runner starts an isolated Compose project with its own PostgreSQL,
Weaviate, network, and volumes. The normal application stack can remain
running because test services do not publish the same host ports.

## Test groups

- unit and mock-based tests (`--unit`)
- PostgreSQL/database and repository tests (`--db`)
- API tests (`--api`)
- model-aware integration tests (`--integration`)
- schema and migration guardrails (`--schema`)
- frontend Node tests (`--frontend`), including canvas-spec and document-tab coverage
- LangGraph runtime integration and recovery (`--langgraph-runtime`;
  `--external-runtime` is an alias)
- LangGraph against a configured real provider (`--langgraph-runtime-real`)
- Hermes runtime proof and restart recovery (`--hermes-runtime`)
- standalone collection (`--standalone`)

With no flags, the runner runs frontend tests, the default control-plane
pytest group (`TEST_GROUP` defaults to `all`), and standalone checks.
`--all` / `--all-tests` selects that same control-plane group. Isolated
LangGraph and Hermes Compose proofs still require `--langgraph-runtime` and
`--hermes-runtime`.

Canvas backend coverage lives in test_canvas_spec_pytest.py,
test_canvas_emit_pytest.py, test_canvas_layout_skills_pytest.py, and
test_canvas_api_pytest.py. LangGraph answer-node emit is
langgraph_runtime/tests/test_canvas_publish_pytest.py.

The backend runner maintains explicit ownership in
rag_service/tests/test_inventory.json and validates that repository tests are
assigned or explicitly excluded.

## Useful commands

    ./run_tests.sh --unit
    ./run_tests.sh --db
    ./run_tests.sh --api
    ./run_tests.sh --schema
    ./run_tests.sh --frontend
    ./run_tests.sh --langgraph-runtime
    ./run_tests.sh --hermes-runtime
    ./run_tests.sh --all

Use the options documented in run_tests.sh for individual files, tests,
coverage, verbosity, and retained test containers
(`ASKPDF_KEEP_TEST_CONTAINERS=1`).

## CI

GitHub Actions is configured in .github/workflows/ci.yml. It builds the service
images, verifies import boundaries, runs the default test suite, collects
control-plane and LangGraph tests, runs the LangGraph runtime proof, and runs
a separate Hermes runtime-proof job.

## Coverage that is still thin

The application test suite is strong around backend contracts and runtime
recovery. Browser-level coverage for upload → indexing → chat →
citation/highlight → TTS, multi-user authorization isolation, and
production-scale PDF/model performance remain thinner than the contract tests.
