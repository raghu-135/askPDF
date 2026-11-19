# Testing

The project uses a Docker-native test strategy so local and CI environments use
the same service topology.

## Main entry point

    ./run_tests.sh

The runner uses `COMPOSE_PROFILES=test` with `docker-compose.yml` and
`docker-compose.integration.yml` (plus `docker-compose.cloud-agent.override.yml`
only when `.askpdf-cloud-agent.env` sets `ASKPDF_USE_CLOUD_AGENT_OVERRIDE=1`).
It starts an isolated Compose project with its own PostgreSQL,
Weaviate, network, and volumes. The normal application stack can remain
running because test services do not publish the same host ports.

## Test groups

- unit and mock-based tests (`--unit`)
- PostgreSQL/database and repository tests (`--db`)
- API tests (`--api`)
- model-aware integration tests (`--integration`)
- schema and migration guardrails (`--schema`)
- frontend Node tests (`--frontend`), including TypeScript (`tsc --noEmit`), production
  `next build`, canvas-spec and document-tab coverage
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

## Highlight geometry and retrieval

PDF highlight and bbox alignment are display metadata. They must not alter
chunk boundaries, chunk `body_text`, or `contextualized_text` (embedding inputs).
Highlight-only changes should not bump `EXTRACTION_PIPELINE_VERSION` or
`RETRIEVAL_CHUNKING_VERSION`. The guard test is
`rag_service/tests/test_highlight_retrieval_invariant_pytest.py`.

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

## RAG evals (promptfoo + live stack, on demand)

Bundled CC BY PDFs only: `benchmarks/fixtures/bundled/` (`benchmarks/CORPUS.md`,
`fixtures/LICENSES.md`). Legacy arXiv fetch: `./scripts/download_benchmark_pdfs.sh` (POC only).

```bash
python3 scripts/generate_promptfoo_tests.py
bash scripts/verify_promptfoo_tests_generated.sh
bash scripts/cloud-agent-setup.sh stack
./evals/promptfoo/run_eval.sh --tier smoke --judge --label smoke-1
ASKPDF_EVAL_PROFILE=fast ./evals/promptfoo/run_eval.sh --tier core --profile fast --judge
./evals/promptfoo/run_sweep.sh document_ratio_0.36 --judge
```

Tiers: `smoke` (~5 cases), `core`, `advanced`, `holdout`. Profiles: `free` (default, RPM cap)
or `fast` (parallel). Default reasoning effort `none` via `ASKPDF_EVAL_REASONING_EFFORT`.
Stack readiness: `/health` + `/ready` before cases; indexing polled via files API.

Live baseline config: `benchmarks/benchmark_config.eval.json`. Reports:
`benchmarks/reports/<sha>-<label>.{json,md}`. Shared modules:
`scripts/benchmark_eval_support.py`, `scripts/benchmark_scoring.py`,
`scripts/benchmark_execution.py`, `scripts/rag_eval_suite.py`.

## Coverage that is still thin

The application test suite is strong around backend contracts and runtime
recovery. Browser-level coverage for upload → indexing → chat →
citation/highlight → TTS, multi-user authorization isolation, and
production-scale PDF/model performance remain thinner than the contract tests.
