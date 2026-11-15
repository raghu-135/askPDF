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

## RAG benchmark (on demand)

Shared arXiv fixtures are manifest-only under `benchmarks/fixtures/` (PDFs are
downloaded, not committed). Before upload or benchmark:

```bash
./scripts/download_benchmark_pdfs.sh
./scripts/upload_benchmark_fixtures.sh
```

Context window and packing ratios are pinned in `benchmarks/benchmark_config.json`
(same baseline on every question; optional override via `ASKPDF_BENCHMARK_CONFIG` or
`ASKPDF_BENCHMARK_CONTEXT_OVERRIDES_JSON`). Thread defaults (`replans`, reranker, etc.)
use the same file via `PUT /threads/{id}/settings`.

Run the benchmark suite (not part of default CI or `./run_tests.sh`):

```bash
# stack at http://127.0.0.1:3000 — uses openai/gpt-4.1-mini + gemini-2.5-flash judge
./scripts/run_rag_benchmark.sh --workflows router_rag_agent auto_rag_agent
./scripts/run_rag_benchmark.sh --all-workflows --judge
./scripts/run_rag_benchmark.sh --limit 5                  # smoke
ASKPDF_INCLUDE_HOLDOUT=1 ./scripts/run_rag_benchmark.sh --holdout --judge
```

Machine-readable JSON and a short Markdown summary are written under
`benchmarks/reports/<commit-sha>.{json,md}` (gitignored). The committed
reference scores are in `benchmarks/baseline.md` (aggregate in
`benchmarks/baseline.json`); `run_benchmark.py` prints retrieval/answer deltas
against that file when present. See `benchmarks/rubric.md` and
`benchmarks/HOLDOUT.md` (DPO holdout — do not tune against).

## Coverage that is still thin

The application test suite is strong around backend contracts and runtime
recovery. Browser-level coverage for upload → indexing → chat →
citation/highlight → TTS, multi-user authorization isolation, and
production-scale PDF/model performance remain thinner than the contract tests.
