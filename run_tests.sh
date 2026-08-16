#!/bin/bash
# run_tests.sh - Docker-native test runner wrapper for askPDF.
#
# Usage:
#   ./run_tests.sh                          # Run frontend tests, all pytest tests, plus standalone checks
#   ./run_tests.sh --unit                   # Run unit and mock-based tests
#   ./run_tests.sh --db                     # Run PostgreSQL database tests
#   ./run_tests.sh --api                    # Run API endpoint tests
#   ./run_tests.sh --integration            # Run integration tests
#   ./run_tests.sh --agent-checkpoint       # Run Postgres checkpoint/resume hardening test
#   ./run_tests.sh --phase5                 # Run isolated external-runtime hardening checks
#   ./run_tests.sh --schema                 # Run schema validation tests
#   ./run_tests.sh --standalone             # Run standalone proactive collection script
#   ./run_tests.sh --frontend               # Run frontend tests only
#   ./run_tests.sh --api --strict-warnings   # Fail on coroutine and Pydantic warnings
#   ./run_tests.sh --file test_api_integration_pytest.py --test TestAPIIntegration::test_create_thread_endpoint
#
# Environment:
#   ASKPDF_TEST_PROJECT_NAME=askpdf-test    # Override isolated Compose project name
#   ASKPDF_KEEP_TEST_CONTAINERS=1           # Keep test containers/volumes for debugging

set -e

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is not installed or not on PATH"
    exit 1
fi

if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE=(docker-compose)
else
    echo "Error: docker compose or docker-compose is not installed"
    exit 1
fi

TEST_PROJECT_NAME="${ASKPDF_TEST_PROJECT_NAME:-askpdf-test}"
COMPOSE_ARGS=(-p "$TEST_PROJECT_NAME" -f docker-compose.test.yml)
PHASE5_PROJECT_NAME="${ASKPDF_PHASE5_PROJECT_NAME:-askpdf-phase5-test}"
PHASE5_COMPOSE_ARGS=(-p "$PHASE5_PROJECT_NAME" -f docker-compose.phase5-test.yml)

args=("$@")
for arg in "${args[@]}"; do
    if [ "$arg" = "--phase5" ]; then
        RUN_PHASE5=1
    fi
done

cleanup() {
    if [ "${ASKPDF_KEEP_TEST_CONTAINERS:-}" = "1" ]; then
        echo "Keeping test containers and volumes for project '$TEST_PROJECT_NAME'"
        return
    fi

    "${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" down --volumes --remove-orphans || true
    if [ "${RUN_PHASE5:-0}" = "1" ]; then
        "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" down --volumes --remove-orphans || true
    fi
}

trap cleanup EXIT

run_frontend_tests() {
    echo "Running frontend tests..."
    "${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" run --rm frontend-test-runner
}

if [ "${RUN_PHASE5:-0}" = "1" ]; then
    echo "Starting isolated Phase 5 Compose environment '$PHASE5_PROJECT_NAME'..."
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" up -d postgresql runtime-checkpoint-db-init weaviate rag-service langgraph-runtime
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm test-runner --file test_runtime_contracts_pytest.py
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm test-runner --file test_runtime_http_adapter_pytest.py
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm -e PHASE5_EXTERNAL_SMOKE=true test-runner --file test_external_runtime_smoke_pytest.py
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm test-runner --file test_runtime_service_lifecycle_pytest.py
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm test-runner --file test_agent_runtime_reconciliation_pytest.py
    # The broad legacy suites intentionally exercise the injectable in-process
    # adapter; the external boundary is covered by the focused protocol and
    # lifecycle suites above plus the restart smoke below.
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm -e AGENT_RUNTIME_EXTERNAL_ENABLED=false test-runner --file test_agent_workflows_pytest.py
    echo "Restarting langgraph-runtime to verify readiness and checkpoint service continuity..."
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" stop langgraph-runtime
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" start langgraph-runtime
    runtime_ready=0
    for attempt in $(seq 1 45); do
        if "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
            'import json, urllib.request; health=json.load(urllib.request.urlopen("http://127.0.0.1:8100/healthz", timeout=3)); ready=json.load(urllib.request.urlopen("http://127.0.0.1:8100/readyz", timeout=3)); assert health["status"] == "ok" and ready["status"] == "ok"; print(json.dumps({"health": health, "ready": ready}, sort_keys=True))'; then
            runtime_ready=1
            break
        fi
        sleep 2
    done
    if [ "$runtime_ready" -ne 1 ]; then
        echo "Runtime readiness did not recover after restart" >&2
        exit 1
    fi
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm -e AGENT_RUNTIME_EXTERNAL_ENABLED=false test-runner --unit
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm -e AGENT_RUNTIME_EXTERNAL_ENABLED=false test-runner --db
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm -e AGENT_RUNTIME_EXTERNAL_ENABLED=false test-runner --api
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm -e AGENT_RUNTIME_EXTERNAL_ENABLED=false test-runner --integration
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm -e AGENT_RUNTIME_EXTERNAL_ENABLED=false test-runner --agent-checkpoint
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm -e AGENT_RUNTIME_EXTERNAL_ENABLED=false test-runner --schema
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm -e AGENT_RUNTIME_EXTERNAL_ENABLED=false test-runner --mcp
    exit 0
fi

backend_args=()
run_frontend=0
frontend_only=0

if [ "$#" -eq 0 ]; then
    run_frontend=1
else
    for arg in "${args[@]}"; do
        case "$arg" in
            --frontend)
                run_frontend=1
                frontend_only=1
                ;;
            --all|--all-tests)
                run_frontend=1
                backend_args+=("$arg")
                ;;
            *)
                backend_args+=("$arg")
                ;;
        esac
    done
fi

if [ "$run_frontend" = "1" ]; then
    run_frontend_tests
fi

if [ "$frontend_only" = "1" ] && [ "${#backend_args[@]}" -eq 0 ]; then
    exit 0
fi

"${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" run --rm --build test-runner "${backend_args[@]}"
