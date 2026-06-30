#!/bin/bash
# run_tests.sh - Docker-native test runner wrapper for askPDF.
#
# Usage:
#   ./run_tests.sh                          # Run frontend tests, all pytest tests, plus standalone checks
#   ./run_tests.sh --unit                   # Run unit and mock-based tests
#   ./run_tests.sh --db                     # Run PostgreSQL database tests
#   ./run_tests.sh --api                    # Run API endpoint tests
#   ./run_tests.sh --integration            # Run integration tests
#   ./run_tests.sh --schema                 # Run schema validation tests
#   ./run_tests.sh --standalone             # Run standalone proactive collection script
#   ./run_tests.sh --frontend               # Run frontend tests only
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

cleanup() {
    if [ "${ASKPDF_KEEP_TEST_CONTAINERS:-}" = "1" ]; then
        echo "Keeping test containers and volumes for project '$TEST_PROJECT_NAME'"
        return
    fi

    "${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" down --volumes --remove-orphans || true
}

trap cleanup EXIT

run_frontend_tests() {
    echo "Running frontend tests..."
    "${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" run --rm frontend-test-runner
}

args=("$@")
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
