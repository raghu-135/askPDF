#!/bin/bash
# run_tests.sh - Docker-native test runner wrapper for askPDF.
#
# Usage:
#   ./run_tests.sh                          # Run all pytest tests plus standalone checks
#   ./run_tests.sh --unit                   # Run unit and mock-based tests
#   ./run_tests.sh --db                     # Run PostgreSQL database tests
#   ./run_tests.sh --api                    # Run API endpoint tests
#   ./run_tests.sh --integration            # Run integration tests
#   ./run_tests.sh --schema                 # Run schema validation tests
#   ./run_tests.sh --standalone             # Run standalone proactive collection script
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

"${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" run --rm --build test-runner "$@"
