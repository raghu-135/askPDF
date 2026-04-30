#!/bin/bash
# run_tests.sh - Portable test runner for askPDF project
# Usage: ./run_tests.sh [options]
#
# Options:
#   --help                 Show this help message
#   --verbose              Run tests with verbose output
#   --file <file>          Run specific test file
#   --test <test>          Run specific test function
#   --coverage             Run tests with coverage report
#   --standalone           Run standalone test scripts instead of pytest
#   --pdf <path>           Path to PDF file (for standalone tests)
#   --db-tests             Run PostgreSQL database tests (requires PostgreSQL service)
#   --db-only              Run only database tests, skip other tests
#   --integration          Run DB-agnostic integration tests
#   --api                  Run API endpoint tests
#   --schema               Run schema validation tests
#   --all-tests            Run all DB-agnostic tests (integration + api + schema)
#
# Examples:
#   ./run_tests.sh                          # Run all pytest tests
#   ./run_tests.sh --verbose                # Run with verbose output
#   ./run_tests.sh --file test_parsing_pytest.py  # Run specific file
#   ./run_tests.sh --test test_docling_parsing  # Run specific test
#   ./run_tests.sh --standalone --pdf tests/01030000000000.pdf  # Run standalone script
#   ./run_tests.sh --coverage               # Run with coverage
#   ./run_tests.sh --db-tests               # Run PostgreSQL database tests
#   ./run_tests.sh --db-only                # Run only database tests
#   ./run_tests.sh --integration           # Run DB-agnostic integration tests
#   ./run_tests.sh --api                    # Run API endpoint tests
#   ./run_tests.sh --schema                 # Run schema validation tests
#   ./run_tests.sh --all-tests              # Run all DB-agnostic tests

set -e

# Default values
VERBOSE=""
TEST_FILE=""
TEST_FUNCTION=""
COVERAGE=""
STANDALONE=""
PDF_PATH=""
DB_TESTS=""
DB_ONLY=""
INTEGRATION_TESTS=""
API_TESTS=""
SCHEMA_TESTS=""
ALL_TESTS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --help                 Show this help message"
            echo "  --verbose              Run tests with verbose output"
            echo "  --file <file>          Run specific test file"
            echo "  --test <test>          Run specific test function"
            echo "  --coverage             Run tests with coverage report"
            echo "  --standalone           Run standalone test scripts instead of pytest"
            echo "  --pdf <path>           Path to PDF file (for standalone tests)"
            echo "  --db-tests             Run PostgreSQL database tests (requires PostgreSQL service)"
            echo "  --db-only              Run only database tests, skip other tests"
            echo "  --integration          Run DB-agnostic integration tests"
            echo "  --api                  Run API endpoint tests"
            echo "  --schema               Run schema validation tests"
            echo "  --all-tests            Run all DB-agnostic tests (integration + api + schema)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all pytest tests"
            echo "  $0 --verbose                          # Run with verbose output"
            echo "  $0 --file test_parsing_pytest.py      # Run specific file"
            echo "  $0 --test test_docling_parsing        # Run specific test"
            echo "  $0 --standalone --pdf tests/01030000000000.pdf  # Run standalone"
            echo "  $0 --coverage                         # Run with coverage"
            echo "  $0 --db-tests                         # Run PostgreSQL database tests"
            echo "  $0 --db-only                          # Run only database tests"
            echo "  $0 --integration                      # Run DB-agnostic integration tests"
            echo "  $0 --api                              # Run API endpoint tests"
            echo "  $0 --schema                           # Run schema validation tests"
            echo "  $0 --all-tests                        # Run all DB-agnostic tests"
            exit 0
            ;;
        --verbose)
            VERBOSE="-v"
            shift
            ;;
        --file)
            TEST_FILE="$2"
            shift 2
            ;;
        --test)
            TEST_FUNCTION="$2"
            shift 2
            ;;
        --coverage)
            COVERAGE="--cov=app --cov-report=term-missing"
            shift
            ;;
        --standalone)
            STANDALONE="true"
            shift
            ;;
        --pdf)
            PDF_PATH="$2"
            shift 2
            ;;
        --db-tests)
            DB_TESTS="true"
            shift
            ;;
        --db-only)
            DB_ONLY="true"
            shift
            ;;
        --integration)
            INTEGRATION_TESTS="true"
            shift
            ;;
        --api)
            API_TESTS="true"
            shift
            ;;
        --schema)
            SCHEMA_TESTS="true"
            shift
            ;;
        --all-tests)
            ALL_TESTS="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
    echo "Error: docker-compose or docker is not installed"
    exit 1
fi

# Determine docker-compose command
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

# Generate unique test database identifier
TEST_RUN_ID=$(uuidgen 2>/dev/null || echo "test_$(date +%s)_$$")
echo "Test run ID: $TEST_RUN_ID"

# Set up test database directory (inside the container's /data volume)
TEST_DATA_DIR="/data/test_$TEST_RUN_ID"
echo "Test data directory: $TEST_DATA_DIR (will be created inside container)"

# Build the application to capture any code changes
echo "Building Docker images to capture latest code changes..."
if ! $DOCKER_COMPOSE build; then
    echo "Error: Docker build failed. Please fix build errors before running tests."
    exit 1
fi
echo "Docker build completed successfully."

# Check if services are running
echo "Checking if Docker services are running..."
if ! $DOCKER_COMPOSE ps rag-service | grep -q "Up"; then
    echo "Docker services are not running. Starting services..."
    $DOCKER_COMPOSE up -d
    echo "Waiting for services to be healthy..."
    sleep 10
fi

# Check if PostgreSQL is running and create test database
# PostgreSQL is now the primary database for all tests
if $DOCKER_COMPOSE ps postgresql 2>/dev/null | grep -q "Up"; then
    echo "PostgreSQL service is running."
    
    # Generate unique PostgreSQL database name
    TEST_POSTGRES_DB="test_askpdf_${TEST_RUN_ID}"
    echo "Test PostgreSQL database: $TEST_POSTGRES_DB"
    
    # Create the test database
    echo "Creating test PostgreSQL database..."
    $DOCKER_COMPOSE exec -T postgresql psql -U postgres -c "CREATE DATABASE \"$TEST_POSTGRES_DB\";" || {
        echo "Failed to create test database. Cleaning up..."
        exit 1
    }
    POSTGRES_AVAILABLE="true"
else
    echo "WARNING: PostgreSQL service is not running. Database tests will be skipped."
    POSTGRES_AVAILABLE="false"
fi

# Cleanup function to remove test databases
cleanup() {
    echo "Cleaning up test databases..."
    
    # Remove test data directory inside container
    echo "Removing test data directory from container: $TEST_DATA_DIR"
    $DOCKER_COMPOSE exec -T rag-service rm -rf "$TEST_DATA_DIR" 2>/dev/null || true
    
    # Drop PostgreSQL test database if it was created
    if [ "$POSTGRES_AVAILABLE" = "true" ] && [ -n "$TEST_POSTGRES_DB" ]; then
        echo "Dropping test PostgreSQL database: $TEST_POSTGRES_DB"
        $DOCKER_COMPOSE exec -T postgresql psql -U postgres -c "DROP DATABASE IF EXISTS \"$TEST_POSTGRES_DB\";" 2>/dev/null || true
    fi
    
    echo "Cleanup complete."
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Run standalone tests
if [ "$STANDALONE" = "true" ]; then
    if [ -z "$PDF_PATH" ]; then
        echo "Error: --pdf path is required for standalone tests"
        exit 1
    fi

    if [ ! -f "$PDF_PATH" ]; then
        echo "Error: PDF file not found: $PDF_PATH"
        exit 1
    fi

    echo "Running standalone test script with PDF: $PDF_PATH"
    $DOCKER_COMPOSE exec -e DATA_DIR="$TEST_DATA_DIR" rag-service python /app/tests/test_parsing_service.py --pdf "/app/tests/$(basename $PDF_PATH)"
    exit $?
fi

# Run pytest tests
# Check if pytest is installed
echo "Checking for pytest..."
$DOCKER_COMPOSE exec rag-service python -c "import pytest" 2>/dev/null || {
    echo "Installing pytest..."
    $DOCKER_COMPOSE exec rag-service pip install pytest
}

PYTEST_CMD="pytest /app/tests/ $VERBOSE"

# Handle db-only flag - run only database tests
if [ "$DB_ONLY" = "true" ]; then
    PYTEST_CMD="pytest /app/tests/test_database_connection_pytest.py /app/tests/test_models_sqlmodel_pytest.py /app/tests/test_thread_repository_pytest.py /app/tests/test_file_repository_pytest.py /app/tests/test_message_repository_pytest.py /app/tests/test_thread_file_repository_pytest.py /app/tests/test_stats_repository_pytest.py /app/tests/test_repository_transactions_pytest.py /app/tests/test_jsonb_operations_pytest.py $VERBOSE"
fi

# Handle db-tests flag - include database tests with regular tests
if [ "$DB_TESTS" = "true" ]; then
    PYTEST_CMD="pytest /app/tests/ $VERBOSE"
fi

# Handle integration flag - run DB-agnostic integration tests
if [ "$INTEGRATION_TESTS" = "true" ]; then
    echo "WARNING: Integration tests removed. Use --api for API tests or --db-tests for database tests."
    exit 1
fi

# Handle api flag - run API endpoint tests
if [ "$API_TESTS" = "true" ]; then
    PYTEST_CMD="pytest /app/tests/test_api_endpoints_pytest.py $VERBOSE"
fi

# Handle schema flag - run schema validation tests
if [ "$SCHEMA_TESTS" = "true" ]; then
    PYTEST_CMD="pytest /app/tests/test_schema_validation_pytest.py $VERBOSE"
fi

# Handle all-tests flag - run all API and schema tests
if [ "$ALL_TESTS" = "true" ]; then
    PYTEST_CMD="pytest /app/tests/test_schema_validation_pytest.py /app/tests/test_api_endpoints_pytest.py /app/tests/test_api_integration_pytest.py $VERBOSE"
fi

if [ -n "$TEST_FILE" ]; then
    PYTEST_CMD="pytest /app/tests/$TEST_FILE $VERBOSE"
fi

if [ -n "$TEST_FUNCTION" ]; then
    if [ -n "$TEST_FILE" ]; then
        PYTEST_CMD="pytest /app/tests/$TEST_FILE::$TEST_FUNCTION $VERBOSE"
    else
        echo "Error: --test requires --file to be specified"
        exit 1
    fi
fi

if [ -n "$COVERAGE" ]; then
    # Check if pytest-cov is installed
    echo "Checking for pytest-cov..."
    $DOCKER_COMPOSE exec rag-service python -c "import pytest_cov" 2>/dev/null || {
        echo "Installing pytest-cov for coverage reporting..."
        $DOCKER_COMPOSE exec rag-service pip install pytest-cov
    }
    PYTEST_CMD="$PYTEST_CMD $COVERAGE"
fi

# Build environment variables for test databases
TEST_ENV_VARS="-e DATA_DIR=$TEST_DATA_DIR"
if [ "$POSTGRES_AVAILABLE" = "true" ] && [ -n "$TEST_POSTGRES_DB" ]; then
    TEST_DB_URL="postgresql+asyncpg://postgres:postgres@postgresql:5432/$TEST_POSTGRES_DB"
    # Override both DATABASE_URL and TEST_DATABASE_URL to ensure all code paths use test DB
    TEST_ENV_VARS="$TEST_ENV_VARS -e DATABASE_URL=$TEST_DB_URL -e TEST_DATABASE_URL=$TEST_DB_URL"
fi

echo "Running pytest tests..."
echo "Command: $PYTEST_CMD"
echo "Test environment: DATA_DIR=$TEST_DATA_DIR"
if [ "$POSTGRES_AVAILABLE" = "true" ] && [ -n "$TEST_POSTGRES_DB" ]; then
    echo "Test PostgreSQL database: $TEST_POSTGRES_DB"
fi

$DOCKER_COMPOSE exec $TEST_ENV_VARS rag-service $PYTEST_CMD
