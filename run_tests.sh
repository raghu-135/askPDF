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
#
# Examples:
#   ./run_tests.sh                          # Run all pytest tests
#   ./run_tests.sh --verbose                # Run with verbose output
#   ./run_tests.sh --file test_parsing_pytest.py  # Run specific file
#   ./run_tests.sh --test test_docling_parsing  # Run specific test
#   ./run_tests.sh --standalone --pdf tests/01030000000000.pdf  # Run standalone script
#   ./run_tests.sh --coverage               # Run with coverage

set -e

# Default values
VERBOSE=""
TEST_FILE=""
TEST_FUNCTION=""
COVERAGE=""
STANDALONE=""
PDF_PATH=""

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
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all pytest tests"
            echo "  $0 --verbose                          # Run with verbose output"
            echo "  $0 --file test_parsing_pytest.py      # Run specific file"
            echo "  $0 --test test_docling_parsing        # Run specific test"
            echo "  $0 --standalone --pdf tests/01030000000000.pdf  # Run standalone"
            echo "  $0 --coverage                         # Run with coverage"
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

# Check if services are running
echo "Checking if Docker services are running..."
if ! $DOCKER_COMPOSE ps rag-service | grep -q "Up"; then
    echo "Docker services are not running. Starting services..."
    $DOCKER_COMPOSE up -d
    echo "Waiting for services to be healthy..."
    sleep 10
fi

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
    $DOCKER_COMPOSE exec rag-service python /app/tests/test_parsing_service.py --pdf "/app/tests/$(basename $PDF_PATH)"
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

echo "Running pytest tests..."
echo "Command: $PYTEST_CMD"
$DOCKER_COMPOSE exec rag-service $PYTEST_CMD
