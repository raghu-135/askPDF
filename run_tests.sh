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
#   ./run_tests.sh --runtime-contract       # Run the frozen runtime v1 compatibility gate
#   ./run_tests.sh --phase5                 # Run isolated external-runtime integration checks
#   ./run_tests.sh --phase5-real            # Run Phase 5 against a configured real provider
#   ./run_tests.sh --phase7                 # Run deterministic Hermes runtime proof
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
PHASE5_PROJECT_NAME="${ASKPDF_RUNTIME_TEST_PROJECT_NAME:-${ASKPDF_PHASE5_PROJECT_NAME:-askpdf-runtime-integration-test-$$}}"
PHASE5_COMPOSE_ARGS=(-p "$PHASE5_PROJECT_NAME" -f docker-compose.runtime-integration.yml)

args=("$@")
for arg in "${args[@]}"; do
    if [ "$arg" = "--phase5" ]; then
        RUN_PHASE5=1
    fi
    if [ "$arg" = "--phase5-real" ]; then
        RUN_PHASE5=1
        RUN_PHASE5_REAL=1
    fi
    if [ "$arg" = "--phase7" ]; then
        RUN_PHASE7=1
    fi
done

cleanup() {
    if [ "${ASKPDF_KEEP_TEST_CONTAINERS:-}" = "1" ]; then
        echo "Keeping test containers and volumes for project '$TEST_PROJECT_NAME'"
        return
    fi

    "${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" down --volumes --remove-orphans || true
    if [ "${RUN_PHASE5:-0}" = "1" ] || [ "${RUN_PHASE7:-0}" = "1" ]; then
        "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" down --volumes --remove-orphans || true
    fi
}

trap cleanup EXIT

run_frontend_tests() {
    echo "Running frontend tests..."
    "${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" run --rm frontend-test-runner
}

phase5_diagnostics() {
    echo "Phase 5 failed; collecting bounded service diagnostics..." >&2
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" ps || true
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" logs --tail=200 langgraph-runtime rag-service || true
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" logs --tail=100 fake-llm || true
}

phase5_test() {
    if ! "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm --no-deps "$@"; then
        phase5_diagnostics
        return 1
    fi
}

if [ "${RUN_PHASE5:-0}" = "1" ]; then
    trap phase5_diagnostics ERR
    if [ "${RUN_PHASE5_REAL:-0}" = "1" ]; then
        if [ -z "${LLM_API_URL:-}" ]; then
            echo "--phase5-real requires LLM_API_URL" >&2
            exit 1
        fi
        export PHASE5_RUNTIME_LLM_API_URL="$LLM_API_URL"
    else
        export PHASE5_RUNTIME_LLM_API_URL="http://fake-llm:9000/v1"
    fi
    echo "Starting isolated Phase 5 Compose environment '$PHASE5_PROJECT_NAME'..."
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" build rag-service
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" up -d postgresql runtime-checkpoint-db-init weaviate fake-llm rag-service langgraph-runtime
    echo "Verifying the immutable production control-plane image..."
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" exec -T rag-service python -c \
        'import importlib.util, os; from app.runtime.contracts import AgentDefinition; from app.runtime.registry import RuntimeRegistry; assert os.getenv("AGENT_RUNTIME_MODE") is None; assert os.getenv("AGENT_RUNTIME_EXTERNAL_ENABLED") is None; assert importlib.util.find_spec("langgraph") is None; registry=RuntimeRegistry(); registry.initialize(); definition=AgentDefinition(definition_id="router_rag_agent", framework="langgraph", builder_id="langgraph_graph"); adapter=registry.get(definition); assert adapter.__class__.__name__ == "HttpRuntimeAdapter" and adapter.framework == "langgraph"'
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" exec -T rag-service python -c \
        'import json, urllib.request; health=json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)); assert health["status"] == "ok"'
    if "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm --no-deps -e AGENT_RUNTIME_MODE=in_process rag-service python -c \
        'from app.runtime.registry import RuntimeRegistry; RuntimeRegistry().initialize()'; then
        echo "Production control plane accepted AGENT_RUNTIME_MODE=in_process despite missing LangGraph; the mode may have been ignored or the production image contains an unexpected runtime dependency" >&2
        exit 1
    fi
    phase5_test test-runner --file test_runtime_contracts_pytest.py
    phase5_test test-runner --file test_runtime_http_adapter_pytest.py
    if [ "${RUN_PHASE5_REAL:-0}" = "1" ]; then
        if [ -z "${PHASE5_EXTERNAL_LLM_MODEL:-}" ]; then
            echo "--phase5-real requires PHASE5_EXTERNAL_LLM_MODEL" >&2
            exit 1
        fi
        phase5_test -e PHASE5_EXTERNAL_SMOKE=true -e PHASE5_EXTERNAL_LLM_MODEL="$PHASE5_EXTERNAL_LLM_MODEL" test-runner --file test_external_runtime_smoke_pytest.py
    else
        phase5_test -e PHASE5_EXTERNAL_SMOKE=true -e PHASE5_EXTERNAL_LLM_MODEL=phase5-deterministic test-runner --file test_external_runtime_smoke_pytest.py
    fi
    phase5_test test-runner --file test_runtime_service_execution_pytest.py
    phase5_test test-runner --file test_runtime_service_lifecycle_pytest.py
    phase5_test test-runner --file test_agent_runtime_reconciliation_pytest.py
    phase5_test test-runner --file test_control_plane_import_boundary_pytest.py
    echo "Verifying dependency outage isolation and admission recovery..."
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" stop rag-service fake-llm
    dependencies_degraded=0
    for attempt in $(seq 1 30); do
        if "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
            'import json, urllib.request; health=json.load(urllib.request.urlopen("http://127.0.0.1:8100/healthz", timeout=3)); ready=json.load(urllib.request.urlopen("http://127.0.0.1:8100/readyz", timeout=3)); dependencies=json.load(urllib.request.urlopen("http://127.0.0.1:8100/v1/dependencies", timeout=3))["result"]["dependencies"]; assert health["status"] == "ok" and ready["status"] == "ok"; assert dependencies["mcp"]["state"] in {"degraded", "unavailable"} and dependencies["provider"]["state"] in {"degraded", "unavailable"}'; then
            dependencies_degraded=1
            break
        fi
        sleep 1
    done
    if [ "$dependencies_degraded" != "1" ]; then
        echo "Runtime dependencies did not become degraded while readiness remained healthy" >&2
        exit 1
    fi
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
        'import json, urllib.error, urllib.request; payload={"request":{"run_id":"phase5-dependency-outage","thread_id":"phase5-thread","definition_id":"router_rag_agent","framework":"langgraph","builder_id":"langgraph_graph","input":{"question":"test"},"options":{"llm_model":"phase5-deterministic","embedding_model":"phase5-deterministic-embedding"}},"context":{"embedding_model":"phase5-deterministic-embedding","resolved_spec":{"config":{"allowed_tool_ids":["document_evidence"]}}}}; request=urllib.request.Request("http://127.0.0.1:8100/v1/runs/start", data=json.dumps(payload).encode(), headers={"content-type":"application/json"}, method="POST");
try: urllib.request.urlopen(request, timeout=3); raise AssertionError("dependent run was admitted")
except urllib.error.HTTPError as exc: body=json.load(exc); assert exc.code == 503 and body["error"]["code"] == "runtime_dependency_unavailable" and body["error"]["retryable"] is True'
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" start fake-llm rag-service
    dependencies_available=0
    for attempt in $(seq 1 45); do
        if "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
            'import json, urllib.request; dependencies=json.load(urllib.request.urlopen("http://127.0.0.1:8100/v1/dependencies", timeout=3))["result"]["dependencies"]; assert dependencies["mcp"]["state"] == "available" and dependencies["provider"]["state"] == "available"'; then
            dependencies_available=1
            break
        fi
        sleep 1
    done
    if [ "$dependencies_available" != "1" ]; then
        echo "Runtime dependency monitor did not recover after services restarted" >&2
        exit 1
    fi
    echo "Restarting langgraph-runtime to verify readiness and checkpoint service continuity..."
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" stop langgraph-runtime
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" start langgraph-runtime
    runtime_ready=0
    for attempt in $(seq 1 45); do
        if "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
            'import json, urllib.request; health=json.load(urllib.request.urlopen("http://127.0.0.1:8100/healthz", timeout=3)); startup=json.load(urllib.request.urlopen("http://127.0.0.1:8100/startupz", timeout=3)); ready=json.load(urllib.request.urlopen("http://127.0.0.1:8100/readyz", timeout=3)); dependencies=json.load(urllib.request.urlopen("http://127.0.0.1:8100/v1/dependencies", timeout=3))["result"]["dependencies"]; assert health["status"] == "ok" and startup["status"] == "ok" and ready["status"] == "ok"; assert dependencies["mcp"]["state"] == "available" and dependencies["mcp"]["protocol"] == "mcp"; assert dependencies["provider"]["state"] == "available"; print(json.dumps({"health": health, "startup": startup, "ready": ready, "dependencies": dependencies}, sort_keys=True))'; then
            runtime_ready=1
            break
        fi
        sleep 2
    done
    if [ "$runtime_ready" -ne 1 ]; then
        echo "Runtime readiness did not recover after restart" >&2
        exit 1
    fi
    echo "Verifying execution recovery after restart and lease expiry..."
    phase5_test -e AGENT_RUNTIME_RECOVERY_LOOP_ENABLED=true test-runner --file test_runtime_service_lifecycle_pytest.py --test test_recovery_loop_reclaims_a_lease_after_restart
    trap - ERR
    exit 0
fi

if [ "${RUN_PHASE7:-0}" = "1" ]; then
    PHASE7_RECOVERY_RUN_ID="${PHASE7_RECOVERY_RUN_ID:-phase7-recovery-$$}"
    export PHASE7_RECOVERY_RUN_ID
    export PHASE7_HERMES_RUNTIME_ENABLED=true
    export PHASE7_HERMES_INTEGRATION=true
    echo "Starting deterministic Phase 7 Hermes runtime proof..."
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" build rag-service
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" up -d postgresql runtime-checkpoint-db-init weaviate db-migrate fake-llm rag-service hermes-fake hermes-runtime
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm test-runner --file test_hermes_runtime_mcp_contract_pytest.py
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm test-runner --file test_hermes_builder_provider_pytest.py
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm test-runner --file test_hermes_execution_store_pytest.py
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm \
        -e PHASE7_HERMES_INTEGRATION=true \
        -e ASKPDF_FAIL_IF_ALL_SKIPPED=true \
        test-runner --file test_hermes_runtime_integration_pytest.py
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm \
        -e PHASE7_HERMES_SMOKE=true \
        -e HERMES_MODEL=phase7-deterministic-hermes \
        -e PHASE7_PRODUCT_DATABASE_URL=postgresql://postgres:postgres@postgresql:5432/askpdf \
        test-runner --file test_external_hermes_runtime_smoke_pytest.py
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm \
        -e PHASE7_HERMES_INTEGRATION=true \
        -e ASKPDF_FAIL_IF_ALL_SKIPPED=true \
        -e PHASE7_RECOVERY_RUN_ID="$PHASE7_RECOVERY_RUN_ID" \
        test-runner --file test_hermes_runtime_restart_pytest.py --test test_seed_restart_recovery_record
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" restart hermes-runtime
    "${DOCKER_COMPOSE[@]}" "${PHASE5_COMPOSE_ARGS[@]}" run --rm \
        -e PHASE7_HERMES_INTEGRATION=true \
        -e ASKPDF_FAIL_IF_ALL_SKIPPED=true \
        -e PHASE7_RECOVERY_RUN_ID="$PHASE7_RECOVERY_RUN_ID" \
        test-runner --file test_hermes_runtime_restart_pytest.py --test test_recovered_run_reconnects_without_another_upstream_start
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
            --runtime-contract)
                backend_args+=("--runtime-contract")
                ;;
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
