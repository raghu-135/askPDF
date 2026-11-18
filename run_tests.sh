#!/bin/bash
# run_tests.sh - Docker-native test runner wrapper for askPDF.
#
# Usage:
#   ./run_tests.sh                          # Run frontend tests, all pytest tests, plus standalone checks
#   ./run_tests.sh --unit                   # Run unit and mock-based tests
#   ./run_tests.sh --db                     # Run PostgreSQL database tests
#   ./run_tests.sh --api                    # Run API endpoint tests
#   ./run_tests.sh --integration            # Run integration tests
#   ./run_tests.sh --langgraph-runtime                # Run isolated LangGraph runtime integration checks
#   ./run_tests.sh --external-runtime                 # Alias for isolated external runtime checks
#   ./run_tests.sh --langgraph-runtime-real           # Run LangGraph runtime against a configured real provider
#   ./run_tests.sh --hermes-runtime                 # Run deterministic Hermes runtime proof
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

echo "askPDF run_tests.sh at commit $(git -C "$(dirname "$0")" rev-parse HEAD 2>/dev/null || git rev-parse HEAD)"

if [ "${ASKPDF_COMPOSE_PRODUCT_BUILD_GATE:-}" = "1" ]; then
    echo "ASKPDF_COMPOSE_PRODUCT_BUILD_GATE=1: verifying product Hermes compose builds..."
    bash "$(dirname "$0")/scripts/ci_product_compose_build.sh"
fi

if [ -f .askpdf-cloud-agent.env ]; then
    set -a
    # shellcheck disable=SC1091
    source .askpdf-cloud-agent.env
    set +a
fi

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
COMPOSE_FILES=(-f docker-compose.yml -f docker-compose.integration.yml)
if [ "${ASKPDF_USE_CLOUD_AGENT_OVERRIDE:-}" = "1" ] && [ -f docker-compose.cloud-agent.override.yml ]; then
    COMPOSE_FILES+=(-f docker-compose.cloud-agent.override.yml)
fi
TEST_COMPOSE_PROFILES="${ASKPDF_TEST_COMPOSE_PROFILES:-test}"
COMPOSE_ARGS=(-p "$TEST_PROJECT_NAME" "${COMPOSE_FILES[@]}")
EXTERNAL_RUNTIME_PROJECT_NAME="${ASKPDF_RUNTIME_TEST_PROJECT_NAME:-${ASKPDF_EXTERNAL_RUNTIME_PROJECT_NAME:-askpdf-runtime-integration-test-$$}}"
RUNTIME_COMPOSE_PROFILES="${ASKPDF_RUNTIME_COMPOSE_PROFILES:-integration}"
EXTERNAL_RUNTIME_COMPOSE_ARGS=(-p "$EXTERNAL_RUNTIME_PROJECT_NAME" "${COMPOSE_FILES[@]}")
RT_POSTGRESQL=integration-postgresql
RT_WEAVIATE=integration-weaviate
RT_RAG_SERVICE=integration-rag-service
RT_LANGGRAPH_RUNTIME=integration-langgraph-runtime
RT_TEST_RUNNER=integration-test-runner
RT_HERMES=integration-hermes
RT_HERMES_RUNTIME=integration-hermes-runtime
RT_HERMES_CONFIG_INIT=integration-hermes-config-init
test_compose() {
    COMPOSE_PROFILES="$TEST_COMPOSE_PROFILES" "${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" "$@"
}

runtime_compose() {
    COMPOSE_PROFILES="$RUNTIME_COMPOSE_PROFILES" "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" "$@"
}

args=("$@")
for arg in "${args[@]}"; do
    if [ "$arg" = "--langgraph-runtime" ] || [ "$arg" = "--external-runtime" ]; then
        RUN_LANGGRAPH_RUNTIME=1
    fi
    if [ "$arg" = "--langgraph-runtime-real" ]; then
        RUN_LANGGRAPH_RUNTIME=1
        RUN_LANGGRAPH_RUNTIME_REAL=1
    fi
    if [ "$arg" = "--hermes-runtime" ]; then
        RUN_HERMES_RUNTIME=1
    fi
done

cleanup() {
    if [ "${ASKPDF_KEEP_TEST_CONTAINERS:-}" = "1" ]; then
        echo "Keeping test containers and volumes for project '$TEST_PROJECT_NAME'"
        return
    fi

    test_compose down --volumes --remove-orphans || true
    if [ "${RUN_LANGGRAPH_RUNTIME:-0}" = "1" ] || [ "${RUN_HERMES_RUNTIME:-0}" = "1" ]; then
        runtime_compose down --volumes --remove-orphans || true
    fi
}

trap cleanup EXIT

run_frontend_tests() {
    echo "Running frontend contract checks (tsc + next build) and Node tests..."
    test_compose run --rm frontend-test-runner
}

external_runtime_diagnostics() {
    echo "External runtime failed; collecting bounded service diagnostics..." >&2
    runtime_compose ps || true
    runtime_compose logs --tail=200 \
        "$RT_RAG_SERVICE" "$RT_LANGGRAPH_RUNTIME" \
        "$RT_POSTGRESQL" fake-llm "$RT_WEAVIATE" \
        "$RT_HERMES" "$RT_HERMES_RUNTIME" "$RT_HERMES_CONFIG_INIT" || true
}

external_runtime_test() {
    if ! runtime_compose run --rm --no-deps "$@"; then
        external_runtime_diagnostics
        return 1
    fi
}

runtime_compose_images_ready() {
    docker image inspect askpdf-rag-service >/dev/null 2>&1 \
        && docker image inspect askpdf-langgraph-runtime >/dev/null 2>&1 \
        && docker image inspect askpdf-fake-llm:ci >/dev/null 2>&1
}

hermes_compose_images_ready() {
    runtime_compose_images_ready \
        && docker image inspect askpdf-hermes-gateway:ci >/dev/null 2>&1 \
        && docker image inspect askpdf-hermes-runtime:ci >/dev/null 2>&1
}

ensure_runtime_compose_images() {
    if [ "${ASKPDF_SKIP_RUNTIME_COMPOSE_BUILD:-}" = "1" ]; then
        echo "Skipping runtime compose build (ASKPDF_SKIP_RUNTIME_COMPOSE_BUILD=1)"
        return 0
    fi
    if runtime_compose_images_ready; then
        echo "Runtime integration images already present; skipping compose build"
        return 0
    fi
    runtime_compose build "$RT_RAG_SERVICE" "$RT_LANGGRAPH_RUNTIME" fake-llm
}

ensure_hermes_compose_images() {
    if [ "${ASKPDF_SKIP_RUNTIME_COMPOSE_BUILD:-}" = "1" ]; then
        return 0
    fi
    if hermes_compose_images_ready; then
        echo "Hermes integration images already present; skipping compose build"
        return 0
    fi
    runtime_compose build hermes-upstream "$RT_HERMES" "$RT_HERMES_RUNTIME"
}

if [ "${RUN_LANGGRAPH_RUNTIME:-0}" = "1" ]; then
    trap external_runtime_diagnostics ERR
    if [ "${RUN_LANGGRAPH_RUNTIME_REAL:-0}" = "1" ]; then
        if [ -z "${LLM_API_URL:-}" ]; then
            echo "--langgraph-runtime-real requires LLM_API_URL" >&2
            exit 1
        fi
        export EXTERNAL_RUNTIME_LLM_API_URL="$LLM_API_URL"
    else
        export EXTERNAL_RUNTIME_LLM_API_URL="http://fake-llm:9000/v1"
    fi
    echo "Starting isolated LangGraph runtime Compose environment '$EXTERNAL_RUNTIME_PROJECT_NAME'..."
    ensure_runtime_compose_images
    runtime_compose up -d "$RT_POSTGRESQL" "$RT_WEAVIATE" fake-llm "$RT_LANGGRAPH_RUNTIME" "$RT_RAG_SERVICE"
    control_plane_ready=0
    for attempt in $(seq 1 120); do
        if runtime_compose exec -T "$RT_RAG_SERVICE" python -c \
            'import json, urllib.request; health=json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)); assert health["status"] == "ok"' 2>/dev/null; then
            control_plane_ready=1
            break
        fi
        sleep 1
    done
    if [ "$control_plane_ready" != "1" ]; then
        echo "Control plane readiness timed out after 120 seconds" >&2
        external_runtime_diagnostics
        exit 1
    fi
    runtime_started=0
    for attempt in $(seq 1 120); do
        if runtime_compose exec -T "$RT_LANGGRAPH_RUNTIME" python -c \
            'import json, urllib.request; startup=json.load(urllib.request.urlopen("http://127.0.0.1:8100/startupz", timeout=5)); assert startup["status"] == "ok"' 2>/dev/null; then
            runtime_started=1
            break
        fi
        sleep 1
    done
    if [ "$runtime_started" != "1" ]; then
        echo "LangGraph runtime startup readiness timed out after 120 seconds" >&2
        external_runtime_diagnostics
        exit 1
    fi
    echo "Verifying LangGraph runtime prompt assets..."
    runtime_compose exec -T "$RT_LANGGRAPH_RUNTIME" python -c \
        'from langgraph_runtime.prompts.loaders import RUNTIME_PROMPT_FILES, runtime_prompt_path, validate_runtime_prompt_assets; validate_runtime_prompt_assets(); assert all(runtime_prompt_path(name).is_file() for name in RUNTIME_PROMPT_FILES)'
    runtime_ready=0
    for attempt in $(seq 1 120); do
        if runtime_compose exec -T "$RT_LANGGRAPH_RUNTIME" python -c \
            'import json, urllib.request; ready=json.load(urllib.request.urlopen("http://127.0.0.1:8100/readyz", timeout=5)); assert ready["status"] == "ok"' 2>/dev/null; then
            runtime_ready=1
            break
        fi
        sleep 1
    done
    if [ "$runtime_ready" != "1" ]; then
        echo "LangGraph runtime work readiness timed out after 120 seconds" >&2
        external_runtime_diagnostics
        exit 1
    fi
    echo "Verifying the immutable production control-plane image..."
    runtime_compose exec -T "$RT_RAG_SERVICE" python -c \
        'import importlib.util; from runtime_protocol.contracts import AgentDefinition; from app.runtime.registry import RuntimeRegistry; assert importlib.util.find_spec("langgraph") is None; assert importlib.util.find_spec("langgraph_runtime") is None; registry=RuntimeRegistry(); registry.initialize(); definition=AgentDefinition(definition_id="router_rag_agent", framework="langgraph", builder_id="langgraph_graph"); adapter=registry.get(definition); assert adapter.__class__.__name__ == "HttpLangGraphRuntimeAdapter" and adapter.framework == "langgraph"'
    echo "Verifying development bind-mounted control-plane isolation..."
    runtime_compose run --rm --no-deps \
        -v "$PWD/rag_service:/app:ro" "$RT_RAG_SERVICE" python -c \
        'import importlib.util; import app, runtime_protocol; assert importlib.util.find_spec("langgraph") is None; assert importlib.util.find_spec("langgraph_runtime") is None'
    runtime_compose exec -T "$RT_LANGGRAPH_RUNTIME" python -c \
        'import importlib.util; import langgraph_runtime, runtime_protocol; assert importlib.util.find_spec("app") is None'
    external_runtime_test runtime-test-runner
    external_runtime_test "$RT_TEST_RUNNER" --file test_runtime_http_adapter_pytest.py
    if [ "${RUN_LANGGRAPH_RUNTIME_REAL:-0}" = "1" ]; then
        if [ -z "${EXTERNAL_RUNTIME_LLM_MODEL:-}" ]; then
            echo "--langgraph-runtime-real requires EXTERNAL_RUNTIME_LLM_MODEL" >&2
            exit 1
        fi
        external_runtime_test -e EXTERNAL_RUNTIME_SMOKE=true -e EXTERNAL_RUNTIME_LLM_MODEL="$EXTERNAL_RUNTIME_LLM_MODEL" "$RT_TEST_RUNNER" --file test_external_runtime_smoke_pytest.py
    else
        external_runtime_test -e EXTERNAL_RUNTIME_SMOKE=true -e EXTERNAL_RUNTIME_LLM_MODEL=external_runtime-deterministic "$RT_TEST_RUNNER" --file test_external_runtime_smoke_pytest.py
    fi
    external_runtime_test -e RUN_RUNTIME_DB_MIGRATIONS=true -e RUNTIME_TEST_TARGET=/app/langgraph_runtime/tests/test_clarification_terminal_pytest.py runtime-test-runner
    external_runtime_test -e RUN_RUNTIME_DB_MIGRATIONS=true -e RUNTIME_TEST_TARGET=/app/langgraph_runtime/tests/test_router_scope_pytest.py runtime-test-runner
    external_runtime_test -e RUN_RUNTIME_DB_MIGRATIONS=true -e RUNTIME_TEST_TARGET=/app/langgraph_runtime/tests/test_router_clarify_scope_pytest.py runtime-test-runner
    external_runtime_test -e RUN_RUNTIME_DB_MIGRATIONS=true -e RUNTIME_TEST_TARGET=/app/langgraph_runtime/tests/test_runtime_service_execution_pytest.py runtime-test-runner
    external_runtime_test -e RUN_RUNTIME_DB_MIGRATIONS=true -e RUNTIME_TEST_TARGET=/app/langgraph_runtime/tests/test_runtime_service_lifecycle_pytest.py runtime-test-runner
    external_runtime_test -e RUN_RUNTIME_DB_MIGRATIONS=true -e ASKPDF_AGENT_CHECKPOINTER=postgres -e ASKPDF_AGENT_CHECKPOINTER_SETUP=true -e RUNTIME_TEST_TARGET=/app/langgraph_runtime/tests/test_runtime_checkpoint_pytest.py runtime-test-runner
    external_runtime_test "$RT_TEST_RUNNER" --file test_agent_runtime_reconciliation_pytest.py
    # The source-mounted test runner can see sibling runtime packages under
    # /workspace. Production image isolation is asserted above against the
    # immutable rag-service image; keep this suite focused on source imports
    # and legacy paths in the mounted harness.
    external_runtime_test -e ASKPDF_ENFORCE_DEPENDENCY_ISOLATION=0 "$RT_TEST_RUNNER" --file test_control_plane_import_boundary_pytest.py
    runtime_compose exec -T "$RT_LANGGRAPH_RUNTIME" python -c \
        'import json, urllib.error, urllib.request
try: urllib.request.urlopen("http://127.0.0.1:8100/v1/dependencies", timeout=3); raise AssertionError("protected runtime endpoint admitted an anonymous request")
except urllib.error.HTTPError as exc: body=json.load(exc); assert exc.code == 401 and body["error"]["code"] == "runtime_unauthorized"'
    echo "Verifying dependency outage isolation and admission recovery..."
    runtime_compose stop "$RT_RAG_SERVICE" fake-llm
    dependencies_degraded=0
    for attempt in $(seq 1 45); do
        if runtime_compose exec -T "$RT_LANGGRAPH_RUNTIME" python -c \
            'import json, os, urllib.error, urllib.request; health=json.load(urllib.request.urlopen("http://127.0.0.1:8100/healthz", timeout=3)); ready_status=200
try: json.load(urllib.request.urlopen("http://127.0.0.1:8100/readyz", timeout=3))
except urllib.error.HTTPError as exc: ready_status=exc.code
protected=urllib.request.Request("http://127.0.0.1:8100/v1/dependencies", headers={"Authorization": "Bearer " + os.environ["LANGGRAPH_RUNTIME_TOKEN"]}); dependencies=json.load(urllib.request.urlopen(protected, timeout=3))["result"]["dependencies"]; assert health["status"] == "ok" and ready_status == 503; assert dependencies["mcp"]["state"] in {"degraded", "unavailable"} and dependencies["provider"]["state"] in {"degraded", "unavailable"}'; then
            dependencies_degraded=1
            break
        fi
        sleep 1
    done
    if [ "$dependencies_degraded" != "1" ]; then
        echo "Runtime dependencies did not become unavailable and readiness did not fail closed" >&2
        exit 1
    fi
    runtime_compose exec -T "$RT_LANGGRAPH_RUNTIME" python -c \
        'import json, os, urllib.error, urllib.request; payload={"operation_id":"external-runtime-dependency-outage:start","request":{"run_id":"external_runtime-dependency-outage","thread_id":"external_runtime-thread","definition_id":"router_rag_agent","framework":"langgraph","builder_id":"langgraph_graph","input":{"question":"test"},"options":{"llm_model":"external_runtime-deterministic","embedding_model":"external_runtime-deterministic-embedding"}},"context":{"embedding_model":"external_runtime-deterministic-embedding","resolved_spec":{"config":{"allowed_tool_ids":["document_evidence"]}}}}; request=urllib.request.Request("http://127.0.0.1:8100/v1/runs/start", data=json.dumps(payload).encode(), headers={"content-type":"application/json", "Authorization": "Bearer " + os.environ["LANGGRAPH_RUNTIME_TOKEN"]}, method="POST");
try: urllib.request.urlopen(request, timeout=3); raise AssertionError("dependent run was admitted")
except urllib.error.HTTPError as exc: body=json.load(exc); assert exc.code == 503 and body["error"]["code"] == "runtime_dependency_unavailable" and body["error"]["retryable"] is True'
    runtime_compose start fake-llm "$RT_RAG_SERVICE"
    dependencies_available=0
    for attempt in $(seq 1 45); do
        if runtime_compose exec -T "$RT_LANGGRAPH_RUNTIME" python -c \
            'import json, os, urllib.request; request=urllib.request.Request("http://127.0.0.1:8100/v1/dependencies", headers={"Authorization": "Bearer " + os.environ["LANGGRAPH_RUNTIME_TOKEN"]}); dependencies=json.load(urllib.request.urlopen(request, timeout=3))["result"]["dependencies"]; assert dependencies["mcp"]["state"] == "available" and dependencies["provider"]["state"] == "available"'; then
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
    runtime_compose stop "$RT_LANGGRAPH_RUNTIME"
    runtime_compose start "$RT_LANGGRAPH_RUNTIME"
    runtime_ready=0
    for attempt in $(seq 1 45); do
        if runtime_compose exec -T "$RT_LANGGRAPH_RUNTIME" python -c \
            'import json, os, urllib.request; health=json.load(urllib.request.urlopen("http://127.0.0.1:8100/healthz", timeout=3)); startup=json.load(urllib.request.urlopen("http://127.0.0.1:8100/startupz", timeout=3)); ready=json.load(urllib.request.urlopen("http://127.0.0.1:8100/readyz", timeout=3)); request=urllib.request.Request("http://127.0.0.1:8100/v1/dependencies", headers={"Authorization": "Bearer " + os.environ["LANGGRAPH_RUNTIME_TOKEN"]}); dependencies=json.load(urllib.request.urlopen(request, timeout=3))["result"]["dependencies"]; assert health["status"] == "ok" and startup["status"] == "ok" and ready["status"] == "ok"; assert dependencies["mcp"]["state"] == "available" and dependencies["mcp"]["protocol"] == "mcp"; assert dependencies["provider"]["state"] == "available"; print(json.dumps({"health": health, "startup": startup, "ready": ready, "dependencies": dependencies}, sort_keys=True))'; then
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
    external_runtime_test -e RUN_RUNTIME_DB_MIGRATIONS=true -e AGENT_RUNTIME_RECOVERY_LOOP_ENABLED=true -e RUNTIME_TEST_TARGET=/app/langgraph_runtime/tests/test_runtime_service_lifecycle_pytest.py runtime-test-runner --test test_recovery_loop_reclaims_a_lease_after_restart
    trap - ERR
    exit 0
fi

if [ "${RUN_HERMES_RUNTIME:-0}" = "1" ]; then
    HERMES_RUNTIME_RECOVERY_RUN_ID="${HERMES_RUNTIME_RECOVERY_RUN_ID:-hermes-runtime-recovery-$$}"
    export HERMES_RUNTIME_RECOVERY_RUN_ID
    export HERMES_RUNTIME_COMPOSE_PROFILES=hermes
    export HERMES_RUNTIME_INTEGRATION=true
    echo "Starting deterministic Hermes runtime Hermes runtime proof..."
    trap external_runtime_diagnostics ERR
    ensure_hermes_compose_images
    if [ "${ASKPDF_SKIP_RUNTIME_COMPOSE_BUILD:-}" != "1" ] \
        && ! docker image inspect askpdf-rag-service >/dev/null 2>&1; then
        runtime_compose build "$RT_RAG_SERVICE"
    fi
    if ! runtime_compose up -d "$RT_POSTGRESQL" "$RT_WEAVIATE" fake-llm "$RT_LANGGRAPH_RUNTIME" "$RT_RAG_SERVICE" "$RT_HERMES" "$RT_HERMES_RUNTIME"; then
        echo "Hermes runtime Compose environment failed to start" >&2
        exit 1
    fi
    control_plane_ready=0
    for attempt in $(seq 1 120); do
        if runtime_compose exec -T "$RT_RAG_SERVICE" python -c \
            'import json, urllib.request; health=json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)); assert health["status"] == "ok"' 2>/dev/null; then
            control_plane_ready=1
            break
        fi
        sleep 1
    done
    if [ "$control_plane_ready" != "1" ]; then
        echo "Control plane readiness timed out after 120 seconds" >&2
        external_runtime_diagnostics
        exit 1
    fi
    hermes_runtime_ready=0
    for attempt in $(seq 1 120); do
        if runtime_compose exec -T "$RT_HERMES_RUNTIME" python -c \
            'import json, urllib.request; ready=json.load(urllib.request.urlopen("http://127.0.0.1:8200/readyz", timeout=5)); assert ready["status"] == "ok"' 2>/dev/null; then
            hermes_runtime_ready=1
            break
        fi
        sleep 1
    done
    if [ "$hermes_runtime_ready" != "1" ]; then
        echo "Hermes runtime work readiness timed out after 120 seconds" >&2
        external_runtime_diagnostics
        exit 1
    fi
    runtime_compose exec -T "$RT_HERMES_RUNTIME" python -c \
        'import importlib.util; import hermes_runtime, runtime_protocol; assert importlib.util.find_spec("app") is None'
    runtime_compose exec -T "$RT_HERMES" /opt/hermes/.venv/bin/python -c \
        'import importlib.util; import runtime_protocol.provider_tool_schema; assert importlib.util.find_spec("sitecustomize") is not None'
    runtime_compose run --rm \
        -e HERMES_GATEWAY_PATCH_PROOF=true \
        -e HERMES_API_URL=http://hermes:8642 \
        -e HERMES_API_TOKEN=7dd5c723b63643e38fc0ccae72a7bf2a8f65c61ad44d4de88e08079b157a55c \
        "$RT_TEST_RUNNER" --file test_hermes_gateway_pinned_patch_active_pytest.py
    runtime_compose run --rm "$RT_TEST_RUNNER" --file test_hermes_runtime_mcp_contract_pytest.py
    runtime_compose run --rm "$RT_TEST_RUNNER" --file test_hermes_builder_provider_pytest.py
    runtime_compose run --rm "$RT_TEST_RUNNER" --file test_hermes_execution_store_pytest.py
    runtime_compose run --rm \
        -e HERMES_RUNTIME_SMOKE=true \
        -e HERMES_MODEL=hermes-runtime-deterministic-hermes \
        -e HERMES_RUNTIME_PRODUCT_DATABASE_URL=postgresql://postgres:postgres@postgresql:5432/askpdf \
        "$RT_TEST_RUNNER" --file test_external_hermes_runtime_smoke_pytest.py
    runtime_compose run --rm \
        -e HERMES_RUNTIME_REAL_SMOKE=true \
        -e HERMES_MODEL=hermes-runtime-deterministic-hermes \
        -e HERMES_RUNTIME_URL=http://hermes-runtime:8200 \
        "$RT_TEST_RUNNER" --file test_real_hermes_container_smoke_pytest.py
    runtime_compose run --rm \
        -e HERMES_RUNTIME_INTEGRATION=true \
        -e ASKPDF_FAIL_IF_ALL_SKIPPED=true \
        -e HERMES_RUNTIME_RECOVERY_RUN_ID="$HERMES_RUNTIME_RECOVERY_RUN_ID" \
        "$RT_TEST_RUNNER" --file test_hermes_runtime_restart_pytest.py --test test_seed_restart_recovery_record
    runtime_compose restart "$RT_HERMES_RUNTIME"
    runtime_compose run --rm \
        -e HERMES_RUNTIME_INTEGRATION=true \
        -e ASKPDF_FAIL_IF_ALL_SKIPPED=true \
        -e HERMES_RUNTIME_RECOVERY_RUN_ID="$HERMES_RUNTIME_RECOVERY_RUN_ID" \
        "$RT_TEST_RUNNER" --file test_hermes_runtime_restart_pytest.py --test test_recovered_run_reconnects_without_another_upstream_start
    trap - ERR
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

TEST_RUN_ARGS=(run --rm)
if [ "${ASKPDF_SKIP_TEST_RUNNER_BUILD:-}" != "1" ]; then
    TEST_RUN_ARGS+=(--build)
fi
test_compose "${TEST_RUN_ARGS[@]}" test-runner "${backend_args[@]}"

if [ "${SKIP_RAG_IMAGE_SMOKE:-}" != "1" ] && [ "${RUN_LANGGRAPH_RUNTIME:-0}" != "1" ] && [ "${RUN_HERMES_RUNTIME:-0}" != "1" ]; then
    echo "Running rag-service image import smoke..."
    bash "${PWD}/rag_service/scripts/rag_service_image_import_smoke.sh"
fi
