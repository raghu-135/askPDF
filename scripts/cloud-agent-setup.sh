#!/usr/bin/env bash
# Cloud Agent VM bootstrap for askPDF: Docker, host LLM proxy, .env, Docling warmup.
#
# Usage (from repository root):
#   bash scripts/cloud-agent-setup.sh install   # once per VM / environment build
#   bash scripts/cloud-agent-setup.sh start     # every agent boot
#   bash scripts/cloud-agent-setup.sh stack     # install + start + compose up (dev / E2E)
#   bash scripts/cloud-agent-setup.sh tests-env # write .env for ./run_tests.sh only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${ASKPDF_CLOUD_SETUP_LOG_DIR:-/tmp/askpdf-cloud-setup}"
HF_CACHE_DIR="${REPO_ROOT}/.askpdf-hf-cache"
HF_CACHE_SEED_MARKER="${HF_CACHE_DIR}/.askpdf-hf-exported"
HF_CACHE_DOCLING_MARKER="${HF_CACHE_DIR}/.askpdf-docling-warmed"
RAG_IMAGE_SEED="${ASKPDF_RAG_IMAGE_SEED:-askpdf-rag-service-cache-seed}"
SAMPLE_PDF="${REPO_ROOT}/rag_service/tests/01030000000000.pdf"
mkdir -p "$LOG_DIR"
PROXY_PID_FILE="${LOG_DIR}/openrouter-proxy.pid"
DOCKERD_PID_FILE="${LOG_DIR}/dockerd.pid"
COMPOSE_FILES=(-f docker-compose.yml -f docker-compose.vm-hostgateway.override.yml)

export DEBIAN_FRONTEND=noninteractive
if docker buildx version >/dev/null 2>&1; then
  export COMPOSE_DOCKER_CLI_BUILD=1
  export DOCKER_BUILDKIT=1
else
  export COMPOSE_DOCKER_CLI_BUILD=0
  export DOCKER_BUILDKIT=0
fi
# Reduce stuck apt/npm docker-build hangs on slow networks.
export APT_FLAGS='-o Acquire::Retries=5 -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold'
export npm_config_fetch_timeout="${npm_config_fetch_timeout:-600000}"
export npm_config_fetch_retries="${npm_config_fetch_retries:-5}"

log() {
  printf '[cloud-agent-setup] %s\n' "$*" | tee -a "${LOG_DIR}/setup.log"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "missing required command: $1"
    exit 1
  fi
}

docker_cli() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
    return
  fi
  sudo docker "$@"
}

compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif sudo docker compose version >/dev/null 2>&1; then
    sudo docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    sudo docker-compose "$@"
  fi
}

install_docker() {
  if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1 || sudo docker info >/dev/null 2>&1; then
      log "docker already usable"
      return 0
    fi
  fi

  log "installing Docker (docker.io + compose plugin)..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update $APT_FLAGS
    sudo apt-get install -y $APT_FLAGS \
      ca-certificates curl git iptables pigz \
      docker.io docker-compose-v2 docker-buildx-plugin python3
    sudo usermod -aG docker "${USER:-ubuntu}" 2>/dev/null || true
    if ! docker buildx version >/dev/null 2>&1; then
      log "installing docker buildx CLI plugin..."
      mkdir -p "$HOME/.docker/cli-plugins"
      ARCH="$(uname -m)"
      case "$ARCH" in
        x86_64) BUILDX_ARCH=amd64 ;;
        aarch64) BUILDX_ARCH=arm64 ;;
        *) BUILDX_ARCH=amd64 ;;
      esac
      curl -fsSL "https://github.com/docker/buildx/releases/download/v0.19.3/buildx-v0.19.3.linux-${BUILDX_ARCH}" \
        -o "$HOME/.docker/cli-plugins/docker-buildx"
      chmod +x "$HOME/.docker/cli-plugins/docker-buildx"
      sudo ln -sf "$HOME/.docker/cli-plugins/docker-buildx" /usr/local/lib/docker/cli-plugins/docker-buildx 2>/dev/null || \
        sudo cp "$HOME/.docker/cli-plugins/docker-buildx" /usr/local/bin/docker-buildx
    fi
  else
    log "unsupported OS: apt-get not found"
    exit 1
  fi
}

ensure_docker_cli_access() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  if groups | grep -qw docker; then
    return 0
  fi
  if [[ -S /var/run/docker.sock ]]; then
    sudo chmod 666 /var/run/docker.sock 2>/dev/null || true
  fi
  if docker info >/dev/null 2>&1; then
    log "docker CLI access via docker.sock permissions"
    return 0
  fi
  log "docker requires sudo; consider re-login after usermod -aG docker"
  return 1
}

ensure_dockerd() {
  if docker info >/dev/null 2>&1 || sudo docker info >/dev/null 2>&1; then
    ensure_docker_cli_access || true
    return 0
  fi

  log "starting dockerd..."
  if ! command -v dockerd >/dev/null 2>&1; then
    install_docker
  fi

  if [[ -f "$DOCKERD_PID_FILE" ]] && kill -0 "$(cat "$DOCKERD_PID_FILE")" 2>/dev/null; then
    for _ in $(seq 1 60); do
      if docker info >/dev/null 2>&1; then
        return 0
      fi
      sleep 1
    done
  fi

  sudo nohup dockerd \
    --iptables=true \
    --ip-forward=true \
    >"${LOG_DIR}/dockerd.log" 2>&1 &
  echo $! | sudo tee "$DOCKERD_PID_FILE" >/dev/null

  for _ in $(seq 1 120); do
    if docker info >/dev/null 2>&1 || sudo docker info >/dev/null 2>&1; then
      log "dockerd is ready"
      ensure_docker_cli_access || true
      return 0
    fi
    sleep 1
  done
  log "dockerd failed to become ready; see ${LOG_DIR}/dockerd.log"
  tail -50 "${LOG_DIR}/dockerd.log" || true
  exit 1
}

write_env_file() {
  local mode="${1:-stack}"
  if [[ ! -f .env.example ]]; then
    log ".env.example missing"
    exit 1
  fi

  if [[ ! -f .env ]] || [[ "${ASKPDF_FORCE_REWRITE_ENV:-}" = "1" ]]; then
    cp .env.example .env
    log "created .env from .env.example"
  fi

  if [[ "$mode" = "tests" ]]; then
    # ./run_tests.sh uses fake-llm from compose.test.yml; mirror CI env.
    if [[ -f .env.ci ]]; then
      cp .env.ci .env
      log "using .env.ci for docker-native tests"
    fi
    if [[ -f frontend/package-lock.json ]]; then
      log "prefetching frontend node_modules on host (cloud npm ci workaround)..."
      (cd frontend && npm ci --ignore-scripts --fetch-timeout=600000 --fetch-retries=5)
    fi
    cat >.askpdf-cloud-agent.env <<'EOF'
ASKPDF_USE_CLOUD_TEST_OVERRIDE=1
EOF
    log "wrote .askpdf-cloud-agent.env for host-gateway test overrides"
    log "pre-building test runner image and warming postgres/weaviate..."
    compose_cmd -p askpdf-test -f docker-compose.test.yml -f docker-compose.cloud-agent.test.override.yml build --pull test-runner
    prefetch_docling_models
    log "verifying Docling inside test-runner with host HF cache..."
    compose_cmd -p askpdf-test -f docker-compose.test.yml -f docker-compose.cloud-agent.test.override.yml \
      run --rm --entrypoint python test-runner \
      -c "from docling.document_converter import DocumentConverter; DocumentConverter(); print('test_runner_docling_ready')"
    compose_cmd -p askpdf-test -f docker-compose.test.yml -f docker-compose.cloud-agent.test.override.yml up -d postgresql weaviate fake-llm
    for _ in $(seq 1 120); do
      if curl -sf http://127.0.0.1:8080/v1/.well-known/ready >/dev/null 2>&1; then
        if command -v pg_isready >/dev/null 2>&1 && pg_isready -h 127.0.0.1 -p 5432 -U postgres >/dev/null 2>&1; then
          log "postgresql and weaviate ready for tests"
          break
        fi
        if ! command -v pg_isready >/dev/null 2>&1; then
          log "weaviate ready for tests (postgresql assumed up)"
          break
        fi
      fi
      sleep 1
    done
    return 0
  fi

  python3 scripts/fill_env_secrets.py --env .env
  wire_llm_api_key_in_env

  # VM dev stack: Hermes profile + host OpenRouter proxy (containers use host.docker.internal:12345).
  if grep -q '^COMPOSE_PROFILES=' .env; then
    sed -i 's/^COMPOSE_PROFILES=.*/COMPOSE_PROFILES=hermes/' .env
  else
    echo 'COMPOSE_PROFILES=hermes' >>.env
  fi

  if grep -q '^LLM_API_URL=' .env; then
    sed -i 's|^LLM_API_URL=.*|LLM_API_URL=http://host.docker.internal:12345/v1|' .env
  else
    echo 'LLM_API_URL=http://host.docker.internal:12345/v1' >>.env
  fi

  if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    log "warning: no OPENAI_API_KEY / OPENROUTER_API_KEY in host environment (LLM proxy will fail)"
  fi
}

wire_llm_api_key_in_env() {
  OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" OPENAI_API_KEY="${OPENAI_API_KEY:-}" python3 - <<'PY'
import os
import re
from pathlib import Path

env_path = Path(".env")
if not env_path.exists():
    raise SystemExit("wire_llm_api_key_in_env: .env missing")

key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
lines = env_path.read_text().splitlines(keepends=True)
found = False
out: list[str] = []
for line in lines:
    if re.match(r"^OPENAI_API_KEY=", line):
        found = True
        if key:
            out.append(f"OPENAI_API_KEY={key}\n")
        else:
            out.append(line)
    else:
        out.append(line if line.endswith("\n") else line + "\n")
if key and not found:
    out.append(f"OPENAI_API_KEY={key}\n")
env_path.write_text("".join(out))
PY
  if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    log "OPENROUTER_API_KEY is set in the VM environment (wired to .env OPENAI_API_KEY)"
  elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
    log "OPENAI_API_KEY is set in the VM environment (OPENROUTER_API_KEY unset)"
  else
    log "warning: no LLM API key in VM environment; .env OPENAI_API_KEY unchanged"
  fi
}

start_openrouter_proxy() {
  local api_key="${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-}}"
  if [[ -z "$api_key" ]]; then
    log "skipping OpenRouter host proxy (no API key in environment)"
    return 0
  fi

  if [[ -f "$PROXY_PID_FILE" ]]; then
    local old_pid
    old_pid="$(cat "$PROXY_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      if curl -sf "http://127.0.0.1:12345/v1/models" -H "Authorization: Bearer ${api_key}" >/dev/null 2>&1; then
        log "OpenRouter host proxy already running (pid $old_pid)"
        return 0
      fi
      kill "$old_pid" 2>/dev/null || true
    fi
  fi

  log "starting OpenRouter host proxy on :12345..."
  nohup env OPENROUTER_PROXY_QUIET=1 OPENAI_API_KEY="$api_key" \
    python3 scripts/openrouter_host_proxy.py \
    >"${LOG_DIR}/openrouter-proxy.log" 2>&1 &
  echo $! >"$PROXY_PID_FILE"

  for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:12345/v1/models" -H "Authorization: Bearer ${api_key}" >/dev/null 2>&1; then
      log "OpenRouter host proxy is ready"
      return 0
    fi
    sleep 1
  done
  log "OpenRouter proxy did not become ready; see ${LOG_DIR}/openrouter-proxy.log"
  tail -30 "${LOG_DIR}/openrouter-proxy.log" || true
  exit 1
}

ensure_hf_cache_dir() {
  mkdir -p "$HF_CACHE_DIR"
}

normalize_hf_hub_cache_layout() {
  # Docling may download into $HF_HOME/hub when HF_HUB_CACHE is unset; embedding
  # bake uses the repository root. Cloud overrides set HF_HUB_CACHE=/models/hf.
  local legacy_hub="${HF_CACHE_DIR}/hub"
  if [[ ! -d "$legacy_hub" ]]; then
    return 0
  fi
  log "merging ${legacy_hub} into ${HF_CACHE_DIR} for a single HF_HUB_CACHE tree..."
  if [[ -d "${legacy_hub}/blobs" ]]; then
    mkdir -p "${HF_CACHE_DIR}/blobs"
    shopt -s dotglob nullglob
    for blob in "${legacy_hub}/blobs"/*; do
      [[ -e "$blob" ]] || continue
      local name
      name="$(basename "$blob")"
      if [[ ! -e "${HF_CACHE_DIR}/blobs/${name}" ]]; then
        mv "$blob" "${HF_CACHE_DIR}/blobs/${name}"
      fi
    done
    shopt -u dotglob nullglob
    rmdir "${legacy_hub}/blobs" 2>/dev/null || true
  fi
  shopt -s dotglob nullglob
  for item in "$legacy_hub"/*; do
    [[ -e "$item" ]] || continue
    local base
    base="$(basename "$item")"
    if [[ "$base" = "blobs" ]]; then
      continue
    fi
    if [[ -e "${HF_CACHE_DIR}/${base}" ]]; then
      continue
    fi
    mv "$item" "${HF_CACHE_DIR}/${base}"
  done
  shopt -u dotglob nullglob
  rmdir "$legacy_hub" 2>/dev/null || true
  sudo chown -R "${USER:-ubuntu}:${USER:-ubuntu}" "$HF_CACHE_DIR" 2>/dev/null || true
}

seed_hf_cache_from_image() {
  if [[ -f "$HF_CACHE_SEED_MARKER" ]]; then
    log "host HF cache already seeded (${HF_CACHE_DIR})"
    return 0
  fi
  ensure_hf_cache_dir
  log "seeding host HF cache from rag-service image layers..."
  docker_cli run --rm "$RAG_IMAGE_SEED" tar -C /models/hf -cf - . | tar -C "$HF_CACHE_DIR" -xf -
  touch "$HF_CACHE_SEED_MARKER"
}

warm_docling_models_in_cache() {
  if [[ -f "$HF_CACHE_DOCLING_MARKER" ]]; then
    log "Docling layout models already present in host HF cache"
    return 0
  fi
  if [[ ! -f "$SAMPLE_PDF" ]]; then
    log "missing sample PDF for Docling warmup: $SAMPLE_PDF"
    exit 1
  fi
  ensure_hf_cache_dir
  log "warming Docling / Hugging Face layout models into host HF cache (online once)..."
  docker_cli run --rm --user "$(id -u):$(id -g)" \
    -e HF_HOME=/models/hf \
    -e HF_HUB_CACHE=/models/hf \
    -e HF_HUB_OFFLINE=0 \
    -e TRANSFORMERS_OFFLINE=0 \
    -v "${HF_CACHE_DIR}:/models/hf" \
    -v "${SAMPLE_PDF}:/tmp/sample.pdf:ro" \
    "$RAG_IMAGE_SEED" \
    python -c "from pathlib import Path; from app.services.parsing_service import parse_with_docling; data=Path('/tmp/sample.pdf').read_bytes(); parse_with_docling(data, '01030000000000.pdf'); print('docling_sample_pdf_cached')"
  normalize_hf_hub_cache_layout
  touch "$HF_CACHE_DOCLING_MARKER"
}

prefetch_docling_models() {
  log "building rag-service image for HF cache seed..."
  docker_cli build -f rag_service/Dockerfile -t "$RAG_IMAGE_SEED" "${REPO_ROOT}"
  seed_hf_cache_from_image
  normalize_hf_hub_cache_layout
  warm_docling_models_in_cache
  normalize_hf_hub_cache_layout
}

compose_up_stack() {
  log "building and starting askPDF stack (host-gateway override)..."
  compose_cmd --env-file .env "${COMPOSE_FILES[@]}" build \
    --build-arg BUILDKIT_INLINE_CACHE=1
  compose_cmd --env-file .env "${COMPOSE_FILES[@]}" up -d

  for _ in $(seq 1 180); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && curl -sf http://127.0.0.1:3000 >/dev/null 2>&1; then
      log "stack ready: frontend http://127.0.0.1:3000 api http://127.0.0.1:8000"
      return 0
    fi
    sleep 2
  done
  log "stack health check timed out"
  compose_cmd --env-file .env "${COMPOSE_FILES[@]}" ps || true
  exit 1
}

cmd_install() {
  install_docker
  ensure_dockerd
  write_env_file stack
  prefetch_docling_models
  log "install complete"
}

cmd_start() {
  ensure_dockerd
  write_env_file stack
  start_openrouter_proxy
  log "start complete (dockerd + LLM proxy)"
}

cmd_stack() {
  cmd_install
  cmd_start
  compose_up_stack
}

cmd_tests_env() {
  ensure_dockerd
  write_env_file tests
}

usage() {
  cat <<EOF
Usage: bash scripts/cloud-agent-setup.sh <install|start|stack|tests-env>

  install    Install Docker, write .env (Hermes), prefetch Docling models
  start      Ensure dockerd + OpenRouter host proxy
  stack      install + start + docker compose up (vm-hostgateway override)
  tests-env  Ensure dockerd and write .env.ci for ./run_tests.sh
EOF
}

main() {
  local sub="${1:-stack}"
  case "$sub" in
    install) cmd_install ;;
    start) cmd_start ;;
    stack) cmd_stack ;;
    tests-env) cmd_tests_env ;;
    -h | --help | help) usage ;;
    *)
      log "unknown subcommand: $sub"
      usage
      exit 1
      ;;
  esac
}

main "${1:-stack}"
