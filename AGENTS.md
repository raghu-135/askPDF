# Agent notes

## PDF highlight geometry vs retrieval

Highlight and coordinate alignment (bbox, page dimensions, Docling word-cell overlays) are **display-only** relative to retrieval. They must **not** change canonical sentence text, chunk boundaries, `body_text`, or `contextualized_text` used for embedding.

Highlight-only fixes belong in the frontend or in non-chunking metadata on sentences. Do **not** bump `EXTRACTION_PIPELINE_VERSION` or `RETRIEVAL_CHUNKING_VERSION` for highlight-only work unless sentence text or chunk packing logic actually changes.

See `rag_service/tests/test_highlight_retrieval_invariant_pytest.py` for the guard test and `docs/testing.md` for the test runner.

## Cloud Agent VM setup

Fresh Cursor Cloud Agent VMs need Docker, a host OpenRouter proxy, and the Compose
host-gateway override before dev or browser checks.

### One-shot bootstrap

From the repository root:

```bash
bash scripts/cloud-agent-setup.sh stack   # install + start + compose up
```

Subcommands:

| Command | Purpose |
| --- | --- |
| `install` | Install `docker.io`, write `.env` (`COMPOSE_PROFILES=hermes`), warm Docling models |
| `start` | Start `dockerd` and the OpenRouter proxy on `:12345` |
| `stack` | `install` + `start` + `docker compose -f docker-compose.yml -f docker-compose.vm-hostgateway.override.yml up -d` |
| `tests-env` | Docker + copy `.env.ci` for `./run_tests.sh` only |

Repository-managed environments can wire `.cursor/environment.json` (`install` /
`start` call the same script).

### Networking and LLM

- **Bridge DNS timeout:** use `docker-compose.vm-hostgateway.override.yml` so
  containers reach Postgres, Weaviate, and runtimes via `host.docker.internal`
  published ports instead of Compose service DNS.
- **LLM egress:** containers use `LLM_API_URL=http://host.docker.internal:12345/v1`.
  The VM host runs `scripts/openrouter_host_proxy.py`, forwarding to OpenRouter with
  `OPENAI_API_KEY` or `OPENROUTER_API_KEY` from the agent secrets (never committed).

### Docling / Hugging Face

`install` bakes embedding models and runs a sample-PDF Docling parse into the
`rag-service` image, then `docker-compose.vm-hostgateway.override.yml` sets
`HF_HUB_OFFLINE=1` so runtime does not hit unreachable HF egress from containers.

### Tests

```bash
# Stop the dev stack first if it is running (both use host 5432/8080).
docker compose -f docker-compose.yml -f docker-compose.vm-hostgateway.override.yml down 2>/dev/null || true
bash scripts/cloud-agent-setup.sh tests-env
./run_tests.sh
```

`tests-env` writes `.askpdf-cloud-agent.env` (gitignored), which enables
`docker-compose.cloud-agent.test.override.yml`: host `node_modules`, published
DB/Weaviate ports, and `host.docker.internal` URLs when Compose bridge DNS is
flaky. CI does not set this file.

### Dev URLs (stack)

- App: `http://127.0.0.1:3000`
- API: `http://127.0.0.1:8000`
