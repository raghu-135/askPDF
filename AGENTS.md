# Agent notes

## PDF highlight geometry vs retrieval

Highlight and coordinate alignment (bbox, page dimensions, Docling word-cell overlays) are **display-only** relative to retrieval. They must **not** change canonical sentence text, chunk boundaries, `body_text`, or `contextualized_text` used for embedding.

Highlight-only fixes belong in the frontend or in non-chunking metadata on sentences. Do **not** bump `EXTRACTION_PIPELINE_VERSION` or `RETRIEVAL_CHUNKING_VERSION` for highlight-only work unless sentence text or chunk packing logic actually changes.

See `rag_service/tests/test_highlight_retrieval_invariant_pytest.py` for the guard test and `docs/testing.md` for the test runner.

## Real UI verification

Cloud Agents must not substitute API output, recycled screenshots, or fake overlays for browser proof. Follow **`docs/verification.md`**: stack bootstrap, canonical UI flows (project/thread → upload → chat → document tab / HITL), forbidden shortcuts, and the required report format. Run `node scripts/cloud-agent-real-ui-verify.mjs` when proving chat/PDF UI on a VM (optional `--prove-screenshot-guard` for the duplicate-screenshot check).

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
| `stack` | `install` + `start` + compose up with `docker-compose.cloud-agent.override.yml` |
| `tests-env` | Docker + copy `.env.ci` for `./run_tests.sh` only |

Repository-managed environments can wire `.cursor/environment.json` (`install` /
`start` call the same script).

### Networking and LLM

- **Bridge DNS timeout:** `docker-compose.cloud-agent.override.yml` (loaded only on
  cloud VMs) rewires containers to Postgres, Weaviate, and runtimes via
  `host.docker.internal` published ports instead of Compose service DNS.
- **LLM egress:** containers use `LLM_API_URL=http://host.docker.internal:12345/v1`.
  The VM host runs `scripts/openrouter_host_proxy.py`, forwarding to OpenRouter with
  `OPENAI_API_KEY` or `OPENROUTER_API_KEY` from the agent secrets (never committed).

### Docling / Hugging Face

`install` builds `rag-service`, copies image-layer weights into gitignored
`.askpdf-hf-cache/`, then runs one online Docling sample-PDF parse into that
directory. The cloud-agent override mounts `.askpdf-hf-cache` on `/models/hf` (with
`HF_HUB_CACHE=/models/hf` so baked
embedding weights and Docling hub artifacts share one tree) and sets
`HF_HUB_OFFLINE=1` so containers
use the cache instead of Hugging Face egress (often blocked in cloud VMs).

Re-run `bash scripts/cloud-agent-setup.sh install` to rebuild the cache; delete
`.askpdf-hf-cache/` to force a full re-download.

### LLM credentials

`install` / `start` copy `OPENROUTER_API_KEY` (preferred) or `OPENAI_API_KEY`
from the Cursor Cloud Agent secrets into `.env` as `OPENAI_API_KEY`. Containers
call the host OpenRouter proxy at `http://host.docker.internal:12345/v1`
(`LLM_API_URL` in `.env`); the proxy forwards to `https://openrouter.ai/api/v1`.
If chat fails with 401, confirm a secret is set in the environment dashboard
(never commit keys).

### Tests

```bash
# Stop the dev stack first if it is running (both use host 5432/8080).
docker compose -f docker-compose.yml -f docker-compose.cloud-agent.override.yml down 2>/dev/null || true
bash scripts/cloud-agent-setup.sh tests-env
./run_tests.sh
```

Product and LangGraph containers run Postgres migrations at startup (advisory
lock); Compose no longer ships one-shot `db-migrate` / `runtime-db-bootstrap`
services. Hermes gateway builds pull the pinned upstream git context directly
(no `hermes-upstream` Compose service).

`tests-env` writes `.askpdf-cloud-agent.env` (gitignored), which sets
`ASKPDF_USE_CLOUD_AGENT_OVERRIDE=1` so `./run_tests.sh` loads
`docker-compose.cloud-agent.override.yml` (host `node_modules`, published
DB/Weaviate ports, HF cache, host-gateway URLs). CI and laptop runs without
that file use only `docker-compose.yml` + `.env`.

On a normal Mac, `cp .env.example .env && docker compose up --build` respects
`LLM_API_URL` from `.env` (no auto-loaded override). Optional frontend hot reload:
`docker compose -f docker-compose.yml -f docker-compose.development.override.yml up --build`.

### Dev URLs (stack)

- App: `http://127.0.0.1:3000`
- API: `http://127.0.0.1:8000`

### Browser workbench control backend

The `browser-capture` container exposes screenshot/act via **CDP** on the shared Brave
debug port (`control_backend: cdp` in `/health`). Pelorus / Pixelflux Computer Use is
**deferred** on the current X11 + Brave image (`PELORUS=false` in Compose); do not add
Pelorus-only paths in feature PRs unless the image gains a supported desktop-control
stack again.
