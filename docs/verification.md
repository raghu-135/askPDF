# Real UI verification (Cloud Agents)

Agents must prove product changes through the **running web app**, not shortcuts.
This doc is the canonical checklist; the enforced helper is
`scripts/cloud-agent-real-ui-verify.mjs` (built on `scripts/cloud-agent-e2e-lib.mjs`).

## Bring the stack up on a Cloud Agent VM

From the repository root (see also **Cloud Agent VM setup** in `AGENTS.md`):

```bash
bash scripts/cloud-agent-setup.sh stack
```

Uses `docker-compose.yml` plus `docker-compose.vm-hostgateway.override.yml` so
containers reach Postgres, Weaviate, and runtimes via `host.docker.internal`, and
so the host OpenRouter proxy on `:12345` serves `LLM_API_URL` (keys come from
agent secrets, never committed or logged).

**Known pitfalls**

| Issue | What to do |
| --- | --- |
| Compose bridge DNS timeouts | Always use the VM host-gateway override file above. |
| Hugging Face blocked | Rely on warmed `.askpdf-hf-cache/` (`install` subcommand); `HF_HUB_OFFLINE=1` in override. |
| LLM 401 | Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` in the environment dashboard. |
| LLM 429 (rate limits) | Stay under **20 RPM** for free models (e.g. `inclusionai/ling-3.2-flash:free`); exponential backoff (≥4s, 8s, 16s…) between chat turns; do not spam parallel chats. |
| Headless Chrome / computer-use **cannot decode h264** browser streams | Use Playwright screenshots from the helper (or PNG artifacts), not raw screen-share video, as UI proof. |
| Port clash with tests | Stop the dev stack before `./run_tests.sh` (both use host `5432` / `8080`). |

**URLs (stack):** app `http://127.0.0.1:3000`, API `http://127.0.0.1:8000`.

## Canonical real UI flows

Each flow must be driven through the **visible UI** (clicks, keyboard, file picker),
not substituted with API-only setup presented as “UI proof”.

1. **Project & thread** — Create a project and thread (or open existing ones) until the chat header shows **Select LLM** and the composer is reachable.
2. **Upload & index** — Upload a PDF via **Upload PDF** / the file input; wait until the composer leaves “Indexing…” and accepts input.
3. **Chat** — Select the pinned proof model, type in the **real** multiline composer, send (Enter), wait until the assistant message with **Open trace** appears and the composer is enabled again.
4. **Workspace navigation** — After upload, open the **document** workspace tab (PDF viewer) or follow a deep link you changed; screenshot that surface, not the home/onboarding page alone.
5. **HITL (when relevant)** — Turn internet search to “ask me every time”, send a web-search question, wait for `chat-human-review-panel`, click **Approve once** or **Skip this tool**, and wait until the panel closes and the run finishes (do not claim approval is done while **Approving…** is still visible).

## What does **not** count as UI proof

- API-only setup (create project/thread/upload/chat) with a screenshot of `/` or an unrelated page.
- Injected DOM overlays, fake “composer” text drawn on a screenshot, or staging mock HTML unless the task is explicitly fixture-only.
- Reused or **byte-identical** screenshots across steps (before/after must differ).
- Home / empty-state screenshots when the task concerns a thread, canvas, document tab, or HITL panel.
- Terminal `curl` output, JSON responses, or log lines **instead of** screenshots of the UI under test.

If the real UI cannot be reached (stack down, 401, unrecoverable 429, missing model),
**report blocked** with the reason. Never fabricate or recycle proof.

## Running the shared Playwright helper

```bash
# Stack must already be up.
export ASKPDF_E2E_LLM_MODEL='inclusionai/ling-3.2-flash:free'
node scripts/cloud-agent-real-ui-verify.mjs
```

Artifacts: distinct PNGs at **1440×900** under `ASKPDF_E2E_ARTIFACTS` (default `/opt/cursor/artifacts`).
The helper **throws** if:

- Any two screenshots in the run share an MD5 hash.
- `chat-composer-input`, `chat-message-list`, or the active thread workspace is missing when expected.
- The composer cannot be found or never becomes ready after indexing.

Prove the duplicate guard (no stack required):

```bash
node scripts/cloud-agent-real-ui-verify.mjs --prove-screenshot-guard
```

## Required agent report format

When claiming UI verification, include:

| Field | Content |
| --- | --- |
| **Tested SHA** | `git rev-parse HEAD` on the branch you ran. |
| **Tests** | `./run_tests.sh` (or subset) with pass/fail summary. |
| **CI** | Link or status of the PR checks for that SHA. |
| **UI steps** | Ordered list of what was actually clicked/typed (project name, thread, PDF path, question text, HITL action if any). |
| **Screenshots** | Absolute paths to **distinct** PNGs from the helper or equivalent Playwright run. |
| **Unverified** | Anything you could not exercise (and why), or explicit **blocked** state. |