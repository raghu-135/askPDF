# Replacing the homegrown RAG benchmark harness

This note compares evaluation frameworks against the current suite under
`benchmarks/` (PR #101): ~40 JSON-defined cases, live `rag_service` E2E via
`scripts/live_stack_client.py`, retrieval page checks, string/numeric matchers,
optional LLM judge, committed aggregates in `benchmarks/baseline.json`, and
on-demand runs (not default CI).

## Recommendation

**Adopt [promptfoo](https://www.promptfoo.dev/) as the primary harness**, with an
optional **Ragas** (or DeepEval metric) layer later for standardized RAG scores
on retrieved contexts—not as the orchestrator.

**Why promptfoo fits askPDF best**

| Need | Homegrown today | promptfoo |
| --- | --- | --- |
| E2E via HTTP API + both runtimes | Custom Python loop per workflow | Python `file://` / `python:` custom provider reuses `live_stack_client` |
| Grounding / faithfulness / citation | Ad-hoc judge + page intersection | `contains`, `javascript`, `llm-rubric`, `context-*` assertions + metadata from provider |
| OpenRouter target + judge | Env models + `/v1` on rag-service | Native `openrouter:` providers; judge separate from system-under-test |
| Baseline / regression | `baseline.json` delta print | Git-tracked config + `promptfoo eval` JSON output; CI threshold or compare script |
| Dataset versioning | `questions.json` + holdout file | Same JSON/YAML in repo; promptfoo supports `tests: file://` |
| Self-host / license | N/A | MIT, CLI-only, no SaaS required |
| Maintenance | ~440 lines bespoke runner | Thin provider + config; framework handles parallelization, reporting, UI |

Ragas and DeepEval excel at **metric math** on `(question, answer, contexts)` but
do not natively drive multi-workflow agent runs, PDF upload/index waits, or
per-runtime Compose profiles. LangSmith/Langfuse add valuable tracing but tie
regression workflows to SaaS unless you self-host enterprise tiers. Inspect AI and
OpenAI Evals target model capability benchmarks, not product RAG over your API.

## Candidate comparison

### Ragas

- **RAG metrics:** Strong—faithfulness, context precision/recall, answer
  relevancy, plus custom metrics. Citation is indirect (needs context text, not
  askPDF page refs unless you encode them).
- **E2E askPDF:** You write a runner that calls the API, then feed a
  `EvaluationDataset`; Ragas does not invoke threads/workflows for you.
- **OpenRouter:** Via LangChain/OpenAI-compatible wrapper on `openrouter.ai/api/v1`.
  Free judges work if the model supports JSON/tool output Ragas expects.
- **CI / baseline:** Experiments API and `ragas evaluate`; no first-class “compare
  to committed baseline” like promptfoo’s output diff patterns— you script it.
- **Hosting / cost:** Apache 2.0, self-hosted Python; judge LLM cost only.
- **Migration carryover:** All questions/contexts as dataset rows; retrieval page
  rules become custom metrics; `baseline.json` aggregates reimplemented as
  experiment summaries.

### DeepEval

- **RAG metrics:** Faithfulness, contextual precision/recall, G-Eval; good
  agentic extensions.
- **E2E askPDF:** `LLMTestCase` + your API wrapper; `deepeval test run` (pytest) for CI.
- **OpenRouter:** First-class `OpenRouterModel` integration.
- **CI / baseline:** Pytest assertions, Confident AI SaaS optional for dashboards.
- **Hosting / cost:** Apache 2.0 core; Confident AI is paid SaaS.
- **Carryover:** High—Python-native next to existing scripts; ~40 cases map to
  test file or dataset JSON.

### promptfoo (recommended)

- **RAG metrics:** Model-graded (`llm-rubric`, `factuality`, `context-*`) plus
  deterministic assertions; optional Ragas extension in same eval job.
- **E2E askPDF:** Custom Python provider calls `thread_chat` / `scored_agent_run`;
  one config row per workflow (`langgraph` vs `hermes` = env or provider label).
- **OpenRouter:** `openrouter:<model>` for judge; SUT model via `llm_model` on API.
- **CI / baseline:** `promptfoo eval` exit codes, `--output` JSON, shareable web
  report; store golden output or aggregate script vs `baseline.json`.
- **Hosting / cost:** MIT, Node CLI; no account required.
- **Carryover:** Import `questions.json`/`holdout` as test vars; `answer_match`
  → assertions; `expected_pages` → provider metadata + `javascript` assert.

### LangSmith evals

- **RAG metrics:** Datasets, evaluators (including custom code), online/offline.
- **E2E askPDF:** Possible via HTTP tool or wrapping your client in an evaluator
  chain; LangGraph integration is natural if you export traces from runtime.
- **OpenRouter:** Custom model endpoint configuration in LangChain.
- **CI / baseline:** Good in LangChain-centric repos; baseline = dataset versions
  in Smith.
- **Hosting / cost:** SaaS (free tier limits); enterprise self-host available.
- **Carryover:** Questions migrate to Smith datasets; baseline is re-recorded, not
  a literal import of `baseline.json`.

### Langfuse evals

- **RAG metrics:** Scores on traces, LLM-as-judge, experiments.
- **E2E askPDF:** Requires instrumenting rag_service or post-hoc trace ingestion;
  less suited as the driver for upload→index→chat unless already fully traced.
- **OpenRouter:** Via LiteLLM/OpenAI-compatible client on judges.
- **CI / baseline:** Self-host OSS core; eval UI in cloud or self-host.
- **Carryover:** Medium—questions as dataset; committed baseline needs re-baselining
  in Langfuse experiments.

### OpenAI Evals

- **RAG metrics:** Generic grading templates; not RAG-specialized.
- **E2E askPDF:** Custom `class` eval that HTTP-calls your stack.
- **OpenRouter:** Possible via compatibility shims; not documented as primary.
- **CI / baseline:** Registry of evals; declining community momentum vs newer tools.
- **Carryover:** Low ergonomics for multi-workflow PDF suite.

### Inspect AI

- **RAG metrics:** Task/solver abstractions; RAG via custom scorers.
- **E2E askPDF:** Custom solver hitting API—feasible but idiomatic for model evals
  (MMLU, etc.), not long-running indexing pipelines.
- **OpenRouter:** Provider plugins exist.
- **CI / baseline:** Strong for research; heavier ops for product regression.
- **Carryover:** Poor fit for retaining page-level retrieval checks without
  substantial custom code.

### Other options considered

- **Phoenix (Arize) / Braintrust / Maxim:** Strong observability + eval SaaS;
  similar to Langfuse with less alignment to “git-native harness in this repo.”
- **promptfoo + Ragas hybrid:** Matches how many teams split fast PR gates
  (promptfoo) from retrieval tuning (Ragas on demand)—matches `docs/testing.md`
  “benchmark on demand” stance.

## OpenRouter: target and judge

| Framework | SUT (chat) | Judge |
| --- | --- | --- |
| promptfoo | Custom provider → `llm_model` on API (OpenRouter id) | `openrouter:inclusionai/ling-3.2-flash:free` or env pin |
| DeepEval / Ragas | Your HTTP client | `OpenRouterModel` / LangChain base URL |
| LangSmith / Langfuse | Trace + optional wrapper | Configured judge model |

**Free judge:** Use a **named** free model (e.g. `inclusionai/ling-3.2-flash:free`),
not `openrouter/free`. Rate-limit ≤20 RPM (promptfoo `delay` + `maxConcurrency: 1`;
custom backoff on 429 in the provider).

**Hermes vs LangGraph:** Keep one provider; select runtime via Compose profile /
`ASKPDF_BENCHMARK_API` or workflow list in CI matrix—not via framework limitation.

## What carries over from the current suite

| Asset | Carryover |
| --- | --- |
| `benchmarks/questions.json` (40) | Direct import as promptfoo tests (`vars` + `assert`) |
| `holdout_questions.json` | Separate config or `defaultTest` tag; same holdout rules in docs |
| `benchmark_config.json` | Unchanged; provider loads via existing `benchmark_config.py` |
| `fixtures/manifest.json` + download scripts | Unchanged pre-step in eval shell wrapper |
| `baseline.json` / `baseline.md` | One-time translation to promptfoo baseline output or threshold table in CI; then delete JSON baseline when harness is removed |
| `rubric.md` | Becomes `llm-rubric` text / assertion metadata |
| Per-case `expected_pages` | Provider metadata + `javascript` assertion |
| `numeric` / `answer_match` | `javascript` / `contains` / `llm-rubric` |

Roughly **95% of case definitions** move verbatim; only the runner and aggregate
reporting change.

## Migration plan (delete homegrown harness last)

1. **Land promptfoo POC** under `evals/promptfoo/` (5 smoke cases, pinned free
   judge, one workflow). Document commands in `evals/promptfoo/README.md`.
2. **Port provider logic** from `benchmarks/run_benchmark.py` into a shared module
   (e.g. `scripts/benchmark_case.py`) used by both runners temporarily—DRY retrieval
   and scoring helpers.
3. **Generate full test file** from `questions.json` (scripted one-time or CI
   check) so a single source of truth remains JSON until optional YAML split.
4. **Dual-run gate (one release):** `./scripts/run_rag_benchmark.sh --limit 5` and
   `evals/promptfoo/run_eval.sh --smoke` must agree on pass/fail for smoke set.
5. **Expand to 40 cases × default workflows**; add CI job `eval-rag-smoke` (5 cases,
   both runtimes matrix optional/manual) on schedule or `workflow_dispatch`.
6. **Baseline migration:** Run full eval at `main`, commit promptfoo
   `output.json` snapshot or regenerated `benchmarks/baseline.json` from promptfoo
   aggregates; wire compare in CI (fail on retrieval/answer regression thresholds).
7. **Holdout:** Separate promptfoo config invoked only with
   `ASKPDF_INCLUDE_HOLDOUT=1`; document in `HOLDOUT.md`.
8. **Remove** `benchmarks/run_benchmark.py`, `run_rag_benchmark.sh` wrapper, and
   baseline delta code paths; keep `questions.json` as source or fold into
   `evals/promptfoo/tests/` with generator deleted if desired.
9. **Optional phase 2:** Ragas batch on retrieved chunk text for context precision
   when tuning chunking (do not run on every PR—per AGENTS.md versioning rules).

Each step is verifiable without the next; do not delete the homegrown runner until
step 6’s baseline is committed and step 4’s dual-run passes on full smoke.

## Proof of concept

See `evals/promptfoo/README.md` and branch `cursor/eval-framework-research-6c14`.