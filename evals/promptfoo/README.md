# promptfoo + live-stack RAG evals

## Quick start (live stack required)

```bash
bash scripts/cloud-agent-setup.sh stack
./evals/promptfoo/run_eval.sh --tier smoke --judge --label smoke-1
```

The runner waits for `/health` and `/ready`, uploads only **bundled CC BY PDFs**, and
indexes via the files API (no fixed sleeps). See `benchmarks/CORPUS.md`.

## Profiles (CLI + env)

| Profile | Behavior | Key env |
| --- | --- | --- |
| `free` (default) | Sequential, ~20 RPM | `ASKPDF_EVAL_PROFILE=free`, `ASKPDF_EVAL_FREE_RPM=20` |
| `fast` | Parallel cases (bounded workers) | `ASKPDF_EVAL_PROFILE=fast`, `ASKPDF_EVAL_MAX_WORKERS` |

Models: `ASKPDF_BENCHMARK_MODEL` (default `nvidia/nemotron-3.5-super-vl:free`), fallback
`ASKPDF_BENCHMARK_MODEL_FALLBACK=particle-ai/particle-1:free` — never `openrouter/free`.

Reasoning: eval sets `ASKPDF_LLM_REASONING_EFFORT=none` → OpenRouter `reasoning.effort=none`.
Compare with `--reasoning-compare` (core/advanced).

Checkpoints: `benchmarks/reports/.checkpoint-<label>.json` (resume unless `--no-resume`).

## Tiers

```bash
python3 scripts/rag_eval_suite.py --tier smoke    # ~5 cases, CI gate candidate
python3 scripts/rag_eval_suite.py --tier core
python3 scripts/rag_eval_suite.py --tier advanced
python3 scripts/rag_eval_suite.py --tier holdout  # holdout_questions.json
```

## Generate smoke promptfoo tests

```bash
python3 scripts/generate_promptfoo_tests.py   # writes tests.generated.yaml (smoke tier)
bash scripts/verify_promptfoo_tests_generated.sh
```

## Sweeps (12-case `sweep_core` suite)

```bash
./evals/promptfoo/run_sweep.sh document_ratio_0.36 --judge
```

Uses merged sweep config and `--verify-settings`.