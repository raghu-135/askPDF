# promptfoo eval POC (askPDF)

Minimal replacement sketch for `benchmarks/run_benchmark.py`. Requires a live
stack (`bash scripts/cloud-agent-setup.sh stack`) and benchmark PDFs:

```bash
./scripts/download_benchmark_pdfs.sh
```

## Smoke eval (5 Attention-paper questions)

```bash
export ASKPDF_BENCHMARK_MODEL='inclusionai/ling-3.2-flash:free'
export PROMPTFOO_JUDGE_MODEL='openrouter:inclusionai/ling-3.2-flash:free'
./evals/promptfoo/run_poc.sh
```

Uses `scripts/live_stack_client.py` and `scripts/benchmark_config.py` (same
context baseline as `benchmarks/benchmark_config.json`).

Rate limit: `maxConcurrency: 1` and 3.5s delay in config (<20 OpenRouter RPM).
Judge calls use `OPENROUTER_API_KEY` from the environment (never logged).

Outputs: `evals/promptfoo/output/poc-results.json` and console summary.