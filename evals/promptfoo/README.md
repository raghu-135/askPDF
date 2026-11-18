# promptfoo + live-stack RAG evals

## Quick start (live stack required)

```bash
bash scripts/cloud-agent-setup.sh stack
./evals/promptfoo/run_eval.sh --judge --label baseline-1
```

## Generate tests from `benchmarks/questions.json`

```bash
python3 scripts/generate_promptfoo_tests.py   # writes tests.generated.yaml
```

## Sweeps (12-case `sweep_core` suite)

```bash
./evals/promptfoo/run_sweep.sh document_ratio_0.36 --judge
```

Uses `ASKPDF_BENCHMARK_CONFIG` merged from `benchmarks/benchmark_config.eval.json` and
`evals/sweeps/<name>.json`. `--verify-settings` is enabled for sweeps.

Models (override via env): `ASKPDF_BENCHMARK_MODEL=nvidia/nemotron-3.5-super-vl:free`,
judge `ASKPDF_BENCHMARK_JUDGE_MODEL=inclusionai/ling-3.2-flash:free`, rate limit
`ASKPDF_EVAL_MIN_INTERVAL_S=3.5`.