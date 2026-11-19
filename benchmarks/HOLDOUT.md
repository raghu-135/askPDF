# Holdout benchmark split

Holdout questions are in [`holdout_questions.json`](holdout_questions.json). They use the same
**bundled CC BY PLOS PDFs** as the main corpus (`benchmarks/CORPUS.md`).

## Rules

- Do **not** use holdout questions or answers to tune retrieval, prompting, or workflow defaults.
- Do **not** cherry-pick model or workflow changes against holdout scores.
- Report holdout metrics alongside the main benchmark for regression visibility only.

## Running holdout cases

```bash
python3 scripts/rag_eval_suite.py --tier holdout --label holdout-run
```

Main-tuning tiers: `--tier smoke|core|advanced`. See `evals/promptfoo/README.md`.