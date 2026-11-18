# Holdout benchmark split

The DPO paper (`2305.18290.pdf`, *Direct Preference Optimization*) and the
questions in [`holdout_questions.json`](holdout_questions.json) form a **holdout
evaluation split**.

## Rules

- Do **not** use holdout questions or answers to tune retrieval, prompting, or
  workflow defaults.
- Do **not** cherry-pick model or workflow changes against holdout scores.
- Report holdout metrics alongside the main benchmark for regression visibility
  only.

## Running holdout cases

```bash
# Holdout: extend rag_eval_suite with holdout_questions.json (on-demand; not default CI).
```

Main-tuning questions are in [`questions.json`](questions.json).