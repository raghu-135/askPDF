# RAG benchmark baseline — `e95b1d8aca5f`

Committed reference scores (holdout excluded). Per-run outputs stay in gitignored `benchmarks/reports/`.

- Generated: 2026-11-15T05:40:26.867551+00:00
- Chat model: `openai/gpt-4.1-mini`
- Judge model: `google/gemini-2.5-flash`

## Context baseline

- Config: `benchmarks/benchmark_config.json`
- `context_window`: 8192
- `use_reranker`: False
- `use_web_search`: False

- `context_budget`: document_ratio=0.28, durable_memory_ratio=0.06, memory_expanded_ratio=0.12, recent_ratio=0.22, semantic_ratio=0.12, usable_ratio=0.8

## Summary by workflow

| Workflow | Cases | Retrieval | Answers | Median latency (s) | LLM tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `router_rag_agent` | 40 | 85% | 80% | 10.8 | 0 |

### `router_rag_agent` by category

| Category | Cases | Retrieval | Answers | Decline | Latency (med s) | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| facts | 7 | 71% | 86% | — | 9.9 | 0 |
| method | 5 | 80% | 100% | — | 10.8 | 0 |
| multi_doc | 5 | 100% | 100% | — | 13.1 | 0 |
| out_of_scope | 5 | 100% | 0% | 0% | 10.7 | 0 |
| overview | 6 | 83% | 100% | — | 10.9 | 0 |
| results | 5 | 100% | 80% | — | 10.7 | 0 |
| tables | 7 | 71% | 86% | — | 9.9 | 0 |

| `auto_rag_agent` | 40 | 82% | 85% | 11.6 | 0 |

### `auto_rag_agent` by category

| Category | Cases | Retrieval | Answers | Decline | Latency (med s) | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| facts | 7 | 57% | 100% | — | 10.6 | 0 |
| method | 5 | 80% | 100% | — | 11.6 | 0 |
| multi_doc | 5 | 100% | 100% | — | 11.7 | 0 |
| out_of_scope | 5 | 100% | 0% | 0% | 11.7 | 0 |
| overview | 6 | 83% | 100% | — | 11.7 | 0 |
| results | 5 | 100% | 100% | — | 10.6 | 0 |
| tables | 7 | 71% | 86% | — | 11.6 | 0 |