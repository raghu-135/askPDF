# Eval v3 laptop campaign summary

**PR:** https://github.com/raghu-135/askPDF/pull/134  
**Head SHA:** `eacd2c9b220c6b032b0a9770eab0e12458ce647f`  
**Stack:** `COMPOSE_PROJECT_NAME=askpdf-evals3`  
**Model:** `nvidia/nemotron-3.5-super-vl:free` (fallback `particle-ai/particle-1:free`)  
**Workflow:** `router_rag_agent` only  
**Profile:** `free` @ 20 RPM  

## CI / unit tests

| Check | Status |
| --- | --- |
| GitHub Actions CI | **Green** (Docker build, test suite, Hermes runtime proof) |
| `./run_tests.sh` (laptop) | **1449 passed**, 1 skipped |

## Datasets bundled

| Dataset | License | PDFs | Notes |
| --- | --- | --- | --- |
| BEIR SciFact | CC BY 4.0 | 2 | ir-datasets `beir/scifact/test` |
| BEIR NFCorpus | CC BY 4.0 | 2 | |
| BEIR TREC-COVID | CC BY 4.0 | 2 | `beir/trec-covid` |
| PLOS (existing) | CC BY 4.0 | 6 | unchanged |
| HotpotQA | CC BY-SA 4.0 | — | **Skipped** (CDN 403 in build env) |
| SQuAD 2.0 | CC BY-SA 4.0 | — | **Skipped** (deferred; BEIR covers IR) |
| FiQA | — | — | **Skipped** (redistribution terms) |
| Natural Questions | — | — | **Skipped** (size) |
| WikiQA | — | — | **Skipped** (no redistribution) |

## Live smoke (`smoke-v3`)

| Workflow | Cases | Retrieval | Answers | Grounding | Median latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `router_rag_agent` | 6 | 100% | 33% | 100% | 7.4 |

Artifacts: `benchmarks/reports/eacd2c9b220c-smoke-v3.{json,md}` (gitignored; copy in `~/askpdf-worktrees/evals-v3-artifacts/`).

**Notable failures:** IR fact cases returned blanket declines despite `retrieval_hit=true` — likely thin text PDFs + free VL model behavior, not harness breakage.

## Baselines / reasoning compare

**Blocked** on `tier all` upload phase: `plos-biology-0020391.pdf` indexing fails with
`asyncpg.Error: unsupported Unicode escape sequence` when persisting canonical JSON
(file hash `ae0b45941c652101f1de287f9fea8da1`). Status polling returns HTTP 500 until timeout.

**Root cause (likely):** residual invalid `\u` escapes in large Docling payloads on this stack;
`postgres_jsonb_safe` may not cover all edge cases for this paper.

**Request budget used (approx.):** ~40 chat/judge calls (smoke only) — well under 1000/day cap.

## Next steps

1. Harden `postgres_jsonb_safe` / conversion persist path for biology PDF payload.
2. Re-run `baseline-1-v3`, `baseline-2-v3`, `reasoning-none-v3`, `reasoning-high-v3` after indexing fix.
3. Seed Hotpot JSON cache offline and re-enable in `build_eval_corpus_pdfs.py`.