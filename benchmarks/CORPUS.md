# RAG eval benchmark corpus

The live eval harness uses **only PDFs committed under** `benchmarks/fixtures/bundled/`.
Each file is a PLOS research article under **Creative Commons Attribution 4.0 (CC BY 4.0)**,
which allows redistribution with attribution. Sources and licenses are listed in
[`fixtures/LICENSES.md`](fixtures/LICENSES.md) and `fixtures/bundled_manifest.json`.

## Why PLOS CC BY

| Source | License | Bundled in repo? | Used in eval? |
| --- | --- | --- | --- |
| [PLOS journals](https://plos.org) (ONE, Biology, Medicine, Pathogens, Genetics, Computational Biology) | CC BY 4.0 | Yes | **Primary corpus** |
| [arXiv](https://arxiv.org) preprints (Attention, BERT, RAG, LoRA, surveys, DPO, etc.) | arXiv non-exclusive distribution — **do not redistribute PDFs** | No (runtime fetch only) | Legacy `benchmark_papers.py` / POC only |
| [BEIR / NFCorpus / SciFact](https://github.com/beir-cellar/beir) | Dataset-specific (often CC BY); text shards, not full PDFs | No | Not used (future IR metrics) |
| [WikiQA](https://www.microsoft.com/en-us/download/details.aspx?id=52419) | Microsoft Research license | No | Not used |
| US government works (NASA, CDC factsheets) | Public domain / US gov | Possible future add | Not in this PR |

PLOS gives a stable, multi-domain set (ecology, stats editorial, RNA biology, medicine essay,
pathogens, yeast genetics) that exercises tables, author lines, multi-page facts, and declines
without relying on arXiv’s no-redistribution policy.

## Holdout split

Holdout questions live in [`holdout_questions.json`](holdout_questions.json) and reference the
same **bundled** PDFs. Run with `--tier holdout`. Do not tune defaults against holdout scores.

## Removed fixtures

- **Mem0 / arXiv `2504.19413`** — removed from the eval suite; arXiv PDFs are not bundled and Mem0 is out of scope for this harness.