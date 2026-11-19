# RAG eval benchmark corpus

The live eval harness uses **only PDFs committed under** `benchmarks/fixtures/bundled/`.
Sources and licenses are listed in [`fixtures/LICENSES.md`](fixtures/LICENSES.md),
`fixtures/bundled_manifest.json`, and `fixtures/corpus_build_metadata.json` (pins for
the conversion script).

## Bundled sources

| Family | License | Bundled? | Eval? |
| --- | --- | --- | --- |
| [PLOS journals](https://plos.org) | CC BY 4.0 | Yes | **Primary (legacy)** |
| BEIR: SciFact, NFCorpus, TREC-COVID | CC BY 4.0 / typical PMC | Yes (generated PDFs) | **IR passages** |
| HotpotQA (distractor passages) | CC BY-SA 4.0 | Yes | Multi-hop style Qs |
| SQuAD 2.0 passage sample | CC BY-SA 4.0 | Yes | Extractive Qs |

### Not bundled (redistribution or policy)

| Source | Reason |
| --- | --- |
| FiQA (BEIR) | Challenge terms; omitted from committed corpus |
| Natural Questions | Size; CC BY-SA 3.0 would allow a future subset |
| WikiQA | Microsoft license — no redistribution |
| arXiv full PDFs | arXiv non-exclusive distribution policy |
| Mem0 / arXiv `2504.19413` | Out of scope for this harness |

PLOS fixtures exercise real PDF layout (tables, authorship). IR bundles group several
plain-text passages per PDF so retrieval is tested on scientific and encyclopedic prose
without violating dataset redistribution rules.

## Regenerating IR fixtures

```bash
python3 -m venv .corpus-venv
.corpus-venv/bin/pip install -r scripts/requirements-corpus-build.txt
.corpus-venv/bin/python scripts/build_eval_corpus_pdfs.py --merge-questions
python3 scripts/generate_promptfoo_tests.py
```

Then refresh `bundled_manifest.json` shas (the script updates manifest entries) and
commit PDFs + `questions.json`.

## Holdout split

Holdout questions live in [`holdout_questions.json`](holdout_questions.json) and reference the
same **bundled** PDFs. Run with `--tier holdout`. Do not tune defaults against holdout scores.

## Removed fixtures

- **Mem0 / arXiv `2504.19413`** — removed from the eval suite; arXiv PDFs are not bundled and Mem0 is out of scope for this harness.