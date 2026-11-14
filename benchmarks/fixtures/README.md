# Shared benchmark PDF fixtures

Redistributable research PDFs used by the RAG benchmark suite, live VM proofs, and
manual QA. Total payload is about **12 MB** (five benchmark papers plus one holdout).

## Papers

| File | Title | Pages | Role |
| --- | --- | ---: | --- |
| `1706.03762.pdf` | Attention Is All You Need | 15 | benchmark |
| `1810.04805.pdf` | BERT | 16 | benchmark |
| `2005.11401.pdf` | Retrieval-Augmented Generation | 19 | benchmark |
| `2106.09685.pdf` | LoRA | 26 | benchmark |
| `2303.18223.pdf` | A Survey of Large Language Models | 144 | benchmark (long) |
| `2305.18290.pdf` | Direct Preference Optimization (DPO) | 27 | **holdout** |

Machine-readable metadata (source URLs, byte sizes, license text) lives in
[`manifest.json`](manifest.json).

## License

Each PDF is an arXiv preprint distributed under the
[arXiv.org perpetual, non-exclusive license 1.0](https://arxiv.org/licenses/nonexclusive-distrib/1.0/),
which permits redistribution with attribution. This is comparable to allowing
public fixture copies in an open-source repository (not a substitute for CC BY,
but explicitly verified for redistribution).

## Upload into a running stack

From the repository root, with the dev stack up (`http://127.0.0.1:8000`).

Live-stack HTTP helpers used by upload, benchmark, and investigation scripts live in
`scripts/live_stack_client.py`.

```bash
./scripts/upload_benchmark_fixtures.sh
# optional: ASKPDF_BENCHMARK_THREAD_ID=<id> to attach to an existing thread
# optional: ASKPDF_INCLUDE_HOLDOUT=1 to also index the holdout PDF
```

The script creates a thread (or uses `ASKPDF_BENCHMARK_THREAD_ID`), uploads PDFs,
and waits until indexing and projection complete. It prints `file_hash` values for
each fixture.

## Holdout paper

`2305.18290.pdf` (DPO) is indexed only when `ASKPDF_INCLUDE_HOLDOUT=1`.
Questions that target this paper live in [`../holdout_questions.json`](../holdout_questions.json)
and are documented as **not for tuning** in [`../HOLDOUT.md`](../HOLDOUT.md).