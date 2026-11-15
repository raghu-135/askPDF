# Shared benchmark PDF fixtures (manifest only)

Benchmark papers are **not** stored in git. arXiv’s [non-exclusive distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/) does not permit redistributing PDFs in a public repository.

## Manifest

[`manifest.json`](manifest.json) lists each paper’s:

- `arxiv_id` and pinned `arxiv_version`
- `pdf_url` and `sha256` (verified on download)
- page count, title, and license note
- `role`: `benchmark` or `holdout` (DPO — do not tune against)

## Download and verify

```bash
./scripts/download_benchmark_pdfs.sh
# include holdout: ./scripts/download_benchmark_pdfs.sh --include-holdout
```

PDFs land in gitignored **`.benchmark-pdf-cache/`** (`ASKPDF_BENCHMARK_PDF_CACHE` overrides the path).

Upload, benchmarks, and cloud scripts call the same helper: `scripts/benchmark_papers.py`.

## Upload into a running stack

```bash
./scripts/download_benchmark_pdfs.sh
./scripts/upload_benchmark_fixtures.sh
```

Live HTTP helpers: `scripts/live_stack_client.py`.