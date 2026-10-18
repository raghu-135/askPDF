# Agent notes

## PDF highlight geometry vs retrieval

Highlight and coordinate alignment (bbox, page dimensions, pdfplumber overlays) are **display-only** relative to retrieval. They must **not** change canonical sentence text, chunk boundaries, `body_text`, or `contextualized_text` used for embedding.

Highlight-only fixes belong in the frontend or in non-chunking metadata on sentences. Do **not** bump `EXTRACTION_PIPELINE_VERSION` or `RETRIEVAL_CHUNKING_VERSION` for highlight-only work unless sentence text or chunk packing logic actually changes.

When serve/publish-time reading overlay logic changes (for example `apply_table_row_highlight_boxes` in `publish_reading_projection`), bump `READING_PROJECTION_VERSION` in `file_processing_service.py` so already-ingested documents republish from stored canonical JSON without re-parse or re-embed.

See `rag_service/tests/test_highlight_retrieval_invariant_pytest.py` for the guard test and `docs/testing.md` for the test runner.
