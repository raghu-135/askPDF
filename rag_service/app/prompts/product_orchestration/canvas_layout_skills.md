# Research canvas layout skills

These skills apply only when `publish_canvas` is available. They map user intent to a **draft** the server compiles into `canvas_spec_v1`. Do not hand-write blocks or `canvas_spec_v1` JSON.

Draft shape: `layout` (`compare`, `matrix`, or `timeline`), `title`, optional `summary`, and `claims` (each claim has `text`, `source_ids` copied from the Citation menu, optional `group` for compare columns, optional `tone` for matrix cells, optional `time` for timeline). The server builds stats, tables, sources, quotes, and timelines.

Pick exactly one layout skill. Every claim must use `source_id` values from the Citation menu. If evidence is too thin, say so instead of inventing ids.

After a successful publish, mention the canvas title in the answer.

Use `supersedes_id` only when replacing a canvas listed under **Current canvases on this thread**. Never use `file_hash` or document names as `supersedes_id`.

## canvas_compare_papers

Use when the user wants to compare, contrast, or overlap two or more papers, PDFs, or named documents.

- `layout`: `compare`
- Set `group` on each claim to the document or source column name (for example the file name).
- Include one claim per compared point; use `source_ids` for the supporting passage.

## canvas_evidence_matrix

Use when the user wants support, contradiction, or coverage of claims across sources.

- `layout`: `matrix`
- Set `tone` on each claim to `supported`, `contradicted`, `partial`, or `absent` when the evidence supports that label.

## canvas_timeline

Use when the answer depends on chronology, sequence, first/latest, before/after, since, or event order.

- `layout`: `timeline`
- Every claim needs `time` (oldest-first order in the draft).
- Attach the supporting `source_id` on each claim.
