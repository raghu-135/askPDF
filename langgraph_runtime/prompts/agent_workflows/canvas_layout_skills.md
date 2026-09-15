# Research canvas layout skills

These skills apply only when `publish_canvas` is available. They are prompt recipes for the existing `canvas_spec_v1` blocks (`stat`, `table`, `callout`, `markdown`, `sources`, `dag`). Do not invent block types, scripts, or javascript URLs. Pick exactly one layout that matches the user request. Every published canvas must include a `sources` block whose document citations use `file_hash` values attached to this thread.

After a successful publish, mention the canvas title in the answer. If evidence is too thin for the chosen layout, say so instead of fabricating rows.

## canvas_compare_papers

Use when the user wants to compare, contrast, or overlap two or more papers, PDFs, or named documents.

Suggested sections:

1. Overview — 2–4 `stat` blocks (papers compared, shared claims, disagreements, unresolved items).
2. Comparison — one `table` with headers such as `Claim | Source A | Source B` (add a column per compared source). Cells must be short and evidence-backed.
3. Gaps — a `callout` with `tone=warning` for disagreements or missing evaluations.
4. Sources — `sources` citations for every compared document (`kind=document`, `file_hash`, optional `sentence_id`).

Optional: a `dag` of claim nodes pointing at supporting source nodes.

## canvas_evidence_matrix

Use when the user wants support, contradiction, or coverage of claims across sources (an evidence matrix).

Suggested sections:

1. Coverage — `stat` blocks for supported, contradicted, and uncovered claims.
2. Matrix — one `table` with headers `Claim | Source | Support | Notes`. Use Support values such as `supported`, `contradicted`, `absent`, or `partial`.
3. Conflicts — a `callout` when sources disagree.
4. Sources — `sources` citations for every matrix source.

Optional: a `dag` from claim nodes to source nodes.

## canvas_timeline

Use when the answer depends on chronology, sequence, first/latest, before/after, since, or event order.

Suggested sections:

1. Span — `stat` blocks for event count and time range.
2. Timeline — one `table` with headers `Time | Event | Source | Notes`, oldest row first.
3. Caveats — a `callout` when timestamps are thread-added times rather than publication dates. `document_available_in_thread_at` means added to this thread.
4. Sources — `sources` citations (`document`, `conversation`, `web`, or `memory` as appropriate).

Optional: a `dag` chaining events in time order.
