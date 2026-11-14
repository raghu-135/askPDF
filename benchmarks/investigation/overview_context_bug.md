# Investigation: overview answers miss title and abstract

## Symptom

For a broad overview question such as **“What does this paper discuss?”** on a
research PDF, the UI answer can cite early pages (e.g. page 1 for arXiv metadata)
while stating that the **title, abstract, and authors are missing** from context.
Deeper pages (methods, experiments) may carry most of the quoted evidence.

This matches the failure mode in the product screenshot: the model had experiment
pages in evidence but treated front matter as absent.

## Reproduction

With the dev stack running:

```bash
python3 scripts/investigate_overview_retrieval.py
# writes /opt/cursor/artifacts/overview-investigation.json
```

Default PDF: `benchmarks/fixtures/pdfs/1706.03762.pdf` (Attention). Default
workflow: `router_rag_agent`. Use `ASKPDF_OVERVIEW_QUESTION` to vary the prompt.

The investigation script records:

- Retrieval reference pages and bounded previews from debug trace spans
- Whether assembled evidence snippets contain title / abstract tokens
- Whether the final answer text claims missing title/abstract

## Separating causes: (a) context vs (b) model

| Factor | What to check | This reproduction (Attention, Auto → Router, 2026-11-14) |
| --- | --- | --- |
| **(a) Retrieval / context** | Which chunks/pages are in `document_sources` and the synthesizer prompt; does literal title/abstract text appear? | **Yes, problem confirmed.** Chat `document_sources` (10 chunks) were **identical** across models. No chunk contained the string *Attention Is All You Need*. Sources were mostly arXiv margin, related work, references, intro §1 body, and unrelated table rows—**not** a clean title+abstract block from page 1. |
| **(b) Model reasoning** | Same context: does the model infer a good overview vs list “missing” items? | **Amplifies user-visible failure.** With the **same chunks**, `openrouter/free` produced a solid overview (title, Transformer, abstract points). `openai/gpt-4.1-mini` misread related-work as the paper’s topic and added a *“context does not include…”* gap paragraph (same pattern as the screenshot’s **What is missing** list). |

**Screenshot (`openrouter/free`):** that UI used a free model, so **(b) matters**, but the screenshot also cited deep pages while denying title/abstract—consistent with **(a)** omitting front matter from the final prompt even when retrieval touched early pages.

**Verdict:** treat as a **product bug** (users should get title/abstract from any indexed paper). Root defect is **(a) context assembly**; **(b)** determines whether the failure looks like a confident wrong summary (some paid models) vs an explicit “title/abstract missing” answer (free + mini in our runs).

Artifacts: `/opt/cursor/artifacts/overview-model-compare.json`, `scripts/compare_overview_models.py`.

## Root cause — context assembly (factor a)

Retrieval often **does** rank early pages, but the **answer-time context
assembly** drops or trims them before the final LLM call:

1. **`evidence_packet_limit` + tail selection** — `evidence_context_from_packets`
   keeps only the last *N* evidence packets (`evidence_packet_limit`, default 12).
   Broad queries can produce many packets; introductory packets are evicted first.

2. **Per-packet preview caps** — `evidence_packet_content_limit` (default 2000
   chars) stores bounded previews. Front matter may be split across packets; if
   early packets are dropped, the model never sees title/abstract text.

3. **`compact_context_text` tail truncation** — when assembled context exceeds
   `final_context_char_limit`, `compact_context_text` keeps the **suffix**
   (`value[-limit:]`), not the prefix. For overview questions where retrieval
   orders content with summaries first and details later, tail-keep can delete
   the beginning of the bundle—including page 1—while leaving later pages.

4. **Citation vs prompt mismatch** — source lists shown to the user can include
   page pointers from retrieval manifests even when those pages’ text was omitted
   from the final prompt, so the model correctly reports “listed but not included.”

No special-case overview handling exists today; the same pipeline serves all
question types (consistent with the product owner constraint).

## Proposed fix options (general, not overview-specific)

### Option A — Document-level metadata channel (medium risk)

Always attach a small **per-document header** to final context (title, authors,
abstract/snippet from parsed `document_metadata`, arXiv id) independent of chunk
retrieval ranking.

| | |
| --- | --- |
| Files | ingestion metadata writer, `langgraph_runtime/workflows/evidence.py` or prefetch bundle formatter, possibly `document_projection_service` |
| Tradeoff | Small fixed context cost per doc; metadata must stay retrieval-disjoint per `AGENTS.md` (display/metadata only, not chunk text mutation) |
| Benchmark impact | Large gain on `overview` and `summarize` cases; minor token overhead on all runs |
| Risk | Stale/wrong metadata if extraction errors; must not change embeddings |

### Option B — Bounded “read page / read section” tool (higher capability, higher cost)

Expose an existing or new **document inspection** tool so the agent can fetch page 1
or abstract sections when the working set lacks front matter—without keyword routing.

| | |
| --- | --- |
| Files | `router_rag_agent` tool policy, tool implementations under `rag_service/app/tools/`, graph nodes that schedule inspection |
| Tradeoff | Extra tool round-trips (latency/cost); relies on planner quality |
| Benchmark impact | Improves overview and fact-on-page-1 cases when the planner chooses inspection; less deterministic |
| Risk | Tool abuse loops; need global step budgets |

### Option C — Context assembly policy changes (lower scope, medium risk)

Change **general** assembly rules: head+tail window instead of suffix-only
truncation; prioritize **low page numbers** within equal relevance bands; or
interleave one early-page packet per file when building the packet list.

| | |
| --- | --- |
| Files | `langgraph_runtime/workflows/evidence.py` (`compact_context_text`, `evidence_context_from_packets`) — coordinate with open PR #94 |
| Tradeoff | Does not add tokens when under budget; behavior change for all long contexts |
| Benchmark impact | Strong on overview; possible regression if tail details were critical |
| Risk | PR collision with #94; subtle ranking regressions on long-context QA |

## Recommendation

Prefer **Option A + C**: metadata headers for stable bibliographic/overview facts,
and head-preserving truncation / per-file early-packet retention in evidence
assembly. Option B is a good follow-up for exact deep-page facts but is less
deterministic for benchmarking.

## Verdict

**Bug** — users reasonably expect title/abstract to be answerable from any
indexed paper without saying front matter is “missing.”