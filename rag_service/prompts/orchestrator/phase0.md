---
PHASE 0 - PREPROCESS  (internal only - no output visible to user)
---
No upstream query rewriter ran for this request. Execute these four micro-steps
internally before calling any tool or writing the Phase 2 plan:

STEP 1 — COREFERENCE RESOLUTION
Replace every pronoun ("it", "this", "that", "they", "the document", "the method")
with its explicit referent resolved from the conversation messages above.
  Replace pronoun-only or deictic references with explicit entities or concepts.

STEP 2 — STANDALONE-IFY
Add only the minimum subject/domain context so a cold vector search retrieves the right
chunks. Do NOT add subtopics or angles the user never mentioned — extra terms dilute the
embedding vector and return wrong chunks.
  Make the query standalone with minimal context; never add unasked subtopics.

STEP 3 — SCOPE & COVERAGE ASSESSMENT
  a) Preserve the user's question scope exactly — do not widen or narrow it.
  b) Review the PRE-FETCHED CONTEXT block (if present) and internally classify coverage:
       SUFFICIENT          → pre-fetch directly answers the working query; skip retrieval
                             tools (except search_web if enabled).
       PROBABLY_SUFFICIENT → partial content present; one targeted call may sharpen it.
       INSUFFICIENT        → pre-fetch lacks the specific detail; full retrieval needed.
  c) IMPORTANT: pre-fetched document evidence was retrieved with the raw, unprocessed question.
     If your working query is more specific than the original, the pre-fetch may have
     missed relevant chunks — call search tools with the working query to compensate.

STEP 4 — AMBIGUITY CHECK  (high bar)
Does the working query still have 2+ genuinely different interpretations that conversation
history cannot resolve?
  YES → your first (and only) tool call MUST be ask_for_clarification.
  NO  → proceed to Phase 1.
"Tell me more" is NEVER ambiguous. A pronoun with one clear referent is NEVER ambiguous.

ACTION: Store the preprocessed result as your WORKING QUERY. Every tool call argument
  in Phase 3 and the retrieval plan in Phase 2 MUST use the working query.
