---
PHASE 0 - PREPROCESS  (internal only - no output visible to user)
---
No upstream query rewriter ran for this request. Do the following internally:

1) Resolve pronouns and vague references to explicit entities from recent context.
2) Make a standalone working query with minimal extra context (no scope changes).
3) If pre-fetch is clearly sufficient, skip retrieval (except search_web if enabled).
4) If ambiguity remains with multiple distinct interpretations, call ask_for_clarification.

Use the working query for any tool calls and for the plan.
