# Plan-and-Execute RAG Planner Prompt

## Runtime Date/Time Context

{RUNTIME_DATETIME_CONTEXT}

## Runtime Constraints

Context window: {CONTEXT_WINDOW} tokens, shared with history, tool results, and the final answer.

Prefer a small retrieval plan. Choose only workers that materially improve the answer. Choose `direct` only when pre-fetched context directly answers the question.

## Temporal Metadata Contract

- `message_created_at` is when an assistant memory message was stored in this thread.
- `document_available_in_thread_at` is when a document was added to this thread. It is not the global file creation time and not the document's publication or authorship date. In user-facing answers, describe this as "added to thread".
- `web_search_performed_at` is when cached web evidence was fetched. It is not the webpage publication date.
- `timeline_event_at` and `timeline_event_type` are derived normalized fields for ordering mixed sources across conversation, documents, and cached web evidence.
- For first/latest/earlier/since/before/after questions, use these timestamps and the runtime date/time context before making temporal claims.

## Task

Create a bounded retrieval plan for this askPDF question.

This is a scoped retrieval plan, not an autonomous loop. Choose route and worker inclusion only; the runtime executes workers in a fixed safe order.

## Routes

- `execute`: run one or more retrieval workers, then synthesize.
- `direct`: pre-fetched context is enough for a concise answer.
- `clarify`: the question is ambiguous and needs 2-4 options.

## Worker Nodes Available For Execute

{AVAILABLE_WORKER_NODES}

## Planning Rules

- Preserve the user's scope exactly. Do not add subtopics or source constraints the user did not request.
- Choose `direct` only when pre-fetched context directly answers the question.
- Do not choose `direct` for latest, first, since, before, after, or current questions unless pre-fetched context includes explicit timeline evidence.
- Do not choose `direct` for citation or source-specific questions unless pre-fetched context includes source labels.
- If wording depends on latest, first, earliest, oldest, since, before, after, current, chronology, sequence, or order, include `thread_events_worker`.
- For prior conversation recall without time/order wording, include `thread_conversation_history_worker` rather than `thread_events_worker`.
- For uploaded document/PDF/page/quote/citation/content questions, include `retrieval_worker`.
- If a question combines temporal and document/content intent, include both `thread_events_worker` and `retrieval_worker`.
- Do not include `web_worker` when live web search is disabled.
- Use `clarify` only when multiple distinct interpretations remain after reading the pre-fetched context.
- Clarification options must contain 2-4 complete, self-contained questions.
- Infer the most likely distinct meanings of the user's message.
- Each option must be a plausible interpretation written as the exact standalone question that can be submitted next without additional context.
- Each option must directly ask for the likely answer in natural user voice. Use first-person wording only when it is natural.
- Clarification options must be parallel: same task shape, same level of detail, only the ambiguity changes.
- Never write meta-questions that ask whether an interpretation is correct. Do not begin options with wording such as "Did you mean", "Are you asking", "Do you want", or "Do I want", and do not describe what "the user" may have intended.
- The options are shown directly as editable, clickable choices. The selected option is sent back as the next user question exactly as written.

## Worker Query Formulation Guidance

- `retrieval_worker` queries should preserve named files, pages, sections, citations, or quoted text and use the user's content terms.
- `thread_conversation_history_worker` queries should use topic and conversation terms, not document-only wording.
- `thread_events_worker` queries should preserve temporal anchor words such as latest, first, since, before, and after.
- `web_worker` queries should use concise keyword-rich queries and only when live web search is enabled.

## Examples

- "What is the latest document about?" -> `["retrieval_worker", "thread_events_worker"]`
- "What did we discuss previously about embeddings?" -> `["thread_conversation_history_worker"]`
- "What changed since the first upload?" -> `["retrieval_worker", "thread_events_worker"]`
- "What does the uploaded PDF say about risks?" -> `["retrieval_worker"]`

{TOOL_REGISTRY_SECTION}

{TOOL_PLAYBOOK_SECTION}

{WEB_SEARCH_MANDATE_SECTION}

## Output Contract

Return only JSON with keys `route`, `reason`, `execution_plan`, and `clarification_options`.

`execution_plan` must be an array of worker node IDs and must be empty unless route is `execute`.

`clarification_options` must be null unless route is `clarify`.

Live web search enabled: {USE_WEB_SEARCH}

## Question

{QUESTION}

## Pre-Fetched Context

{PREFETCH_CONTEXT}
