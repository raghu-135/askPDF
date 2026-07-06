# Postgres Schema Notes

This schema keeps relational columns for identity, ordering, joins, and cleanup.
Flexible chat content lives in JSONB only where the fields vary per interaction.

## Current Tables

- `threads`
  - Owns chat thread identity, settings, metadata, and denormalized QA/document stats.
  - Stats fields live here directly: `total_qa_pairs`, `total_qa_chars`, `avg_qa_chars`, `last_qa_at`, `documents_meta`, `stats_last_updated_at`.

- `chat_turns`
  - One row per user interaction.
  - `payload JSONB` stores flexible turn content such as question, rewritten question, answer, reasoning, sources, clarification options, and structured errors.
  - API compatibility expands one turn into user/assistant message bubbles at the boundary.

- `agent_runs`
  - One row per agent execution, including `checkpoint_thread_id` for resumable LangGraph runs.
  - HITL pauses keep the run in `awaiting_human`; stale-running cleanup must not mark these rows failed.
  - Old terminal runs can have their LangGraph checkpoints pruned through the checkpointer API before or alongside run-retention cleanup.

- `files`
  - One row per unique content object, keyed by `file_hash`.
  - Stores global file metadata, parsing output, and status.
  - A file can be linked to multiple threads.

- `thread_files`
  - Many-to-many join between threads and files.
  - This table should stay while files can be reused across threads, forks, or embedding-model flows.
  - Live data after the chat-turn migration had shared files, so collapsing this into `files.thread_id` would lose real associations.
  - Also owns optional per-thread, per-file annotation snapshots in `annotations JSONB`.
  - `added_at` is the association creation time. `annotations_updated_at` is nullable and only meaningful when annotations are non-empty.

## Removed Tables

- `messages`
  - Replaced by `chat_turns`.

- `messages_legacy`
  - Temporary rollback table from the chat-turn migration, now dropped.

- `thread_stats`
  - Merged into `threads`.

- `thread_file_annotations`
  - Merged into `thread_files`.

## LangGraph Checkpoint Tables

When `ASKPDF_AGENT_CHECKPOINTER=postgres`, LangGraph owns its checkpoint tables
(`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, and
`checkpoint_migrations`). askPDF should clean these through the configured
checkpointer (`adelete_thread`) instead of raw SQL so LangGraph schema changes
remain isolated.

## Simplification Rules

- Prefer relational columns for identifiers, ownership, ordering, filtering, and cascade cleanup.
- Prefer JSONB for flexible payload details that are not query-critical yet.
- Do not collapse `files` and `thread_files` unless the product stops supporting file reuse across threads.
- Keep annotations on `thread_files`; do not move them onto `files` unless annotations become global per file rather than thread-specific.
