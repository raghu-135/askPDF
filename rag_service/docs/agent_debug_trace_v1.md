# Agent Debug Trace v1

`debug.trace` is the backend-owned normalized debug document returned by
`GET /api/agent-runs/{id}`. The debug UI renders this normalized trace directly.
Each span preserves the original runtime event for that unit of work under
`span.raw` for escape-hatch debugging and trace reconstruction.

The v1 trace is inspired by OpenTelemetry spans, LangSmith runs, and
OpenInference AI span conventions, but it is not an OTLP export format.

The machine-readable contract lives next to this document in
`agent_debug_trace_v1.schema.json`. The schema is intentionally permissive about
additional properties so v1 can stay additive while preserving a stable required
shape for consumers.

## Goals

- Give the UI one generic trace shape for Router RAG and Plan-and-Execute RAG.
- Expose stable spans, events, attributes, inputs, outputs, links, and
  artifacts for UI/debug consumers.
- Preserve per-span raw runtime payloads only as an escape hatch for deeper
  debugging.
- Represent skipped graph nodes as skipped work, not warnings.
- Store refs and bounded previews instead of full source bodies.
- Surface retry telemetry without changing retry behavior.

## Trace Document

Top-level fields:

- `schema_version`: currently `1`.
- `trace_id`: stable trace identifier. V1 uses the agent run ID.
- `run_id`, `thread_id`, `chat_turn_id`, `user_id`: owning entities.
- `template_id`, `template_version_id`, `pattern_type`: selected agent pattern.
- `status`: run status.
- `started_at`, `completed_at`, `duration_ms`: run timing when available.
- `attributes`: root run attributes duplicated from the root span for quick access.
- `metrics`: persisted run metrics.
- `spans`: root, node, and tool spans.
- `links`: flattened span links to refs/artifacts.
- `artifacts`: flattened artifact refs derived from span outputs.

## Span Shape

Every span uses this neutral shape:

- `span_id`: stable within the trace, for example `run:{run_id}`,
  `node:{node_id}:{index}`, or `tool:{tool_name}:{index}`.
- `parent_span_id`: root span parent is `null`; node spans parent to the root;
  tool spans parent to their caller node when known.
- `name`: human-readable span name.
- `kind`: one of `AGENT`, `CHAIN`, `LLM`, `RETRIEVER`, `TOOL`, or `PROMPT`
  where practical. V1 node/tool spans currently use `AGENT`, `CHAIN`,
  `RETRIEVER`, and `TOOL`.
- `status`: `completed`, `skipped`, `error`, or the run status for the root span.
- `start_time`, `end_time`, `duration_ms`: optional timing. New runtime events
  include node/tool wall-clock timestamps when elapsed timing is available; older
  runs may only have `duration_ms`.
- `attributes`: standard-ish and askPDF namespaced metadata.
- `input`: bounded input preview/refs.
- `output`: bounded output preview/refs/summary.
- `events`: decisions, prompts, tool lifecycle events, warnings, skips, and
  exceptions.
- `links`: refs extracted from output refs.
- `raw`: original node/tool event payload for deeper debugging. Consumers should
  prefer normalized span fields for rendering and automation.

## Span Kinds

- Root run span: `AGENT`.
- Router/planner node spans: `AGENT`.
- Retrieval worker spans: `RETRIEVER`.
- Other graph node spans: `CHAIN`.
- First-party tool spans:
  - `RETRIEVER` for document, memory, timeline, and web categories.
  - `TOOL` for other categories.

## Common Attributes

Root span attributes:

- `session.id`: thread ID.
- `user.id`: user ID when available.
- `askpdf.run.id`
- `askpdf.thread.id`
- `askpdf.chat_turn.id`
- `askpdf.template.id`
- `askpdf.template_version.id`
- `askpdf.pattern_type`
- `askpdf.route`
- `askpdf.route_reason`
- `askpdf.use_web_search`
- `askpdf.use_reranker`
- `askpdf.context_window`
- `askpdf.warning_count`
- `askpdf.error_count`

Node span attributes:

- `askpdf.node.id`
- `askpdf.node.name`
- `askpdf.route`
- `askpdf.route_reason`
- `askpdf.skip_reason`
- `askpdf.execution_plan`
- `askpdf.evidence_chars`
- `askpdf.answer_chars`
- `askpdf.document_source_count`
- `askpdf.web_source_count`
- `askpdf.used_chat_id_count`
- `askpdf.timeline_event_count`

Tool span attributes:

- `tool.name`
- `tool.id`
- `tool.description`
- `askpdf.tool.category`
- `askpdf.caller_node`
- `askpdf.result_chars`
- `askpdf.source_count`
- `askpdf.artifact_keys`
- `askpdf.known_warning_codes`

## Span Events

Generic event names:

- `decision.made`: route, route reason, and execution plan from router/planner
  nodes.
- `prompt.rendered`: prompt section/name, prompt character count, system message
  preview, and prompt preview.
- `normalization.applied`: planner normalization/clamp notes.
- `llm.completed`: model name, response character count, token counts, and
  reasoning availability when provider metadata exposes them.
- `llm.retry`: retry attempt number, delay, reason, HTTP status code, and compact
  exception details for retryable LLM/provider failures.
- `tool.called`: tool name, contract ID, and category.
- `tool.completed`: result character count, source count, and warning count.
- `warning`: real runtime/tool warning codes.
- `skipped`: skipped node status and `askpdf.skip_reason`.
- `exception`: failed runtime/tool information.

Skips are not warnings. A skipped Plan-and-Execute worker should have
`status: "skipped"` and a `skipped` span event.

## Retrieval Data

Retrieval and tool refs belong in span `input.refs`, `output.refs`, `links`, and
`artifacts`. V1 supports refs for:

- Document matches: file hash, file name, chunk ID, pages, score, rerank score,
  temporal metadata, and bounded preview.
- Memory matches: message ID, turn ID, role, created time, score/rerank score,
  and bounded preview.
- Timeline events: source type, event time/type, message/file/url refs.
- Web sources: URL, title, search query, score, searched timestamp.

The trace should not store full source bodies.

## LLM Usage Data

LLM usage fields are optional. They are captured only when the provider response
or LangChain message exposes them.

Supported `llm.completed` attributes:

- `llm.model_name`
- `llm.response_chars`
- `llm.token_count.prompt`
- `llm.token_count.completion`
- `llm.token_count.total`
- `llm.token_count.reasoning`
- `llm.token_count.cached`
- `llm.reasoning_available`
- `llm.reasoning_format`
- `llm.reasoning_chars`
- `llm.retry_count`

The `llm.completed` event may also include `output.reasoning_preview`, a bounded
preview of provider-supplied reasoning text when the model/server exposes it.
The full reasoning body is not stored in the normalized trace.

Trace `metrics` also includes aggregate LLM usage when available:

- `llm_span_count`
- `llm_token_count_prompt`
- `llm_token_count_completion`
- `llm_token_count_total`
- `llm_token_count_reasoning`
- `llm_token_count_cached`
- `llm_retry_count`

Supported `llm.retry` attributes:

- `llm.retry.attempt`
- `llm.retry.delay_ms`
- `llm.retry.reason`
- `http.status_code`
- `exception.type`
- `exception.message`

## Size Guardrails

Generated trace `input.value`, `output.value`, `input.refs`, `output.refs`, and
tool output refs are compacted recursively. Span-level `raw` payloads preserve
the original runtime event for deeper debugging.

## Evolution Rules

- `debug.trace` is the public debug rendering contract.
- Each span may include `raw`, the original runtime event used to build that
  normalized span.
- New trace fields should be additive within `schema_version: 1`.
