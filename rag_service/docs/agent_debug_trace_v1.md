# Agent Debug Trace v1

`debug.trace` is the backend-owned normalized debug document returned by
`GET /api/agent-runs/{id}`. It is additive: existing `debug.node_events` and
`debug.tool_events` remain available for older clients and for raw escape-hatch
debugging.

The v1 trace is inspired by OpenTelemetry spans, LangSmith runs, and
OpenInference AI span conventions, but it is not an OTLP export format.

## Goals

- Give the UI one generic trace shape for Router RAG and Plan-and-Execute RAG.
- Preserve raw runtime events while exposing stable spans, events, attributes,
  inputs, outputs, links, and artifacts.
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
- `raw`: preserved raw `node_events`, `tool_events`, and run error.

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
- `raw`: original node/tool event payload for compatibility.

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

Supported `llm.retry` attributes:

- `llm.retry.attempt`
- `llm.retry.delay_ms`
- `llm.retry.reason`
- `http.status_code`
- `exception.type`
- `exception.message`

## Size Guardrails

Generated trace `input.value`, `output.value`, `input.refs`, `output.refs`, and
tool output refs are compacted recursively. Raw escape-hatch payloads under
`debug.trace.raw` preserve the original node/tool event bodies for compatibility
and deeper debugging.

## Compatibility Rules

- `debug.trace` is preferred by new UI code.
- `debug.node_events` and `debug.tool_events` remain unchanged.
- `debug.trace.raw.node_events` and `debug.trace.raw.tool_events` preserve the
  raw events used to build the normalized trace.
- Unknown or missing `debug.trace` should fall back to legacy debug fields.
- New trace fields should be additive within `schema_version: 1`.
