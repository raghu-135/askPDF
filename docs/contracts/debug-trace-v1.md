# Agent debug trace v1

Agent runs persist a backend-owned debug payload with version, trace, and
summary. GET /api/agent-runs/{id} also returns an API-derived graph view model.
The UI should render normalized fields and use span.raw only for escape-hatch
inspection.

The machine-readable schema is
[agent_debug_trace_v1.schema.json](agent_debug_trace_v1.schema.json).

The trace is inspired by OpenTelemetry and OpenInference conventions but is not
an OTLP export format.

## Goals

- Provide one trace shape for all supported workflows.
- Expose stable spans, events, attributes, inputs, outputs, links, and artifacts.
- Preserve bounded previews and references instead of full source bodies.
- Represent skipped nodes as skipped work.
- Surface retry and human-interrupt telemetry.

## Top-level contract

The trace document contains:

- schema_version
- trace_id and run_id
- thread_id, chat_turn_id, and user_id where available
- workflow_id and workflow_type
- status and timing
- attributes and metrics
- spans, links, and artifacts

## Span contract

Spans contain stable IDs, parent relationships, names, kinds, status, timing,
attributes, bounded input/output, events, links, and optional raw runtime data.
Kinds include AGENT, CHAIN, LLM, RETRIEVER, TOOL, and PROMPT.

Common events include decision.made, prompt.rendered, llm.completed,
llm.retry, tool.called, tool.completed, warning, skipped, exception,
checkpoint.created, interrupt.requested, resume.requested, graph.resumed,
interrupt.rejected, and interrupt.expired.

Skipped work is not a warning. Human-interrupt events remain attached to the
root run span so a paused and resumed run retains one trace.

## Retrieval and privacy

Retrieval references may include file hash, chunk, page, score, message,
timeline, URL, title, and bounded preview metadata. Full source bodies and
resume tokens must not be stored in the normalized trace.

The schema is additive within version 1. Update the JSON schema and trace
projection tests together when adding fields.
