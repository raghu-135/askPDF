# Research canvases

A research canvas is a durable, chat-native workspace document. The agent (or
the product API) publishes a typed JSON spec; the frontend renders it with
first-party React. There is no TSX compiler and no new graph runtime for
compare-papers, evidence-matrix, or timeline layouts.

## Product shape

The document is `canvas_spec_v1`: a title, optional summary, and sections of
typed blocks. Allowed blocks are `stat`, `table`, `callout`, `markdown`,
`sources`, and `dag`. Unknown types, scripts, and javascript URLs are rejected.

Citations live in a `sources` block:

- `document`: `file_hash` (must be attached to the thread), optional `sentence_id`
- `web`: http(s) `url`
- `memory`: `memory_id`
- `conversation`: `message_id`

Agent publish requires at least one citation. Document hashes that are not on
the thread fail admission.

The Pydantic contract is rag_service/app/models/canvas.py. The renderer and
client types are frontend/src/components/canvas/CanvasDocument.tsx and
frontend/src/lib/canvas-spec.ts.

## Persistence

Canvases are `research_canvas` artifacts
(`application/vnd.askpdf.canvas+json`) in the shared content store, not a
separate blob table. Chat-only publishes are thread-owned and do not need an
AgentTask. Agent publishes may store `agent_run_id` so the ChatTurn can show
an Open canvas card.

Revisions use `supersedes_id`. List defaults to current (non-superseded)
documents. Idempotent creates use `idempotency_key` (agent publishes hash the
spec when the caller omits one).

The introducing migration is `c7f2a9d4e1b8_store_research_canvases_as_artifacts`.
Implementation: rag_service/app/services/canvas_service.py and
rag_service/app/api/canvases.py.

## Workbench and chat

Each thread workspace includes a Research canvas tab. Chat messages that
own a canvas expose `canvas_ref` (`id`, `title`); the Open canvas control
opens that tab. Document citations in the canvas jump to the PDF highlight
when the file is on the thread.

## Agent emit

The MCP tool is `publish_canvas` (contract id `research_canvas_publish`).
Admission and persist happen in rag_service/app/tools/publish_canvas.py.

Layout skills (`canvas_compare_papers`, `canvas_evidence_matrix`,
`canvas_timeline`) are prompt recipes over the same blocks. They are injected
only when that tool is in `allowed_tool_ids`. They are not separate workflows.

### LangGraph

Evidence workers stay query-only. Answer and synthesis nodes bind
`publish_canvas` when the workflow admits it: `synthesizer`, `direct_answer`,
`answer_reviser`, and `deep_task_synthesizer`. The runtime lifts flattened
block-shaped arguments into `canvas_spec_v1` before MCP validation and strips
prose `publish_canvas(...)` fakes from the visible answer.

Implementation: langgraph_runtime/workflows/canvas_publish.py.

### Hermes

Hermes builtins list `publish_canvas` as an MCP tool. At definition resolve,
layout skill ids and the skill markdown are appended to instructions when the
tool is admitted; they are stripped when it is not. Hermes `config.skills`
remains opaque ids; the recipe text lives in the system prompt.

## Human approval

Thread setting `hitl_canvas_publish` (default off) compiles an ASK policy for
`publish_canvas` at the MCP boundary, the same mechanism as other human-gated
tools. See [Tool approval](security/tool-approval.md).
