# Data model

## Ownership model

PostgreSQL is the authoritative product store. Weaviate and runtime databases
are derived or service-owned stores.

## Core product entities

- Project: name, description, embedding model, settings, and activity.
- Thread: project membership, immutable embedding model, chat settings,
  metadata, and denormalized statistics.
- File: content-addressed object metadata, processing state, and parsed output.
- ThreadFile: thread/file association and thread-specific annotations.
- ProjectFile: project-level knowledge association.
- ChatTurn: one persisted interaction whose variable content is stored in JSONB.
- AgentTaskArtifact: durable content-addressed objects, including task reports
  and chat-native research canvases (`research_canvas`,
  `application/vnd.askpdf.canvas+json`). Chat-only canvases are thread-owned
  and do not require a deep-research task. Agent publishes use the
  `publish_canvas` tool and may store the emitting `agent_run_id` so the
  current ChatTurn can surface an Open canvas card.
  Compare-papers, evidence-matrix, and timeline canvases are prompt skills
  over the same spec, injected only when publish is admitted.

Files and associations remain separate because one file can be reused across
threads and projects. Annotations are association-specific, not globally
attached to the file.

## Agent entities

- AgentWorkflow: executable definition, framework, builder, version, and spec.
- AgentRun: one frozen workflow execution and opaque runtime binding.
- AgentRunEvent: canonical framework-neutral event journal.
- ToolInvocation: invocation identity, argument hash, status, and result.
- ToolApprovalDecision: durable approval bound to a run, tool, invocation, and scope.
- AgentTask: durable user-facing long-running task.
- Task plan, todo, subagent, artifact, event, command, and runtime-delta records.

The SQLModel source is rag_service/app/db/models_sqlmodel.py. Product migrations
are under rag_service/alembic/versions/.

## Memory and vectors

Memory records support user, project, and thread scopes, review state,
overrides, indexing state, and source references. Effective memories are
materialized into model-specific Weaviate collections.

Document, chat, web-search, and memory vectors are separated by embedding model
to prevent dimension/model mismatches.

## Runtime persistence

LangGraph checkpoint tables and runtime execution journals belong to
langgraph-runtime. The control plane stores only opaque runtime bindings and
product projections. It must not query or mutate runtime tables directly.

## Migration rules

- Add an Alembic migration for every database schema or persisted-data change.
- Keep relational columns for identity, ownership, ordering, joins, and cleanup.
- Use JSONB for flexible payloads that are not query-critical.
- Preserve cascade and restrictive foreign-key behavior.
- Keep product and runtime migration graphs independent.

The historical schema notes are preserved in the implementation model and
migration history; this page is the current conceptual contract.
