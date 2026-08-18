"""Seed the Corrective/Self-RAG built-in workflow.

Revision ID: e7c4a1b9d2f6
Revises: d5f1a2b3c4e6
"""

import json

from alembic import op
import sqlalchemy as sa


revision = "e7c4a1b9d2f6"
down_revision = "d5f1a2b3c4e6"
branch_labels = None
depends_on = None

BUILTIN_ID = "corrective_self_rag_agent"
BUILTIN_NAME = "Corrective/Self-RAG Agent"
BUILTIN_DESCRIPTION = "A bounded Corrective/Self-RAG workflow that grades retrieval, runs targeted corrective waves, and verifies claim-level citation support before finalization."
# Immutable v1 database snapshot. The JSON built-in remains the runtime source of truth.
SPEC_JSON = json.loads('{"config":{"allowed_tool_ids":["document_evidence","focused_document_evidence","thread_conversation_history","durable_memory","thread_events","live_web_recon","clarify_intent"],"context_policy":{"evidence_compression":"compact","evidence_dedupe":true,"evidence_packet_content_limit":2000,"evidence_packet_limit":12,"final_context_char_limit":25536,"final_prompt_assembly":"evidence_packets"},"corrective_policy":{"allow_web_fallback":true,"insufficient_evidence_mode":"verified_only","max_answer_revisions":1,"max_corrective_waves":2,"max_total_tool_attempts":12,"max_total_work_items":8,"memory_evidence_mode":"policy_scoped","minimum_relevance_confidence":0.65,"minimum_supported_claim_ratio":1.0},"custom_instructions":"","graph":{"edges":[{"from":"START","to":"context_loader"},{"from":"context_loader","to":"planner"},{"conditional":true,"from":"planner","route_fn":"planner_route","routes":{"clarify":"finalizer","direct":"direct_answer","execute":"parallel_dispatch"}},{"from":"direct_answer","to":"answer_evaluator"},{"conditional":true,"from":"answer_evaluator","route_fn":"answer_quality_route","routes":{"finalize_cautious":"finalizer","pass":"finalizer","revise":"direct_answer_reviser"}},{"from":"direct_answer_reviser","to":"answer_evaluator"},{"conditional":true,"from":"parallel_dispatch","route_fn":"parallel_dispatch_route","routes":{"dispatch":"aggregator"}},{"dynamic":true,"from":"parallel_dispatch","to":"retrieval_worker"},{"dynamic":true,"from":"parallel_dispatch","to":"thread_conversation_history_worker"},{"dynamic":true,"from":"parallel_dispatch","to":"durable_memory_worker"},{"dynamic":true,"from":"parallel_dispatch","to":"thread_events_worker"},{"dynamic":true,"from":"parallel_dispatch","to":"web_worker"},{"from":"retrieval_worker","to":"aggregator"},{"from":"thread_conversation_history_worker","to":"aggregator"},{"from":"durable_memory_worker","to":"aggregator"},{"from":"thread_events_worker","to":"aggregator"},{"from":"web_worker","to":"aggregator"},{"from":"aggregator","to":"retrieval_quality_grader"},{"conditional":true,"from":"retrieval_quality_grader","route_fn":"corrective_retrieval_route","routes":{"correct":"replanner","insufficient":"synthesizer","synthesize":"synthesizer"}},{"from":"replanner","to":"parallel_dispatch"},{"from":"synthesizer","to":"grounded_answer_verifier"},{"conditional":true,"from":"grounded_answer_verifier","route_fn":"grounded_answer_route","routes":{"correct":"replanner","finalize_cautious":"finalizer","pass":"finalizer","revise":"grounded_answer_reviser"}},{"from":"grounded_answer_reviser","to":"grounded_answer_verifier"},{"from":"finalizer","to":"END"}],"nodes":[{"id":"context_loader","type":"context_loader"},{"id":"planner","type":"planner"},{"id":"parallel_dispatch","type":"parallel_dispatch"},{"id":"retrieval_worker","type":"retrieval_worker"},{"id":"thread_conversation_history_worker","type":"thread_conversation_history_worker"},{"id":"durable_memory_worker","type":"durable_memory_worker"},{"id":"thread_events_worker","type":"thread_events_worker"},{"id":"web_worker","type":"web_worker"},{"id":"aggregator","type":"aggregator"},{"id":"retrieval_quality_grader","type":"retrieval_quality_grader"},{"id":"replanner","type":"replanner"},{"id":"direct_answer","type":"direct_answer"},{"id":"synthesizer","type":"synthesizer"},{"id":"answer_evaluator","type":"answer_evaluator"},{"id":"direct_answer_reviser","type":"answer_reviser"},{"id":"grounded_answer_reviser","type":"answer_reviser"},{"id":"grounded_answer_verifier","type":"grounded_answer_verifier"},{"id":"finalizer","type":"finalizer"}]},"hitl_policy":{"enabled":false,"gates":{}},"loop_policy":{"default_max_node_visits":1,"max_total_visits":45,"node_visit_limits":{"aggregator":3,"answer_evaluator":2,"direct_answer_reviser":1,"durable_memory_worker":3,"grounded_answer_reviser":1,"grounded_answer_verifier":4,"parallel_dispatch":3,"replanner":2,"retrieval_quality_grader":3,"retrieval_worker":3,"synthesizer":3,"thread_conversation_history_worker":3,"thread_events_worker":3,"web_worker":3}},"parallel_policy":{"continue_on_insufficient_successes":true,"continue_on_partial_failure":true,"default_worker_timeout_ms":30000,"dispatch_timeout_ms":60000,"enabled":true,"max_attempts":2,"max_concurrency":4,"max_work_items":4,"minimum_successes":1,"web_worker_timeout_ms":45000},"prefetch_policy":{"enabled":true,"mode":"routing"},"replans":2,"system_role":"","tool_instructions":{},"use_reranker":true,"use_web_search":false},"runtime":{"failure_code":"corrective_self_rag_execution_failed","failure_context":"Corrective/Self-RAG Agent execution failed gracefully.","failure_reason_prefix":"Exception during Corrective/Self-RAG Agent execution","features":{"supports_answer_quality":true,"supports_corrective_retrieval":true,"supports_parallel_dispatch":true,"supports_replans":true},"kind":"compiled_rag","label":"Corrective/Self-RAG Agent","prompt_preview":"corrective_self_rag","success_context":"Context retrieved and verified by the Corrective/Self-RAG Agent workflow."},"schema_version":2,"version":1,"workflow_id":"corrective_self_rag_agent"}')


def upgrade() -> None:
    bind = op.get_bind()
    if bind.execute(sa.text("SELECT to_regclass('agent_workflows')")).scalar() is None:
        return
    name_row = bind.execute(
        sa.text("SELECT id, is_builtin FROM agent_workflows WHERE name = :name"),
        {"name": BUILTIN_NAME},
    ).mappings().first()
    if name_row and (name_row["id"] != BUILTIN_ID or not name_row["is_builtin"]):
        raise RuntimeError("cannot seed Corrective/Self-RAG: workflow name belongs to a non-built-in workflow")

    id_row = bind.execute(
        sa.text("SELECT is_builtin FROM agent_workflows WHERE id = :id"),
        {"id": BUILTIN_ID},
    ).mappings().first()
    if id_row and not id_row["is_builtin"]:
        raise RuntimeError("cannot seed Corrective/Self-RAG: workflow id belongs to a non-built-in workflow")

    payload = {
        "id": BUILTIN_ID,
        "name": BUILTIN_NAME,
        "description": BUILTIN_DESCRIPTION,
        "visibility": "builtin",
        "is_builtin": True,
        "schema_version": 2,
        "spec_json": json.dumps(SPEC_JSON, separators=(",", ":"), sort_keys=True),
        "validation_result_json": json.dumps({"valid": True, "errors": []}),
        "metadata_json": json.dumps({"source": "builtin", "builtin_key": BUILTIN_ID, "version": 1, "version_id": f"{BUILTIN_ID}:v1"}),
    }
    if id_row:
        bind.execute(sa.text("""
            UPDATE agent_workflows
            SET name=:name, description=:description, visibility=:visibility,
                is_builtin=:is_builtin, schema_version=:schema_version,
                spec_json=CAST(:spec_json AS jsonb),
                validation_result_json=CAST(:validation_result_json AS jsonb),
                metadata_json=CAST(:metadata_json AS jsonb), updated_at=now()
            WHERE id=:id
        """), payload)
    else:
        bind.execute(sa.text("""
            INSERT INTO agent_workflows
                (id, name, description, visibility, is_builtin, schema_version,
                 spec_json, validation_result_json, metadata_json, created_at)
            VALUES
                (:id, :name, :description, :visibility, :is_builtin, :schema_version,
                 CAST(:spec_json AS jsonb), CAST(:validation_result_json AS jsonb),
                 CAST(:metadata_json AS jsonb), now())
        """), payload)


def downgrade() -> None:
    bind = op.get_bind()
    tables = bind.execute(
        sa.text("SELECT to_regclass('agent_workflows'), to_regclass('agent_runs')")
    ).one()
    if tables[0] is None:
        return
    referenced = bool(
        tables[1] is not None
        and bind.execute(
            sa.text("SELECT EXISTS (SELECT 1 FROM agent_runs WHERE workflow_id = :id)"),
            {"id": BUILTIN_ID},
        ).scalar()
    )
    if referenced:
        bind.execute(
            sa.text("UPDATE agent_workflows SET visibility='deleted', is_builtin=false, updated_at=now() WHERE id=:id"),
            {"id": BUILTIN_ID},
        )
    else:
        bind.execute(sa.text("DELETE FROM agent_workflows WHERE id=:id"), {"id": BUILTIN_ID})
