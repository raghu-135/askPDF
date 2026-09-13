"""Add the consolidated agent runtime application schema.

Revision ID: a9c7e1f3b5d2
Revises: a8d3f1c6e4b2
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "a9c7e1f3b5d2"
down_revision = "a8d3f1c6e4b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_workflows", sa.Column("framework", sa.String(), nullable=False, server_default="langgraph"))
    op.add_column("agent_workflows", sa.Column("builder_id", sa.String(), nullable=False, server_default="langgraph_graph"))
    op.add_column("agent_workflows", sa.Column("category", sa.String(), nullable=True))
    op.create_index("ix_agent_workflows_framework", "agent_workflows", ["framework"])
    op.create_index("ix_agent_workflows_builder_id", "agent_workflows", ["builder_id"])
    op.create_index("ix_agent_workflows_category", "agent_workflows", ["category"])

    op.add_column("agent_runs", sa.Column("framework", sa.String(), nullable=False, server_default="langgraph"))
    op.add_column("agent_runs", sa.Column("builder_id", sa.String(), nullable=False, server_default="langgraph_graph"))
    op.add_column("agent_runs", sa.Column("definition_category", sa.String(), nullable=True))
    op.add_column(
        "agent_runs",
        sa.Column("runtime_binding_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column("agent_runs", sa.Column("runtime_binding_version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("agent_runs", sa.Column("runtime_binding_status", sa.String(), nullable=False, server_default="active"))
    op.create_index("ix_agent_runs_framework", "agent_runs", ["framework"])
    op.create_index("ix_agent_runs_builder_id", "agent_runs", ["builder_id"])
    op.create_index("ix_agent_runs_definition_category", "agent_runs", ["definition_category"])

    op.execute(
        """
        UPDATE agent_workflows
           SET framework = COALESCE(NULLIF(metadata_json->>'framework', ''), 'langgraph'),
               builder_id = COALESCE(NULLIF(metadata_json->>'builder_id', ''), 'langgraph_graph'),
               category = COALESCE(NULLIF(metadata_json->>'category', ''), CASE id
                   WHEN 'router_rag_agent' THEN 'router'
                   WHEN 'plan_execute_rag_agent' THEN 'replanner'
                   WHEN 'evaluator_replanner_rag_agent' THEN 'replanner'
                   WHEN 'orchestrator_worker_rag_agent' THEN 'replanner'
                   WHEN 'corrective_self_rag_agent' THEN 'replanner'
                   WHEN 'deep_research_agent' THEN 'deep'
                   ELSE NULL
               END)
        """
    )
    op.execute(
        """
        UPDATE agent_runs AS runs
           SET framework = workflows.framework,
               builder_id = workflows.builder_id,
               definition_category = workflows.category,
               runtime_binding_json = CASE
                   WHEN runs.checkpoint_thread_id IS NOT NULL THEN jsonb_build_object(
                       'binding_type', 'langgraph_checkpoint',
                       'binding_version', 1,
                       'payload', jsonb_build_object('checkpoint_thread_id', runs.checkpoint_thread_id)
                   )
                   ELSE '{}'::jsonb
               END,
               runtime_binding_version = 1,
               runtime_binding_status = 'active'
          FROM agent_workflows AS workflows
         WHERE runs.workflow_id = workflows.id
        """
    )

    op.create_table(
        "agent_run_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_run_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("trace_id", sa.String(), nullable=True),
        sa.Column("terminal", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("source_metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("attempt >= 1", name="ck_agent_run_events_attempt"),
        sa.CheckConstraint("sequence >= 0", name="ck_agent_run_events_sequence"),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("agent_run_id", "event_id", name="uq_agent_run_events_run_event"),
    )
    for name, column in (
        ("ix_agent_run_events_agent_run_id", "agent_run_id"),
        ("ix_agent_run_events_event_id", "event_id"),
        ("ix_agent_run_events_kind", "kind"),
        ("ix_agent_run_events_trace_id", "trace_id"),
    ):
        op.create_index(name, "agent_run_events", [column])
    op.create_index("idx_agent_run_events_run_sequence", "agent_run_events", ["agent_run_id", "attempt", "sequence"])

    op.add_column("agent_task_events", sa.Column("event_id", sa.String(), nullable=True))
    op.add_column("agent_task_events", sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_task_events", sa.Column("terminal", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("agent_task_events", sa.Column("source_metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")))
    op.create_index("ix_agent_task_events_event_id", "agent_task_events", ["event_id"])

    op.drop_constraint("ck_agent_task_commands_action", "agent_task_commands", type_="check")
    op.create_check_constraint(
        "ck_agent_task_commands_action",
        "agent_task_commands",
        "action in ('start','pause','resume','cancel','retry','expire','delete','steer')",
    )

    op.create_table(
        "agent_runtime_operations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("operation", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("request_fingerprint", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="in_progress"),
        sa.Column("result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("error_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("claim_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status in ('in_progress', 'completed', 'failed')", name="ck_agent_runtime_operations_status"),
        sa.CheckConstraint("length(btrim(idempotency_key)) > 0", name="ck_agent_runtime_operations_key_nonempty"),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "operation", "idempotency_key", name="uq_agent_runtime_operation_idempotency"),
    )
    op.create_index("ix_agent_runtime_operations_run_id", "agent_runtime_operations", ["run_id"])
    op.create_index("ix_agent_runtime_operations_operation", "agent_runtime_operations", ["operation"])
    op.create_index("ix_agent_runtime_operations_status", "agent_runtime_operations", ["status"])
    op.create_index("idx_agent_runtime_operations_run_operation", "agent_runtime_operations", ["run_id", "operation"])
    op.execute("UPDATE agent_runtime_operations SET claim_expires_at = claimed_at + INTERVAL '5 minutes'")


def downgrade() -> None:
    op.drop_index("idx_agent_runtime_operations_run_operation", table_name="agent_runtime_operations")
    op.drop_index("ix_agent_runtime_operations_status", table_name="agent_runtime_operations")
    op.drop_index("ix_agent_runtime_operations_operation", table_name="agent_runtime_operations")
    op.drop_index("ix_agent_runtime_operations_run_id", table_name="agent_runtime_operations")
    op.drop_table("agent_runtime_operations")
    op.drop_constraint("ck_agent_task_commands_action", "agent_task_commands", type_="check")
    op.create_check_constraint(
        "ck_agent_task_commands_action",
        "agent_task_commands",
        "action in ('start','pause','resume','cancel','retry','expire','delete')",
    )
    op.drop_index("ix_agent_task_events_event_id", table_name="agent_task_events")
    for column in ("source_metadata_json", "terminal", "occurred_at", "event_id"):
        op.drop_column("agent_task_events", column)
    for name in (
        "idx_agent_run_events_run_sequence",
        "ix_agent_run_events_trace_id",
        "ix_agent_run_events_kind",
        "ix_agent_run_events_event_id",
        "ix_agent_run_events_agent_run_id",
    ):
        op.drop_index(name, table_name="agent_run_events")
    op.drop_table("agent_run_events")
    for name in ("ix_agent_runs_definition_category", "ix_agent_runs_builder_id", "ix_agent_runs_framework"):
        op.drop_index(name, table_name="agent_runs")
    for column in ("runtime_binding_status", "runtime_binding_version", "runtime_binding_json", "definition_category", "builder_id", "framework"):
        op.drop_column("agent_runs", column)
    for name in ("ix_agent_workflows_category", "ix_agent_workflows_builder_id", "ix_agent_workflows_framework"):
        op.drop_index(name, table_name="agent_workflows")
    for column in ("category", "builder_id", "framework"):
        op.drop_column("agent_workflows", column)
