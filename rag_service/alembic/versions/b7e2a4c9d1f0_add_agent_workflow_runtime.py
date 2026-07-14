"""Add agent workflow runtime tables

Revision ID: b7e2a4c9d1f0
Revises: a1f4c8d9e2b3
Create Date: 2026-07-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "b7e2a4c9d1f0"
down_revision = "a1f4c8d9e2b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_workflows",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False, server_default=""),
        sa.Column("visibility", sa.String(), nullable=False, server_default="builtin"),
        sa.Column("is_builtin", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="2"),
        sa.Column(
            "spec_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "validation_result_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_agent_workflow_builtin", "agent_workflows", ["is_builtin"], unique=False)
    op.create_index(op.f("ix_agent_workflows_name"), "agent_workflows", ["name"], unique=False)
    op.create_index(op.f("ix_agent_workflows_visibility"), "agent_workflows", ["visibility"], unique=False)

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("workflow_id", sa.String(), nullable=False),
        sa.Column(
            "run_metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "resolved_spec_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("status", sa.String(), nullable=False, server_default="running"),
        sa.Column("checkpoint_thread_id", sa.String(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "metrics_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("debug_trace_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["workflow_id"], ["agent_workflows.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_agent_run_thread_started", "agent_runs", ["thread_id", "started_at"], unique=False)
    op.create_index(op.f("ix_agent_runs_status"), "agent_runs", ["status"], unique=False)
    op.create_index(op.f("ix_agent_runs_workflow_id"), "agent_runs", ["workflow_id"], unique=False)
    op.create_index(op.f("ix_agent_runs_thread_id"), "agent_runs", ["thread_id"], unique=False)
    op.create_index(op.f("ix_agent_runs_user_id"), "agent_runs", ["user_id"], unique=False)

    op.add_column("chat_turns", sa.Column("agent_run_id", sa.String(), nullable=True))
    op.add_column("chat_turns", sa.Column("agent_run_turn_kind", sa.String(), nullable=True))
    op.add_column("chat_turns", sa.Column("agent_run_sequence", sa.Integer(), nullable=True))
    op.add_column(
        "chat_turns",
        sa.Column("agent_trace_refs_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_foreign_key(
        "fk_chat_turns_agent_run_id_agent_runs",
        "chat_turns",
        "agent_runs",
        ["agent_run_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("idx_chat_turn_agent_run_sequence", "chat_turns", ["agent_run_id", "agent_run_sequence"], unique=False)
    op.create_index(op.f("ix_chat_turns_agent_run_id"), "chat_turns", ["agent_run_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_chat_turns_agent_run_id"), table_name="chat_turns")
    op.drop_index("idx_chat_turn_agent_run_sequence", table_name="chat_turns")
    op.drop_constraint("fk_chat_turns_agent_run_id_agent_runs", "chat_turns", type_="foreignkey")
    op.drop_column("chat_turns", "agent_trace_refs_json")
    op.drop_column("chat_turns", "agent_run_sequence")
    op.drop_column("chat_turns", "agent_run_turn_kind")
    op.drop_column("chat_turns", "agent_run_id")

    op.drop_index(op.f("ix_agent_runs_user_id"), table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_thread_id"), table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_workflow_id"), table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_status"), table_name="agent_runs")
    op.drop_index("idx_agent_run_thread_started", table_name="agent_runs")
    op.drop_table("agent_runs")

    op.drop_index(op.f("ix_agent_workflows_visibility"), table_name="agent_workflows")
    op.drop_index(op.f("ix_agent_workflows_name"), table_name="agent_workflows")
    op.drop_index("idx_agent_workflow_builtin", table_name="agent_workflows")
    op.drop_table("agent_workflows")
