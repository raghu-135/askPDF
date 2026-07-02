"""Add agent pattern runtime tables

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
        "agent_pattern_templates",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False, server_default=""),
        sa.Column("visibility", sa.String(), nullable=False, server_default="builtin"),
        sa.Column("owner_id", sa.String(), nullable=True),
        sa.Column("current_version_id", sa.String(), nullable=True),
        sa.Column("is_builtin", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_agent_pattern_template_builtin", "agent_pattern_templates", ["is_builtin"], unique=False)
    op.create_index(op.f("ix_agent_pattern_templates_current_version_id"), "agent_pattern_templates", ["current_version_id"], unique=False)
    op.create_index(op.f("ix_agent_pattern_templates_name"), "agent_pattern_templates", ["name"], unique=False)
    op.create_index(op.f("ix_agent_pattern_templates_owner_id"), "agent_pattern_templates", ["owner_id"], unique=False)
    op.create_index(op.f("ix_agent_pattern_templates_visibility"), "agent_pattern_templates", ["visibility"], unique=False)

    op.create_table(
        "agent_pattern_template_versions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("template_id", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
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
        sa.Column("changelog", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["template_id"], ["agent_pattern_templates.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_agent_pattern_template_version_unique", "agent_pattern_template_versions", ["template_id", "version"], unique=True)
    op.create_index(op.f("ix_agent_pattern_template_versions_template_id"), "agent_pattern_template_versions", ["template_id"], unique=False)
    op.create_index(op.f("ix_agent_pattern_template_versions_version"), "agent_pattern_template_versions", ["version"], unique=False)

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("template_id", sa.String(), nullable=False),
        sa.Column("template_version_id", sa.String(), nullable=False),
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
        sa.ForeignKeyConstraint(["template_id"], ["agent_pattern_templates.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["template_version_id"], ["agent_pattern_template_versions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_agent_run_thread_started", "agent_runs", ["thread_id", "started_at"], unique=False)
    op.create_index(op.f("ix_agent_runs_status"), "agent_runs", ["status"], unique=False)
    op.create_index(op.f("ix_agent_runs_template_id"), "agent_runs", ["template_id"], unique=False)
    op.create_index(op.f("ix_agent_runs_template_version_id"), "agent_runs", ["template_version_id"], unique=False)
    op.create_index(op.f("ix_agent_runs_thread_id"), "agent_runs", ["thread_id"], unique=False)
    op.create_index(op.f("ix_agent_runs_user_id"), "agent_runs", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_runs_user_id"), table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_thread_id"), table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_template_version_id"), table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_template_id"), table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_status"), table_name="agent_runs")
    op.drop_index("idx_agent_run_thread_started", table_name="agent_runs")
    op.drop_table("agent_runs")

    op.drop_index(op.f("ix_agent_pattern_template_versions_version"), table_name="agent_pattern_template_versions")
    op.drop_index(op.f("ix_agent_pattern_template_versions_template_id"), table_name="agent_pattern_template_versions")
    op.drop_index("idx_agent_pattern_template_version_unique", table_name="agent_pattern_template_versions")
    op.drop_table("agent_pattern_template_versions")

    op.drop_index(op.f("ix_agent_pattern_templates_visibility"), table_name="agent_pattern_templates")
    op.drop_index(op.f("ix_agent_pattern_templates_owner_id"), table_name="agent_pattern_templates")
    op.drop_index(op.f("ix_agent_pattern_templates_name"), table_name="agent_pattern_templates")
    op.drop_index(op.f("ix_agent_pattern_templates_current_version_id"), table_name="agent_pattern_templates")
    op.drop_index("idx_agent_pattern_template_builtin", table_name="agent_pattern_templates")
    op.drop_table("agent_pattern_templates")
