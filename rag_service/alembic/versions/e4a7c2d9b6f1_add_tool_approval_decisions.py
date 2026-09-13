"""Persist framework-neutral human tool permissions."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "e4a7c2d9b6f1"
down_revision = "d6f2a8c4e1b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_approval_decisions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("interrupt_id", sa.String(), nullable=False),
        sa.Column("tool_name", sa.String(), nullable=False),
        sa.Column("invocation_id", sa.String(), nullable=False),
        sa.Column("argument_hash", sa.String(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("decision", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "interrupt_id", name="uq_tool_approval_interrupt"),
    )
    op.create_index("ix_tool_approval_scope", "tool_approval_decisions", ["scope", "scope_id", "tool_name", "created_at"])
    op.create_table(
        "tool_invocations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("invocation_id", sa.String(), nullable=False),
        sa.Column("tool_name", sa.String(), nullable=False),
        sa.Column("argument_hash", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("result_json", JSONB(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("run_id", "invocation_id", name="uq_tool_invocation"),
    )


def downgrade() -> None:
    op.drop_table("tool_invocations")
    op.drop_table("tool_approval_decisions")
