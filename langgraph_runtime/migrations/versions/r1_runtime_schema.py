"""Create the complete runtime execution schema."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "r1_runtime_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runtime_executions",
        sa.Column("run_id", sa.Text(), primary_key=True),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column("request", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("next_sequence", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("continuation", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("owner_id", sa.Text(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("request_fingerprint", sa.Text(), nullable=True),
        sa.Column("last_operation_id", sa.Text(), nullable=True),
        sa.Column("retry_source_attempt", sa.Integer(), nullable=True),
    )
    op.create_table(
        "runtime_operations",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("operation_id", sa.Text(), nullable=False),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column("request_fingerprint", sa.Text(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="queued"),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["run_id"], ["runtime_executions.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "operation_id"),
    )
    op.create_table(
        "runtime_events",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("occurred_at", sa.Text(), nullable=True),
        sa.Column("trace_id", sa.Text(), nullable=True),
        sa.Column("continuation", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("terminal", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["run_id"], ["runtime_executions.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "sequence"),
        sa.UniqueConstraint("run_id", "event_id"),
    )


def downgrade() -> None:
    op.drop_table("runtime_events")
    op.drop_table("runtime_operations")
    op.drop_table("runtime_executions")
