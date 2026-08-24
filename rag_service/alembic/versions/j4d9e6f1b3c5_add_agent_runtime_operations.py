"""Add product-owned runtime operation idempotency records."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "j4d9e6f1b3c5"
down_revision = "f1a9c7e3b5d2"
branch_labels = None
depends_on = None


def upgrade() -> None:
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
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("status in ('in_progress', 'completed', 'failed')", name="ck_agent_runtime_operations_status"),
        sa.CheckConstraint("length(btrim(idempotency_key)) > 0", name="ck_agent_runtime_operations_key_nonempty"),
        sa.UniqueConstraint("run_id", "operation", "idempotency_key", name="uq_agent_runtime_operation_idempotency"),
    )
    op.create_index("ix_agent_runtime_operations_run_id", "agent_runtime_operations", ["run_id"])
    op.create_index("ix_agent_runtime_operations_operation", "agent_runtime_operations", ["operation"])
    op.create_index("ix_agent_runtime_operations_status", "agent_runtime_operations", ["status"])
    op.create_index("idx_agent_runtime_operations_run_operation", "agent_runtime_operations", ["run_id", "operation"])


def downgrade() -> None:
    op.drop_index("idx_agent_runtime_operations_run_operation", table_name="agent_runtime_operations")
    op.drop_index("ix_agent_runtime_operations_status", table_name="agent_runtime_operations")
    op.drop_index("ix_agent_runtime_operations_operation", table_name="agent_runtime_operations")
    op.drop_index("ix_agent_runtime_operations_run_id", table_name="agent_runtime_operations")
    op.drop_table("agent_runtime_operations")
