"""Add the product applied-runtime-delta ledger."""

from alembic import op
import sqlalchemy as sa


revision = "b2d8e4f6a1c3"
down_revision = "a9c7e1f3b5d2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_task_runtime_deltas",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column("agent_run_id", sa.String(), nullable=False),
        sa.Column("attempt_id", sa.String(), nullable=False),
        sa.Column("operation_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("payload_sha256", sa.String(), nullable=False),
        sa.Column("observed_task_version", sa.Integer(), nullable=False),
        sa.Column("observed_plan_revision", sa.Integer(), nullable=False),
        sa.Column("applied_task_version", sa.Integer(), nullable=False),
        sa.Column("applied_plan_revision", sa.Integer(), nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "observed_task_version >= 0 and observed_plan_revision >= 0 and "
            "applied_task_version >= 1 and applied_plan_revision >= 0",
            name="ck_agent_task_runtime_delta_versions",
        ),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["task_id"], ["agent_tasks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("agent_run_id", "event_id", name="uq_agent_task_runtime_delta_event"),
        sa.UniqueConstraint("task_id", "idempotency_key", name="uq_agent_task_runtime_delta_idempotency"),
    )
    op.create_index("ix_agent_task_runtime_deltas_task_id", "agent_task_runtime_deltas", ["task_id"])
    op.create_index("ix_agent_task_runtime_deltas_agent_run_id", "agent_task_runtime_deltas", ["agent_run_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_task_runtime_deltas_agent_run_id", table_name="agent_task_runtime_deltas")
    op.drop_index("ix_agent_task_runtime_deltas_task_id", table_name="agent_task_runtime_deltas")
    op.drop_table("agent_task_runtime_deltas")
