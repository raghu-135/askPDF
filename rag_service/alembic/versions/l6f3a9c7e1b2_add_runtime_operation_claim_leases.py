"""Add bounded leases to runtime-operation idempotency claims."""

from alembic import op
import sqlalchemy as sa


revision = "l6f3a9c7e1b2"
down_revision = "k5e2a8c4d7f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_runtime_operations",
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.add_column(
        "agent_runtime_operations",
        sa.Column("claim_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        "UPDATE agent_runtime_operations "
        "SET claim_expires_at = claimed_at + INTERVAL '5 minutes' "
        "WHERE claim_expires_at IS NULL"
    )
    op.alter_column("agent_runtime_operations", "claim_expires_at", nullable=False)


def downgrade() -> None:
    op.drop_column("agent_runtime_operations", "claim_expires_at")
    op.drop_column("agent_runtime_operations", "claimed_at")
