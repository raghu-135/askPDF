"""Add semantic memory timestamps and durable memory-manager idempotency."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "c4e8a1b6d2f0"
down_revision = "7f3c1a9d5e2b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column(
            "semantic_updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.execute(
        "UPDATE memories SET semantic_updated_at = COALESCE(updated_at, created_at, now()) "
        "WHERE semantic_updated_at IS NULL"
    )
    op.alter_column("memories", "semantic_updated_at", nullable=False)
    op.create_index("idx_memory_semantic_updated_at", "memories", ["semantic_updated_at"], unique=False)

    op.create_table(
        "memory_manager_idempotency",
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("plan_hash", sa.String(), nullable=False),
        sa.Column("status", sa.String(), server_default="in_progress", nullable=False),
        sa.Column(
            "result_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("actor_id", sa.String(), server_default="ui", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "length(btrim(idempotency_key)) > 0",
            name="ck_memory_manager_idempotency_key_nonempty",
        ),
        sa.CheckConstraint(
            "length(btrim(plan_hash)) > 0",
            name="ck_memory_manager_idempotency_plan_hash_nonempty",
        ),
        sa.CheckConstraint(
            "status in ('in_progress', 'committed')",
            name="ck_memory_manager_idempotency_status",
        ),
        sa.PrimaryKeyConstraint("idempotency_key"),
    )
    op.create_index(
        "ix_memory_manager_idempotency_plan_hash",
        "memory_manager_idempotency",
        ["plan_hash"],
        unique=False,
    )
    op.create_index(
        "ix_memory_manager_idempotency_status",
        "memory_manager_idempotency",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_memory_manager_idempotency_status", table_name="memory_manager_idempotency")
    op.drop_index("ix_memory_manager_idempotency_plan_hash", table_name="memory_manager_idempotency")
    op.drop_table("memory_manager_idempotency")
    op.drop_index("idx_memory_semantic_updated_at", table_name="memories")
    op.drop_column("memories", "semantic_updated_at")
