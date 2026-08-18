"""Remove the retired memory promotion candidate queue.

Revision ID: 1c7d9e4a2b6f
Revises: d9a4e7c2b1f6
"""

from alembic import op
import sqlalchemy as sa


revision = "1c7d9e4a2b6f"
down_revision = "d9a4e7c2b1f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("memory_candidates")


def downgrade() -> None:
    op.create_table(
        "memory_candidates",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("source_thread_id", sa.String(), nullable=True),
        sa.Column("source_project_id", sa.String(), nullable=True),
        sa.Column("source_agent_run_id", sa.String(), nullable=True),
        sa.Column("source_turn_id", sa.String(), nullable=True),
        sa.Column("proposed_scope_type", sa.String(), server_default="thread", nullable=False),
        sa.Column("proposed_scope_id", sa.String(), nullable=False),
        sa.Column("memory_type", sa.String(), server_default="semantic", nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), server_default="0", nullable=False),
        sa.Column("reason", sa.String(), server_default="", nullable=False),
        sa.Column("status", sa.String(), server_default="pending", nullable=False),
        sa.Column("promoted_memory_id", sa.String(), nullable=True),
        sa.Column("resolved_by", sa.String(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("proposed_scope_type in ('user', 'project', 'thread')", name="ck_memory_candidates_scope_type"),
        sa.CheckConstraint("memory_type in ('semantic', 'episodic', 'procedural')", name="ck_memory_candidates_memory_type"),
        sa.CheckConstraint("status in ('pending', 'approved', 'rejected', 'auto_approved')", name="ck_memory_candidates_status"),
        sa.CheckConstraint("confidence >= 0 and confidence <= 1", name="ck_memory_candidates_confidence"),
        sa.CheckConstraint("length(btrim(proposed_scope_id)) > 0", name="ck_memory_candidates_scope_id_nonempty"),
        sa.CheckConstraint("length(btrim(content)) > 0", name="ck_memory_candidates_content_nonempty"),
        sa.ForeignKeyConstraint(["promoted_memory_id"], ["memories.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in (
        "source_thread_id",
        "source_project_id",
        "source_agent_run_id",
        "source_turn_id",
        "proposed_scope_type",
        "proposed_scope_id",
        "memory_type",
        "status",
        "promoted_memory_id",
        "resolved_by",
        "created_by",
    ):
        op.create_index(f"ix_memory_candidates_{column}", "memory_candidates", [column], unique=False)
    op.create_index(
        "idx_memory_candidate_scope_status",
        "memory_candidates",
        ["proposed_scope_type", "proposed_scope_id", "status"],
        unique=False,
    )
    op.create_index(
        "idx_memory_candidate_source_thread",
        "memory_candidates",
        ["source_thread_id", "created_at"],
        unique=False,
    )
