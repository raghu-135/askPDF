"""Allow Global memory consistency review state.

Revision ID: 6d2f8a9b3c1e
Revises: 4b7e2d9a1c5f
"""

from alembic import op


revision = "6d2f8a9b3c1e"
down_revision = "4b7e2d9a1c5f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_memory_review_context_type", "memory_review_states", type_="check")
    op.create_check_constraint(
        "ck_memory_review_context_type",
        "memory_review_states",
        "context_type in ('user', 'project', 'thread')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_memory_review_context_type", "memory_review_states", type_="check")
    op.execute("delete from memory_review_states where context_type = 'user'")
    op.create_check_constraint(
        "ck_memory_review_context_type",
        "memory_review_states",
        "context_type in ('project', 'thread')",
    )
