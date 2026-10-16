"""Add thread-owned research canvases."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "b4e1c8a7d2f0"
down_revision = "f9a3c7e1b5d2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "thread_canvases",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("chat_turn_id", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("spec_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("supersedes_id", sa.String(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chat_turn_id"], ["chat_turns.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["supersedes_id"], ["thread_canvases.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("thread_id", "idempotency_key", name="uq_thread_canvases_idempotency"),
    )
    op.create_index("idx_thread_canvas_thread_created", "thread_canvases", ["thread_id", "created_at"])
    op.create_index(op.f("ix_thread_canvases_thread_id"), "thread_canvases", ["thread_id"])
    op.create_index(op.f("ix_thread_canvases_chat_turn_id"), "thread_canvases", ["chat_turn_id"])
    op.create_index(op.f("ix_thread_canvases_supersedes_id"), "thread_canvases", ["supersedes_id"])
    op.create_index(op.f("ix_thread_canvases_idempotency_key"), "thread_canvases", ["idempotency_key"])


def downgrade() -> None:
    op.drop_index(op.f("ix_thread_canvases_idempotency_key"), table_name="thread_canvases")
    op.drop_index(op.f("ix_thread_canvases_supersedes_id"), table_name="thread_canvases")
    op.drop_index(op.f("ix_thread_canvases_chat_turn_id"), table_name="thread_canvases")
    op.drop_index(op.f("ix_thread_canvases_thread_id"), table_name="thread_canvases")
    op.drop_index("idx_thread_canvas_thread_created", table_name="thread_canvases")
    op.drop_table("thread_canvases")
