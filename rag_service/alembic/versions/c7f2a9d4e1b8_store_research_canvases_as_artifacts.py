"""Store research canvases as artifacts instead of a dedicated blob table."""

from alembic import op
import sqlalchemy as sa


revision = "c7f2a9d4e1b8"
down_revision = "b4e1c8a7d2f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_thread_canvases_idempotency_key", table_name="thread_canvases")
    op.drop_index("ix_thread_canvases_supersedes_id", table_name="thread_canvases")
    op.drop_index("ix_thread_canvases_chat_turn_id", table_name="thread_canvases")
    op.drop_index("ix_thread_canvases_thread_id", table_name="thread_canvases")
    op.drop_index("idx_thread_canvas_thread_created", table_name="thread_canvases")
    op.drop_table("thread_canvases")

    op.drop_constraint("ck_agent_task_artifacts_kind", "agent_task_artifacts", type_="check")
    op.alter_column("agent_task_artifacts", "task_id", existing_type=sa.String(), nullable=True)
    op.alter_column("agent_task_artifacts", "agent_run_id", existing_type=sa.String(), nullable=True)
    op.add_column("agent_task_artifacts", sa.Column("thread_id", sa.String(), nullable=True))
    op.add_column("agent_task_artifacts", sa.Column("chat_turn_id", sa.String(), nullable=True))
    op.add_column("agent_task_artifacts", sa.Column("idempotency_key", sa.String(), nullable=True))
    op.create_index("ix_agent_task_artifacts_thread_id", "agent_task_artifacts", ["thread_id"])
    op.create_index("ix_agent_task_artifacts_chat_turn_id", "agent_task_artifacts", ["chat_turn_id"])
    op.create_index("ix_agent_task_artifacts_idempotency_key", "agent_task_artifacts", ["idempotency_key"])
    op.create_index(
        "idx_agent_task_artifact_thread_kind_created",
        "agent_task_artifacts",
        ["thread_id", "kind", "created_at"],
    )
    op.create_index(
        "uq_agent_task_artifact_thread_idempotency",
        "agent_task_artifacts",
        ["thread_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL AND kind = 'research_canvas'"),
    )
    op.create_foreign_key(
        "fk_agent_task_artifacts_thread_id",
        "agent_task_artifacts",
        "threads",
        ["thread_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_agent_task_artifacts_chat_turn_id",
        "agent_task_artifacts",
        "chat_turns",
        ["chat_turn_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "ck_agent_task_artifacts_kind",
        "agent_task_artifacts",
        "kind in ('tool_output','intermediate_report','context_summary','final_report','research_canvas')",
    )
    op.create_check_constraint(
        "ck_agent_task_artifacts_owner",
        "agent_task_artifacts",
        "(task_id IS NOT NULL AND agent_run_id IS NOT NULL) OR (kind = 'research_canvas' AND thread_id IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_agent_task_artifacts_owner", "agent_task_artifacts", type_="check")
    op.drop_constraint("ck_agent_task_artifacts_kind", "agent_task_artifacts", type_="check")
    op.drop_constraint("fk_agent_task_artifacts_chat_turn_id", "agent_task_artifacts", type_="foreignkey")
    op.drop_constraint("fk_agent_task_artifacts_thread_id", "agent_task_artifacts", type_="foreignkey")
    op.drop_index("uq_agent_task_artifact_thread_idempotency", table_name="agent_task_artifacts")
    op.drop_index("idx_agent_task_artifact_thread_kind_created", table_name="agent_task_artifacts")
    op.drop_index("ix_agent_task_artifacts_idempotency_key", table_name="agent_task_artifacts")
    op.drop_index("ix_agent_task_artifacts_chat_turn_id", table_name="agent_task_artifacts")
    op.drop_index("ix_agent_task_artifacts_thread_id", table_name="agent_task_artifacts")
    op.drop_column("agent_task_artifacts", "idempotency_key")
    op.drop_column("agent_task_artifacts", "chat_turn_id")
    op.drop_column("agent_task_artifacts", "thread_id")
    op.alter_column("agent_task_artifacts", "agent_run_id", existing_type=sa.String(), nullable=False)
    op.alter_column("agent_task_artifacts", "task_id", existing_type=sa.String(), nullable=False)
    op.create_check_constraint(
        "ck_agent_task_artifacts_kind",
        "agent_task_artifacts",
        "kind in ('tool_output','intermediate_report','context_summary','final_report')",
    )
    op.create_table(
        "thread_canvases",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("chat_turn_id", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("spec_json", sa.JSON(), nullable=False),
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
    op.create_index("ix_thread_canvases_thread_id", "thread_canvases", ["thread_id"])
    op.create_index("ix_thread_canvases_chat_turn_id", "thread_canvases", ["chat_turn_id"])
    op.create_index("ix_thread_canvases_supersedes_id", "thread_canvases", ["supersedes_id"])
    op.create_index("ix_thread_canvases_idempotency_key", "thread_canvases", ["idempotency_key"])
