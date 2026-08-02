"""Add explicit memory overrides and simplify canonical memory rows.

Revision ID: 3a8d7c5e1f2b
Revises: 1c7d9e4a2b6f
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "3a8d7c5e1f2b"
down_revision = "1c7d9e4a2b6f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    table_names = set(inspector.get_table_names())
    override_table_existed = "memory_overrides" in table_names
    if not override_table_existed:
        op.create_table(
            "memory_overrides",
            sa.Column("overriding_memory_id", sa.String(), nullable=False),
            sa.Column("overridden_memory_id", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.CheckConstraint(
                "overriding_memory_id <> overridden_memory_id",
                name="ck_memory_overrides_not_self",
            ),
            sa.ForeignKeyConstraint(
                ["overriding_memory_id"], ["memories.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["overridden_memory_id"], ["memories.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("overriding_memory_id", "overridden_memory_id"),
        )
        override_indexes: set[str] = set()
    else:
        override_indexes = {
            item["name"] for item in inspector.get_indexes("memory_overrides")
        }
    if "ix_memory_overrides_target" not in override_indexes:
        op.create_index(
            "ix_memory_overrides_target",
            "memory_overrides",
            ["overridden_memory_id"],
            unique=False,
        )
    if override_table_existed:
        override_columns = {
            item["name"]: item for item in inspector.get_columns("memory_overrides")
        }
        if override_columns["created_at"].get("nullable"):
            op.execute("update memory_overrides set created_at = now() where created_at is null")
            op.alter_column(
                "memory_overrides",
                "created_at",
                existing_type=sa.DateTime(timezone=True),
                nullable=False,
            )

    memory_columns = {item["name"] for item in inspector.get_columns("memories")}
    if "fork_origin_json" in memory_columns:
        op.execute(
            """
            update memories
               set source_refs_json = coalesce(source_refs_json, '{}'::jsonb)
                   || jsonb_build_object('fork_origin', fork_origin_json)
             where fork_origin_json is not null
            """
        )

    constraint_names = {
        item["name"] for item in inspector.get_check_constraints("memories")
    }
    for constraint in (
        "ck_memories_memory_type",
        "ck_memories_status",
        "ck_memories_visibility",
        "ck_memories_confidence",
    ):
        if constraint in constraint_names:
            op.drop_constraint(constraint, "memories", type_="check")
    memory_indexes = {item["name"] for item in inspector.get_indexes("memories")}
    if "idx_memory_scope_status" in memory_indexes:
        op.drop_index("idx_memory_scope_status", table_name="memories")
    for index in (
        "ix_memories_memory_type",
        "ix_memories_status",
        "ix_memories_visibility",
        "ix_memories_created_by",
    ):
        if index in memory_indexes:
            op.drop_index(index, table_name="memories")
    if "idx_memory_scope" not in memory_indexes:
        op.create_index("idx_memory_scope", "memories", ["scope_type", "scope_id"], unique=False)

    for column in (
        "memory_type",
        "summary",
        "confidence",
        "status",
        "visibility",
        "created_by",
        "expires_at",
        "fork_origin_json",
    ):
        if column in memory_columns:
            op.drop_column("memories", column)


def downgrade() -> None:
    op.add_column("memories", sa.Column("fork_origin_json", postgresql.JSONB(), nullable=True))
    op.add_column("memories", sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("memories", sa.Column("created_by", sa.String(), nullable=True))
    op.add_column("memories", sa.Column("visibility", sa.String(), nullable=False, server_default="private"))
    op.add_column("memories", sa.Column("status", sa.String(), nullable=False, server_default="active"))
    op.add_column("memories", sa.Column("confidence", sa.Float(), nullable=False, server_default="1"))
    op.add_column("memories", sa.Column("summary", sa.String(), nullable=False, server_default=""))
    op.add_column("memories", sa.Column("memory_type", sa.String(), nullable=False, server_default="semantic"))
    op.execute(
        """
        update memories
           set fork_origin_json = source_refs_json -> 'fork_origin'
         where source_refs_json ? 'fork_origin'
        """
    )
    op.create_check_constraint(
        "ck_memories_memory_type", "memories",
        "memory_type in ('semantic', 'episodic', 'procedural')",
    )
    op.create_check_constraint("ck_memories_status", "memories", "status = 'active'")
    op.create_check_constraint(
        "ck_memories_visibility", "memories",
        "visibility in ('private', 'project', 'internal')",
    )
    op.create_check_constraint(
        "ck_memories_confidence", "memories", "confidence >= 0 and confidence <= 1"
    )
    op.drop_index("idx_memory_scope", table_name="memories")
    op.create_index("ix_memories_memory_type", "memories", ["memory_type"], unique=False)
    op.create_index("ix_memories_status", "memories", ["status"], unique=False)
    op.create_index("ix_memories_visibility", "memories", ["visibility"], unique=False)
    op.create_index("ix_memories_created_by", "memories", ["created_by"], unique=False)
    op.create_index(
        "idx_memory_scope_status", "memories", ["scope_type", "scope_id", "status"], unique=False
    )
    op.drop_index("ix_memory_overrides_target", table_name="memory_overrides")
    op.drop_table("memory_overrides")
