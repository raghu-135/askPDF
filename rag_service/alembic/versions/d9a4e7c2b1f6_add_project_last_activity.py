"""Add project last activity timestamp.

Revision ID: d9a4e7c2b1f6
Revises: f4b8c2d7e1a9
"""

from alembic import op
import sqlalchemy as sa


revision = "d9a4e7c2b1f6"
down_revision = "f4b8c2d7e1a9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        """
        update projects p
           set last_activity_at = greatest(
               p.created_at,
               coalesce((
                   select max(t.created_at)
                     from threads t
                    where t.project_id = p.id
               ), p.created_at),
               coalesce((
                   select max(ct.created_at)
                     from chat_turns ct
                     join threads t on t.id = ct.thread_id
                    where t.project_id = p.id
                      and ct.status <> 'cancelled'
               ), p.created_at),
               coalesce((
                   select max(pf.added_at)
                     from project_files pf
                    where pf.project_id = p.id
               ), p.created_at),
               coalesce((
                   select max(greatest(m.created_at, coalesce(m.updated_at, m.created_at)))
                     from memories m
                    where m.scope_type = 'project'
                      and m.scope_id = p.id
               ), p.created_at)
           )
        """
    )
    op.alter_column(
        "projects",
        "last_activity_at",
        nullable=False,
        server_default=sa.text("now()"),
    )
    op.create_index(
        "idx_project_last_activity_at",
        "projects",
        ["last_activity_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_project_last_activity_at", table_name="projects")
    op.drop_column("projects", "last_activity_at")
