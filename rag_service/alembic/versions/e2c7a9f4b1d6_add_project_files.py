"""add project files

Revision ID: e2c7a9f4b1d6
Revises: c9e6a1b4d3f8
"""

from alembic import op
import sqlalchemy as sa


revision = "e2c7a9f4b1d6"
down_revision = "c9e6a1b4d3f8"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "project_files",
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("file_hash", sa.String(), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["file_hash"], ["files.file_hash"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("project_id", "file_hash"),
    )
    op.create_index("idx_project_files_file_hash", "project_files", ["file_hash"])
    op.create_index("idx_project_files_added_at", "project_files", ["added_at"])


def downgrade():
    op.drop_index("idx_project_files_added_at", table_name="project_files")
    op.drop_index("idx_project_files_file_hash", table_name="project_files")
    op.drop_table("project_files")
