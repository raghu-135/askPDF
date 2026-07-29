"""add thread document annotation overlays

Revision ID: f4b8c2d7e1a9
Revises: e2c7a9f4b1d6
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "f4b8c2d7e1a9"
down_revision = "e2c7a9f4b1d6"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    if not sa.inspect(connection).has_table("thread_document_annotations"):
        op.create_table(
            "thread_document_annotations",
            sa.Column("thread_id", sa.String(), nullable=False),
            sa.Column("file_hash", sa.String(), nullable=False),
            sa.Column(
                "annotations",
                postgresql.JSONB(astext_type=sa.Text()),
                server_default=sa.text("'[]'::jsonb"),
                nullable=False,
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["file_hash"], ["files.file_hash"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("thread_id", "file_hash"),
        )
    op.execute(
        """
        create index if not exists idx_thread_document_annotations_file_hash
        on thread_document_annotations (file_hash)
        """
    )
    op.execute(
        """
        insert into thread_document_annotations (
            thread_id, file_hash, annotations, created_at, updated_at
        )
        select thread_id, file_hash, annotations, added_at, annotations_updated_at
        from thread_files
        where annotations is not null
          and annotations <> '[]'::jsonb
        on conflict (thread_id, file_hash) do nothing
        """
    )


def downgrade():
    op.execute(
        """
        update thread_files tf
        set annotations = tda.annotations,
            annotations_updated_at = tda.updated_at
        from thread_document_annotations tda
        where tf.thread_id = tda.thread_id
          and tf.file_hash = tda.file_hash
        """
    )
    op.drop_index(
        "idx_thread_document_annotations_file_hash",
        table_name="thread_document_annotations",
    )
    op.drop_table("thread_document_annotations")
