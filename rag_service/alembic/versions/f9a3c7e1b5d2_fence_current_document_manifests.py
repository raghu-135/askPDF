"""Fence current document projection manifests by exact version metadata."""

from alembic import op
import sqlalchemy as sa


revision = "f9a3c7e1b5d2"
down_revision = "f7c9e1a2b3d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_chunk_manifests",
        sa.Column("extraction_fingerprint", sa.String(), nullable=False, server_default=""),
    )
    op.add_column(
        "document_chunk_manifests",
        sa.Column("source_version", sa.String(), nullable=False, server_default=""),
    )
    op.add_column(
        "document_chunk_manifests",
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index(
        "idx_document_chunk_manifests_extraction_fingerprint",
        "document_chunk_manifests",
        ["extraction_fingerprint"],
    )
    op.create_index(
        "idx_document_chunk_manifests_source_version",
        "document_chunk_manifests",
        ["source_version"],
    )
    op.create_index(
        "idx_document_chunk_manifests_current",
        "document_chunk_manifests",
        ["file_hash", "embedding_model", "is_current"],
    )
    op.create_index(
        "uq_document_chunk_manifests_current",
        "document_chunk_manifests",
        ["file_hash", "embedding_model"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )


def downgrade() -> None:
    op.drop_index("uq_document_chunk_manifests_current", table_name="document_chunk_manifests")
    op.drop_index("idx_document_chunk_manifests_current", table_name="document_chunk_manifests")
    op.drop_index("idx_document_chunk_manifests_source_version", table_name="document_chunk_manifests")
    op.drop_index("idx_document_chunk_manifests_extraction_fingerprint", table_name="document_chunk_manifests")
    op.drop_column("document_chunk_manifests", "is_current")
    op.drop_column("document_chunk_manifests", "source_version")
    op.drop_column("document_chunk_manifests", "extraction_fingerprint")
