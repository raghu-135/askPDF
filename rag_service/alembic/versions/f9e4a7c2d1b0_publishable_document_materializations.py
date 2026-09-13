"""Add publishable materializations and safe document source identifiers.

Revision ID: f9e4a7c2d1b0
Revises: f8d2e4b6c1a3
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "f9e4a7c2d1b0"
down_revision = "f8d2e4b6c1a3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("uq_document_chunk_manifests_target", "document_chunk_manifests", type_="unique")
    op.add_column(
        "document_chunk_manifests",
        sa.Column("expected_source_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
    )
    op.add_column("document_chunk_manifests", sa.Column("published_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("document_chunk_manifests", sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(
        "idx_document_chunk_manifests_published",
        "document_chunk_manifests",
        ["file_hash", "embedding_model", "published_at"],
    )

    op.add_column("document_chunks", sa.Column("source_id", sa.String(), nullable=True))
    op.execute(
        "UPDATE document_chunks SET source_id = 'src_legacy_' || md5(manifest_id || ':' || chunk_id) WHERE source_id IS NULL"
    )
    op.alter_column("document_chunks", "source_id", nullable=False)
    op.drop_constraint("document_chunks_pkey", "document_chunks", type_="primary")
    op.create_primary_key("document_chunks_pkey", "document_chunks", ["manifest_id", "chunk_id"])
    op.create_unique_constraint("uq_document_chunks_source", "document_chunks", ["manifest_id", "source_id"])
    op.create_index("idx_document_chunks_source_id", "document_chunks", ["source_id"])

    # Existing vectors do not carry a manifest/source identity and therefore
    # cannot satisfy the new availability contract. They are repaired lazily.
    op.execute(
        "UPDATE document_chunk_manifests SET vector_status = 'missing', vector_count = 0, published_at = NULL, superseded_at = NULL, expected_source_ids = '[]'::jsonb"
    )


def downgrade() -> None:
    op.drop_index("idx_document_chunks_source_id", table_name="document_chunks")
    op.drop_constraint("uq_document_chunks_source", "document_chunks", type_="unique")
    op.drop_constraint("document_chunks_pkey", "document_chunks", type_="primary")
    op.create_primary_key("document_chunks_pkey", "document_chunks", ["chunk_id"])
    op.drop_column("document_chunks", "source_id")
    op.drop_index("idx_document_chunk_manifests_published", table_name="document_chunk_manifests")
    op.drop_column("document_chunk_manifests", "superseded_at")
    op.drop_column("document_chunk_manifests", "published_at")
    op.drop_column("document_chunk_manifests", "expected_source_ids")
    op.create_unique_constraint(
        "uq_document_chunk_manifests_target",
        "document_chunk_manifests",
        ["file_hash", "embedding_model", "generation", "chunking_fingerprint"],
    )
